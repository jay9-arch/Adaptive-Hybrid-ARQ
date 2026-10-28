"""
arq.py : event driven simulator of Go-Back-N and Selective Repeat (Stop-and-Wait = window 1).

TIME is measured in bit times: the link sends 1 bit per time unit in each direction (full duplex).
    sending a frame of L bits takes L time units
    one way propagation delay = prop (a number, or a function of time) plus random jitter
    an ACK has 32 bits: [ seq 8 | status 8 | CRC-16 16 ]
The status byte carries what the receiver saw: fresh flag, 2 error flag, number of Hamming
corrections. The sender hands these to the policy, which is how the adaptive scheme will learn.

PROTOCOL RULES
  Go-Back-N : receiver accepts only the next expected frame and sends a cumulative ACK.
              When the oldest frame times out, the sender resends EVERYTHING outstanding.
  Selective Repeat : receiver buffers out of order frames and ACKs each one individually.
              Each frame has its own timer and only the frame that timed out is resent.
  Stop-and-Wait    : Go-Back-N with a window of 1.

A frame is only accepted by the receiver if its CRC passes. Rejected frames get no reply, so the
sender finds out through its timer.
"""
import heapq
from collections import deque
from dataclasses import dataclass, field
import numpy as np

from crc import CRC16
from frame import FrameCodec, MODE_NAMES, CRC_ONLY
from rto import FixedRTO
from policy import StaticPolicy

SEQ_SPACE = 256
ACK_BITS = 32


@dataclass
class ArqResult:
    protocol: str
    window: int
    completed: bool
    sim_time: float                 # time at which the last frame was delivered, in bit times
    message_bits: int
    delivered_bits: int
    frames: int                     # number of distinct frames created
    transmissions: int              # frames put on the wire, including retransmissions
    bits_sent: int                  # total bits put on the forward link
    timeouts: int = 0
    acks_lost: int = 0
    rejected: int = 0               # frames the receiver threw away (CRC or mode field failed)
    undetected_errors: int = 0      # frames accepted although their payload was wrong
    corrections: int = 0            # Hamming corrections made by the receiver
    mode_tx: dict = field(default_factory=dict)      # transmissions per mode
    delays: list = field(default_factory=list)       # per frame: first send to in order delivery
    trace: list = field(default_factory=list)

    @property
    def throughput(self):
        """Useful payload bits delivered per bit time of the link (1.0 = perfect use of the link)."""
        return self.delivered_bits / self.sim_time if self.sim_time else 0.0

    @property
    def bit_efficiency(self):
        """Useful bits divided by all bits sent (ignores idle time)."""
        return self.delivered_bits / self.bits_sent if self.bits_sent else 0.0

    @property
    def retransmissions(self):
        return self.transmissions - self.frames

    @property
    def spurious(self):
        """Retransmissions beyond what the channel really destroyed: they came from timers that fired too early."""
        return max(0, self.retransmissions - self.rejected)

    @property
    def avg_delay(self):
        return float(np.mean(self.delays)) if self.delays else 0.0

    def summary(self):
        usage = ", ".join(f"{MODE_NAMES[m]}: {n}" for m, n in sorted(self.mode_tx.items()))
        return (f"{self.protocol.upper()} W={self.window} | completed={self.completed} | "
                f"throughput={self.throughput:.3f} | bit efficiency={self.bit_efficiency:.3f} | "
                f"retransmissions={self.retransmissions} | avg delay={self.avg_delay:.0f} | "
                f"undetected={self.undetected_errors} | {usage}")


def simulate_arq(protocol, window, message_bits, fwd_channel, codec=None, policy=None, rto=None,
                 prop=1500, jitter=0, jitter_dist="uniform", rev_channel=None, seed=0, abort_factor=30, trace=0):
    label = protocol
    if protocol == "sw":
        protocol, window = "gbn", 1
    if protocol not in ("gbn", "sr"):
        raise ValueError("protocol must be 'sw', 'gbn' or 'sr'")
    if window < 1 or (protocol == "gbn" and window > SEQ_SPACE - 1) or (protocol == "sr" and window > SEQ_SPACE // 2):
        raise ValueError("window too large for the 8 bit sequence number space")
    if message_bits % 8:
        raise ValueError("message_bits must be a multiple of 8")

    rng = np.random.default_rng(seed)
    codec = codec or FrameCodec()
    policy = policy or StaticPolicy(CRC_ONLY, 1024)
    prop_fn = prop if callable(prop) else (lambda t: prop)      # one way delay may change with time
    rto = rto or FixedRTO(2 * (prop_fn(0.0) + jitter) + ACK_BITS + 50)

    res = ArqResult(label, window, False, 0.0, message_bits, 0, 0, 0, 0)

    # ---- sender state, indexed by LOGICAL frame number (the wire only carries it modulo 256)
    payloads, first_start, n_tx, version, tx_end, tx_mode, acked = [], [], [], [], [], [], []
    tx_len = []
    created_bits = 0
    base = nxt = 0
    resend, in_resend = deque(), set()
    line_free, wake_pending = 0.0, False
    # ---- receiver state (the two counters live in lists so the nested functions can change them)
    rcv_next_ref, rcv_base_ref = [0], [0]
    rcv_buf = {}
    state = {"done": False, "now": 0.0}

    heap, counter = [], 0

    def push(t, kind, data=None):
        nonlocal counter
        counter += 1
        heapq.heappush(heap, (t, counter, kind, data))

    def log(t, text):
        if len(res.trace) < trace:
            res.trace.append(f"t={t:>7.0f}  {text}")

    def delay(t):
        """One way delay of a packet that starts travelling at time t (mean extra delay = jitter / 2)."""
        extra = 0.0
        if jitter:
            extra = rng.uniform(0, jitter) if jitter_dist == "uniform" else rng.exponential(jitter / 2)
        return prop_fn(t) + extra

    # ------------------------------------------------------------------ receiver
    def deliver(idx, now):
        res.delivered_bits += len(payloads[idx])
        res.delays.append(now - first_start[idx])
        log(now, f"            RX delivers frame {idx} to the application")
        if res.delivered_bits >= message_bits:
            state["done"] = True
            res.completed = True
            res.sim_time = now

    def send_ack(now, seq, fresh, r=None):
        corrected = r.corrected if (r is not None and fresh) else 0
        double = 1 if (r is not None and fresh and r.double_detected) else 0
        status = fresh | (double << 1) | (min(corrected, 63) << 2)
        ack = CRC16.encode(np.unpackbits(np.array([seq, status], dtype=np.uint8)))
        ack_rx = rev_channel.transmit(ack) if rev_channel is not None else ack
        log(now, f"            RX sends ACK {seq}{'' if fresh else ' (repeat)'}")
        push(now + ACK_BITS + delay(now + ACK_BITS), "ACK", ack_rx)

    def on_rx(now, rx_bits):
        r = codec.decode(rx_bits)
        if not r.ok:
            res.rejected += 1
            log(now, f"RX frame REJECTED ({r.reason})")
            return
        res.corrections += r.corrected
        s = r.seq
        if protocol == "gbn":
            off = (s - rcv_next_ref[0]) % SEQ_SPACE
            if off == 0:
                idx = rcv_next_ref[0]
                if idx >= len(payloads):
                    res.undetected_errors += 1
                    return
                if not np.array_equal(r.payload, payloads[idx]):
                    res.undetected_errors += 1
                rcv_next_ref[0] += 1
                send_ack(now, s, 1, r)
                deliver(idx, now)
            else:
                log(now, f"RX frame {s} out of order, discarded")
                if rcv_next_ref[0] > 0:
                    send_ack(now, (rcv_next_ref[0] - 1) % SEQ_SPACE, 0)
        else:
            off = (s - rcv_base_ref[0]) % SEQ_SPACE
            if off < window:
                idx = rcv_base_ref[0] + off
                if idx >= len(payloads):
                    res.undetected_errors += 1
                    return
                if idx not in rcv_buf:
                    if not np.array_equal(r.payload, payloads[idx]):
                        res.undetected_errors += 1
                    rcv_buf[idx] = True
                    send_ack(now, s, 1, r)
                else:
                    send_ack(now, s, 0)
                while rcv_base_ref[0] in rcv_buf:
                    del rcv_buf[rcv_base_ref[0]]
                    deliver(rcv_base_ref[0], now)
                    rcv_base_ref[0] += 1
                    if state["done"]:
                        break
            elif off >= SEQ_SPACE - window:
                send_ack(now, s, 0)

    # -------------------------------------------------------------------- sender
    def try_send(now):
        nonlocal line_free, wake_pending, nxt, created_bits
        if state["done"]:
            return
        if now < line_free:
            if not wake_pending:
                push(line_free, "WAKE")
                wake_pending = True
            return
        idx = None
        while resend:                                    # retransmissions have priority
            cand = resend.popleft()
            in_resend.discard(cand)
            if cand >= base and not acked[cand]:
                idx = cand
                break
        limit = min(window, policy.window_limit()) if hasattr(policy, "window_limit") else window
        if idx is None and created_bits < message_bits and nxt - base < limit:
            size = min(policy.new_frame_payload(), message_bits - created_bits)
            payloads.append(rng.integers(0, 2, size, dtype=np.uint8))
            first_start.append(now); n_tx.append(0); version.append(0)
            tx_end.append(0.0); tx_mode.append(CRC_ONLY); acked.append(False); tx_len.append(0)
            created_bits += size
            res.frames += 1
            idx = nxt
            nxt += 1
        if idx is None:
            return                                       # nothing allowed to be sent right now

        mode = policy.tx_mode(idx, n_tx[idx])
        tx = codec.encode(idx % SEQ_SPACE, payloads[idx], mode)
        rx_bits = fwd_channel.transmit(tx)
        L = tx.size
        line_free = now + L
        n_tx[idx] += 1
        version[idx] += 1
        tx_end[idx], tx_mode[idx], tx_len[idx] = now + L, mode, L
        res.transmissions += 1
        res.bits_sent += L
        res.mode_tx[mode] = res.mode_tx.get(mode, 0) + 1
        log(now, f"TX frame {idx} (try {n_tx[idx]}, {MODE_NAMES[mode]}, {L} bits)")
        push(now + L + delay(now + L), "RX", rx_bits)
        push(now + L + rto.value(), "TO", (idx, version[idx]))
        push(line_free, "WAKE")
        wake_pending = True
        if res.bits_sent > abort_factor * message_bits:
            state["done"] = True                         # hopeless channel: give up
            res.sim_time = now + L

    def on_ack(now, ack_rx):
        nonlocal base
        if not CRC16.check(ack_rx):
            res.acks_lost += 1
            log(now, "TX side: ACK corrupted, ignored")
            return
        b = np.packbits(ack_rx[:16]).tobytes()
        seq, status = b[0], b[1]
        fresh, double, corrected = status & 1, (status >> 1) & 1, status >> 2
        off = (seq - base) % SEQ_SPACE
        if off < nxt - base:
            idx = base + off
            if protocol == "gbn":
                newly = range(base, idx + 1)
            else:
                newly = [idx] if not acked[idx] else []
            for j in newly:
                acked[j] = True
                version[j] += 1                          # cancels its timer
            if newly:
                log(now, f"TX side: ACK {seq} received")
                if n_tx[idx] == 1:
                    rto.on_sample(now - tx_end[idx])          # Karn: only never-retransmitted frames
                else:
                    rto.on_progress()
                if fresh:
                    policy.on_ack(tx_mode[idx], corrected, double, tx_len[idx], n_tx[idx] - 1)
            while base < nxt and acked[base]:
                base += 1
        try_send(now)

    def on_timeout(now, idx, ver):
        if idx < base or acked[idx] or ver != version[idx]:
            return                                       # stale timer
        res.timeouts += 1
        rto.on_timeout(now)
        policy.on_timeout(tx_mode[idx], tx_len[idx], n_tx[idx] - 1)
        log(now, f"TX side: TIMEOUT for frame {idx}")
        if protocol == "gbn":
            for j in range(base, nxt):
                version[j] += 1
            resend.clear(); in_resend.clear()
            resend.extend(range(base, nxt)); in_resend.update(range(base, nxt))
        else:
            version[idx] += 1
            if idx not in in_resend:
                resend.append(idx); in_resend.add(idx)
        try_send(now)

    # ---------------------------------------------------------------- event loop
    push(0.0, "WAKE")
    wake_pending = True
    while heap and not state["done"]:
        now, _, kind, data = heapq.heappop(heap)
        if kind == "WAKE":
            wake_pending = False
            try_send(now)
        elif kind == "RX":
            on_rx(now, data)
        elif kind == "ACK":
            on_ack(now, data)
        else:
            on_timeout(now, *data)
    return res
