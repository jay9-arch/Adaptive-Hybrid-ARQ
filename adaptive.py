"""
adaptive.py : the proposed Adaptive Hybrid ARQ controller (the heart of the project).

IDEA
  Pick, for every new frame, the (error control mode, payload size) pair with the highest
  EXPECTED EFFICIENCY, using what the sender has learned about the channel from ACKs and timeouts.

      efficiency(mode, P) = (P / bits on air) x P(frame accepted)

MODEL (one number per mode, learned online)
  CRC only    : P(accepted) = exp(-hc x L)    L = bits on air      hc = hazard per bit
  Hamming(7,4): P(accepted) = exp(-q x nb)    nb = 7 bit blocks    q  = hazard per block
  A bigger hazard means a worse channel. Shrinking the frame raises the acceptance probability,
  but the fixed header and CRC then cost more, so there is a best size for every hazard.

LEARNING
  * Only the FIRST attempt of each frame is used (retries follow a failure and would bias it).
  * Each mode keeps a sliding window of recent outcomes. Hazard = -ln(1 - failure rate) / mean exposure.
  * A mode that has not been used recently is estimated from the other mode with a random error
    model (cross estimate). A small fraction of frames (probe) are sent in the other mode so that
    both estimates stay fresh. This matters on bursty channels where the random model is wrong.

STABILITY
  * Hysteresis: the configuration only changes if the best candidate beats the current one by `margin`.
  * Escalation: a frame that already failed `escalate_after` times is retried in Hamming mode.

WINDOW
  window_limit() sizes the Selective Repeat window from the bandwidth delay product, so that
  small frames still keep the link busy.
"""
import math
from collections import deque
import numpy as np

from frame import CRC_ONLY, HAMMING74, MODE_FIELD_BITS

SIZES = (64, 128, 256, 512, 1024, 2048)


class AdaptivePolicy:
    name = "adaptive"

    def __init__(self, codec, sizes=SIZES, adapt_mode=True, adapt_size=True,
                 fixed_mode=CRC_ONLY, fixed_size=1024, rto=None, rtt_hint=3500, max_window=128,
                 history=150, short=24, stale_after=1000, probe=0.05, margin=1.03, escalate_after=2,
                 n_min=6, prior_ber=1e-4, seed=0):
        self.codec = codec
        self.sizes = tuple(sizes) if adapt_size else (fixed_size,)
        self.modes = (CRC_ONLY, HAMMING74) if adapt_mode else (fixed_mode,)
        self.adapt_mode, self.adapt_size = adapt_mode, adapt_size
        self.rto, self.rtt_hint, self.max_window = rto, rtt_hint, max_window
        self.history, self.short, self.stale_after = history, short, stale_after
        self.probe, self.margin, self.escalate_after = probe, margin, escalate_after
        self.n_min, self.prior_ber = n_min, prior_ber
        self.rng = np.random.default_rng(seed)

        self.hist = {CRC_ONLY: deque(maxlen=history), HAMMING74: deque(maxlen=history)}
        self.tick = 0
        self.current = (self.modes[0], self.sizes[len(self.sizes) // 2])
        self.frame_mode = {}
        self._pending = self.current[0]
        self._cur_bits = codec.frame_bits(self.current[1], self.current[0])
        self.switches = 0

    # ------------------------------------------------------------------ model
    def _exposure(self, mode, bits):
        return bits if mode == CRC_ONLY else (bits - MODE_FIELD_BITS) / 7.0

    def _hazard(self, mode):
        """Learned hazard of `mode`, or None if there is not enough recent data."""
        h = self.hist[mode]
        while h and self.tick - h[0][0] > self.stale_after:
            h.popleft()
        if len(h) < self.n_min:
            return None

        def hazard_of(entries):
            n = len(entries)
            fails = sum(1 for _, _, ok in entries if not ok)
            f = min(max(fails / n, 0.5 / n), 1 - 0.5 / n)
            return -math.log(1 - f) / (sum(e for _, e, _ in entries) / n), fails

        entries = list(h)
        long_h, _ = hazard_of(entries)
        recent = entries[-self.short:]
        short_h, short_fails = hazard_of(recent)
        # Trust the short window only when its failure count is clearly above what the long
        # window predicts (about 2 standard deviations), so random noise does not cause false alarms.
        f_long = sum(1 for _, _, ok in entries if not ok) / len(entries)
        expected = len(recent) * f_long
        threshold = expected + 2.0 * math.sqrt(max(expected * (1 - f_long), 0.25)) + 1
        return max(long_h, short_h) if short_fails > threshold else long_h

    @staticmethod
    def _crc_to_ham(hc):
        """Random error model: per bit hazard -> per block hazard of Hamming(7,4)."""
        b = 1 - math.exp(-hc)
        p_block_ok = (1 - b) ** 7 + 7 * b * (1 - b) ** 6
        return -math.log(max(p_block_ok, 1e-12))

    @staticmethod
    def _ham_to_crc(q):
        """Inverse of the above: block failure probability ~ 21 b^2."""
        p_fail = 1 - math.exp(-q)
        b = math.sqrt(p_fail / 21.0)
        return -math.log(1 - min(b, 0.49))

    def estimates(self):
        hc, qh = self._hazard(CRC_ONLY), self._hazard(HAMMING74)
        if hc is None and qh is None:
            hc = -math.log(1 - self.prior_ber)
            qh = self._crc_to_ham(hc)
        elif hc is None:
            hc = self._ham_to_crc(qh)
        elif qh is None:
            qh = self._crc_to_ham(hc)
        return hc, qh

    def _rtt_timeout(self):
        rtt = getattr(self.rto, "srtt", None) or self.rtt_hint
        timeout = getattr(self.rto, "base", None) or 1.4 * self.rtt_hint
        return rtt, timeout

    def _window_for(self, bits):
        """Window that covers a round trip + timeout for up to two failures in a row, capped by the sequence space."""
        rtt, timeout = self._rtt_timeout()
        return max(2, min(self.max_window, math.ceil(3.0 * (rtt + timeout + bits) / bits) + 2))

    def predicted_efficiency(self, mode, payload, hc, qh):
        """
        Expected useful bits per bit time of link, for one frame:   payload / (time the link spends per frame)
        time per frame = attempts x frame time + idle time caused by window stalls.
        A frame that fails K times in a row keeps the window head busy for K x R, where
        R = round trip + timeout + frame time. If that exceeds what the window can cover, the link idles.
        """
        bits = self.codec.frame_bits(payload, mode)
        exposure = self._exposure(mode, bits)
        ok = math.exp(-(hc if mode == CRC_ONLY else qh) * exposure)
        f = 1.0 - ok
        rtt, timeout = self._rtt_timeout()
        reach = self._window_for(bits) * bits              # data the window can send while the head waits
        recovery = rtt + timeout + bits
        idle, k = 0.0, 1
        while k < 40:
            weight = (f ** k) * ok                          # exactly k failures, then a success
            if weight < 1e-7:
                break
            idle += weight * max(0.0, k * recovery - reach)
            k += 1
        return payload / (bits / ok + idle)

    def _best(self, modes, hc, qh):
        return max(((m, p) for m in modes for p in self.sizes),
                   key=lambda c: self.predicted_efficiency(c[0], c[1], hc, qh))

    # --------------------------------------------------------------- decisions
    def _decide(self):
        hc, qh = self.estimates()
        best = self._best(self.modes, hc, qh)
        cur_eff = self.predicted_efficiency(*self.current, hc, qh)
        if best != self.current and self.predicted_efficiency(*best, hc, qh) > self.margin * cur_eff:
            self.current = best
            self.switches += 1
        return self.current

    def new_frame_payload(self):
        mode, payload = self._decide()
        if self.adapt_mode and mode == HAMMING74 and self.rng.random() < self.probe:
            hc, qh = self.estimates()                               # probe CRC only mode with a small frame
            mode, payload = self._best((CRC_ONLY,), hc, qh)
        self._pending = mode
        self._cur_bits = self.codec.frame_bits(payload, mode)
        return payload

    def tx_mode(self, idx, attempt):
        if attempt == 0:
            self.frame_mode[idx] = self._pending
            return self._pending
        base = self.frame_mode.get(idx, self.current[0])
        if self.adapt_mode and attempt >= self.escalate_after:
            return HAMMING74
        return base

    def window_limit(self):
        """
        After a loss the window cannot slide until the lost frame is timed out, resent and
        acknowledged, so it must cover a round trip PLUS a timeout (for two failures in a row),
        not just the round trip. Large frames need fewer frames in flight than small ones.
        """
        return self._window_for(self.codec.frame_bits(self.current[1], self.current[0]))

    # ---------------------------------------------------------------- feedback
    def _record(self, mode, bits, ok, attempt):
        self.tick += 1
        if attempt == 0 and mode in self.hist and bits:
            self.hist[mode].append((self.tick, self._exposure(mode, bits), ok))

    def on_ack(self, mode, corrected, double, bits=0, attempt=0):
        self._record(mode, bits, True, attempt)

    def on_timeout(self, mode, bits=0, attempt=0):
        self._record(mode, bits, False, attempt)
