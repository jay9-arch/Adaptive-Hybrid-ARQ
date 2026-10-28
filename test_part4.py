"""test_part4.py : validates arq.py against theory and shows how the protocols differ. Run: python test_part4.py"""
import numpy as np
from channel import BSCChannel
from frame import FrameCodec, CRC_ONLY, HAMMING74
from policy import StaticPolicy
from arq import simulate_arq, ACK_BITS

PROP = 1500                                   # one way propagation delay in bit times
codec = FrameCodec()
L_CRC = codec.frame_bits(1024, CRC_ONLY)      # 1054 bits
RTT = 2 * PROP + ACK_BITS
RTO = RTT + 50


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


class ScriptedChannel:
    """Corrupts exactly the transmissions you name (0 = first frame sent). For demonstrations."""
    def __init__(self, corrupt):
        self.corrupt, self.calls = set(corrupt), 0

    def transmit(self, bits):
        n, self.calls = self.calls, self.calls + 1
        out = bits.copy()
        if n in self.corrupt:
            out[20:23] ^= 1
        return out


section("1. Error free channel: simulation versus formula")
M = 1024 * 300
print(f"frame = {L_CRC} bits, ACK = {ACK_BITS} bits, one way delay = {PROP}, pipe holds about {(L_CRC + RTT) / L_CRC:.1f} frames")
theory_sw = 1024 / (L_CRC + RTT)
theory_win = 1024 / L_CRC
for proto, W, theory, note in (("sw", 1, theory_sw, "payload / (L + 2*delay + ACK)"),
                               ("gbn", 8, theory_win, "payload / L (pipe is full)"),
                               ("sr", 8, theory_win, "payload / L (pipe is full)")):
    r = simulate_arq(proto, W, M, BSCChannel(0.0, seed=1))
    print(f"{proto.upper():<4} W={W}: simulated {r.throughput:.3f} | formula {theory:.3f}  [{note}]")
    assert r.completed and abs(r.throughput - theory) / theory < 0.03

section("2. Stop-and-Wait on a noisy channel (BER 3e-4): formula uses the measured loss rate")
r = simulate_arq("sw", 1, 1024 * 400, BSCChannel(3e-4, seed=2))
p = r.retransmissions / r.transmissions
expected_time = (L_CRC + RTT) + (p / (1 - p)) * (L_CRC + RTO)
theory = 1024 / expected_time
print(f"measured frame loss rate p = {p:.3f}")
print(f"simulated throughput {r.throughput:.3f} | formula {theory:.3f}")
assert abs(r.throughput - theory) / theory < 0.06

section("3. Timeline: ONE corrupted frame (the 2nd one) under each protocol")
small = StaticPolicy(CRC_ONLY, 64)
for proto in ("gbn", "sr"):
    r = simulate_arq(proto, 4, 64 * 8, ScriptedChannel({1}), policy=small, prop=300, trace=34)
    print(f"\n--- {proto.upper()} ---  (window 4, 8 frames, frame 1 is damaged on its first try)")
    print("\n".join(r.trace))
    print(f"transmissions needed: {r.transmissions} for {r.frames} frames | total time {r.sim_time:.0f}")

section("4. Protocol comparison, random errors, throughput (1.0 = perfect use of the link)")
M = 1024 * 150
for mode, bers in ((CRC_ONLY, (1e-5, 1e-4, 3e-4, 1e-3)), (HAMMING74, (1e-4, 1e-3, 3e-3, 1e-2))):
    print(f"\nmode: {'CRC only' if mode == CRC_ONLY else 'Hamming(7,4)+CRC'}   (GBN window 16, SR window 16)")
    print(f"{'BER':>8} | {'Stop-Wait':>9} | {'Go-Back-N':>9} | {'Sel.Repeat':>10} | transmissions per frame (SW / GBN / SR)")
    for ber in bers:
        rows = []
        for proto, W in (("sw", 1), ("gbn", 16), ("sr", 16)):
            r = simulate_arq(proto, W, M, BSCChannel(ber, seed=3), policy=StaticPolicy(mode, 1024))
            assert r.completed and r.undetected_errors == 0 and r.delivered_bits == M
            rows.append(r)
        tpf = " / ".join(f"{r.transmissions / r.frames:5.2f}" for r in rows)
        print(f"{ber:>8.0e} | {rows[0].throughput:>9.3f} | {rows[1].throughput:>9.3f} | {rows[2].throughput:>10.3f} | {tpf}")

section("5. Effect of window size (Selective Repeat, CRC only, BER 1e-4)")
print(f"{'window':>7} | throughput")
for W in (1, 2, 4, 8, 16, 32):
    r = simulate_arq("sr", W, 1024 * 300, BSCChannel(1e-4, seed=5))
    print(f"{W:>7} | {r.throughput:.3f}")
print("The curve flattens once the window covers the pipe (about 5 frames here).")
print("\nAll checks passed.")
