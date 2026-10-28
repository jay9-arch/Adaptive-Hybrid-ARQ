"""test_part3.py : checks and demos for frame.py. Run: python test_part3.py"""
import numpy as np
from channel import BSCChannel, GilbertElliottChannel
from frame import FrameCodec, CRC_ONLY, HAMMING74, HAMMING84, MODE_NAMES

rng = np.random.default_rng(21)
codec = FrameCodec()
PAYLOAD = 1024


def section(title):
    print("\n" + "=" * 66 + "\n" + title + "\n" + "=" * 66)


section("1. Clean channel round trip for every mode")
for mode in (CRC_ONLY, HAMMING74, HAMMING84):
    payload = rng.integers(0, 2, PAYLOAD, dtype=np.uint8)
    tx = codec.encode(77, payload, mode)
    r = codec.decode(tx)
    print(f"{MODE_NAMES[mode]:<18} bits on air: {tx.size:>5} | accepted: {r.ok} | seq: {r.seq} | "
          f"payload intact: {np.array_equal(r.payload, payload)}")
    assert r.ok and r.seq == 77 and np.array_equal(r.payload, payload)

section("2. Overhead of each mode (1024 bit payload, no errors at all)")
print(f"{'mode':<18} | {'bits on air':>11} | {'efficiency':>10}")
for mode in (CRC_ONLY, HAMMING74, HAMMING84):
    print(f"{MODE_NAMES[mode]:<18} | {codec.frame_bits(PAYLOAD, mode):>11} | {codec.overhead_efficiency(PAYLOAD, mode):>10.3f}")

section("3. The receiver survives a damaged mode field (no crash, frame rejected)")
payload = rng.integers(0, 2, PAYLOAD, dtype=np.uint8)
tx = codec.encode(5, payload, HAMMING74)
bad = tx.copy(); bad[0] ^= 1                                   # 1 error in mode field: majority vote fixes it
print("1 bit error in mode field  -> accepted:", codec.decode(bad).ok)
bad = tx.copy(); bad[0] ^= 1; bad[1] ^= 1                      # 2 of 3 copies wrong: mode misread
r = codec.decode(bad)
print("2 bit errors in same group -> accepted:", r.ok, "| misread mode:", r.mode, "| reason:", r.reason)
assert not r.ok

section("4. Preview of ARQ efficiency (each rejected frame is simply sent again)")
print("efficiency = (useful bits / bits on air) x (fraction of frames accepted)")
print("This ignores waiting time for ACKs. Part 4 will add the real protocols.\n")
modes = (CRC_ONLY, HAMMING74, HAMMING84)
frames = 600
for chan_name, make in (("RANDOM errors (BSC)", lambda ber, s: BSCChannel(ber, seed=s)),
                        ("BURSTY errors (Gilbert-Elliott, mean burst 20 bits)",
                         lambda ber, s: GilbertElliottChannel.from_average_ber(ber, 20, seed=s))):
    print(chan_name)
    print(f"{'BER':>8} | " + " | ".join(f"{MODE_NAMES[m]:>18}" for m in modes) + " | best mode")
    for ber in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        effs, wrong_accepts = [], 0
        for m in modes:
            ch = make(ber, 4)
            acc = 0
            for i in range(frames):
                payload = rng.integers(0, 2, PAYLOAD, dtype=np.uint8)
                r = codec.decode(ch.transmit(codec.encode(i % 256, payload, m)))
                if r.ok:
                    acc += 1
                    wrong_accepts += (not np.array_equal(r.payload, payload)) or r.seq != i % 256
            effs.append(codec.overhead_efficiency(PAYLOAD, m) * acc / frames)
        best = MODE_NAMES[modes[int(np.argmax(effs))]]
        print(f"{ber:>8.0e} | " + " | ".join(f"{e:>18.3f}" for e in effs) + f" | {best}")
    print(f"(frames accepted with wrong content, i.e. missed by CRC: {wrong_accepts})\n")
print("All checks passed.")
