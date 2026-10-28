"""test_part2.py : checks and demos for hamming.py. Run: python test_part2.py"""
import itertools
import numpy as np
from channel import BSCChannel, GilbertElliottChannel
from hamming import (hamming74_encode, hamming74_decode, hamming84_encode, hamming84_decode,
                     interleave, deinterleave)

rng = np.random.default_rng(5)
bits = lambda s: np.array([int(c) for c in s], dtype=np.uint8)
text = lambda a: "".join(map(str, a))


def section(title):
    print("\n" + "=" * 62 + "\n" + title + "\n" + "=" * 62)


section("1. Worked example you can repeat by hand")
d = bits("1011")
cw = hamming74_encode(d)
print("data 1011 -> codeword", text(cw), "(expected 0110011)")
assert text(cw) == "0110011"
rx = cw.copy(); rx[4] ^= 1                                   # flip position 5
print("flip position 5      ->", text(rx))
out, fixed = hamming74_decode(rx)
print("decoder output       ->", text(out), "| corrections made:", fixed)
assert text(out) == "1011"

section("2. Exhaustive checks on all 16 possible data words")
words = [np.array(w, dtype=np.uint8) for w in itertools.product([0, 1], repeat=4)]
ok74 = ok84 = 0
for w in words:
    for p in range(7):
        c = hamming74_encode(w); c[p] ^= 1
        ok74 += np.array_equal(hamming74_decode(c)[0], w)
    for p in range(8):
        c = hamming84_encode(w); c[p] ^= 1
        ok84 += np.array_equal(hamming84_decode(c)[0], w)
print(f"Hamming(7,4) single errors corrected : {ok74} of {16 * 7}")
print(f"Hamming(8,4) single errors corrected : {ok84} of {16 * 8}")
assert ok74 == 112 and ok84 == 128

wrong74 = flagged84 = silent84 = total = 0
for w in words:
    for a, b in itertools.combinations(range(8), 2):
        total += 1
        c = hamming84_encode(w); c[a] ^= 1; c[b] ^= 1
        _, _, dbl = hamming84_decode(c)
        flagged84 += dbl > 0
        if b < 7:                                            # both errors inside the 7 bit part
            c7 = hamming74_encode(w); c7[a] ^= 1; c7[b] ^= 1
            wrong74 += not np.array_equal(hamming74_decode(c7)[0], w)
print(f"Hamming(7,4) double errors giving WRONG data silently : {wrong74} of {16 * 21}")
print(f"Hamming(8,4) double errors DETECTED (flagged)         : {flagged84} of {total}")
assert flagged84 == total

section("3. Burst of L consecutive errors in a 1040 bit frame (200 trials each)")
payload_len = 1040
print(f"{'burst L':>8} | {'no interleaver':>15} | {'with interleaver':>16}   (frames decoded correctly)")
for L in (1, 2, 3, 5, 10, 20, 50, 100):
    good = {False: 0, True: 0}
    for _ in range(200):
        data = rng.integers(0, 2, payload_len, dtype=np.uint8)
        coded = hamming74_encode(data)
        for use_il in (False, True):
            tx = interleave(coded, 7) if use_il else coded.copy()
            start = int(rng.integers(0, tx.size - L))
            tx[start:start + L] ^= 1
            rx = deinterleave(tx, 7) if use_il else tx
            good[use_il] += np.array_equal(hamming74_decode(rx)[0], data)
    print(f"{L:>8} | {good[False] / 200:>15.0%} | {good[True] / 200:>16.0%}")

section("4. Frames delivered with residual errors after decoding (2000 frames each)")
schemes = {
    "no coding":        (lambda b: b,                                  lambda r: r),
    "Hamming(7,4)":     (hamming74_encode,                             lambda r: hamming74_decode(r)[0]),
    "H(7,4)+interleave": (lambda b: interleave(hamming74_encode(b), 7),
                          lambda r: hamming74_decode(deinterleave(r, 7))[0]),
    "H(8,4)+interleave": (lambda b: interleave(hamming84_encode(b), 8),
                          lambda r: hamming84_decode(deinterleave(r, 8))[0]),
}
frames = 2000
for chan_name, make in (("RANDOM errors (BSC)", lambda ber, s: BSCChannel(ber, seed=s)),
                        ("BURSTY errors (Gilbert-Elliott, mean burst 20 bits)",
                         lambda ber, s: GilbertElliottChannel.from_average_ber(ber, 20, seed=s))):
    print("\n" + chan_name)
    print(f"{'BER':>8} | " + " | ".join(f"{n:>18}" for n in schemes))
    for ber in (1e-4, 1e-3, 3e-3, 1e-2):
        row = []
        for name, (enc, dec) in schemes.items():
            ch = make(ber, 11)
            bad = 0
            for _ in range(frames):
                data = rng.integers(0, 2, payload_len, dtype=np.uint8)
                bad += not np.array_equal(dec(ch.transmit(enc(data))), data)
            row.append(f"{bad / frames:>18.1%}")
        print(f"{ber:>8.0e} | " + " | ".join(row))
print("\nAll checks passed.")
