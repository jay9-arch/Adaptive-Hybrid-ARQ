"""test_part1.py : checks and small demos for channel.py and crc.py. Run: python test_part1.py"""
import numpy as np
from channel import BSCChannel, GilbertElliottChannel, count_bit_errors
from crc import CRC, CRC16, CRC32, crc_remainder_bitwise

rng = np.random.default_rng(1)


def section(title):
    print("\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60)


# ---------------------------------------------------------------- CRC
section("1. CRC : textbook example and standard check values")
print("data 100100, generator 1101 -> remainder", crc_remainder_bitwise([1, 0, 0, 1, 0, 0], [1, 1, 0, 1]))
assert crc_remainder_bitwise([1, 0, 0, 1, 0, 0], [1, 1, 0, 1]) == [0, 0, 1]

print("CRC-16 of '123456789' = 0x%04X (standard check value 0x29B1)" % CRC16.compute(b"123456789"))
print("CRC-32 of '123456789' = 0x%08X (standard check value 0xCBF43926)" % CRC32.compute(b"123456789"))
assert CRC16.compute(b"123456789") == 0x29B1
assert CRC32.compute(b"123456789") == 0xCBF43926

plain16 = CRC(16, 0x1021, 0, 0, False)
data = rng.integers(0, 2, 64, dtype=np.uint8)
fast = plain16.compute(np.packbits(data).tobytes())
slow = int("".join(map(str, crc_remainder_bitwise(data, [1] + [int(b) for b in format(0x1021, "016b")]))), 2)
print("fast table CRC == bitwise long division CRC :", fast == slow)
assert fast == slow

section("2. CRC-16 detection power on a 1024 bit payload")
frame = CRC16.encode(rng.integers(0, 2, 1024, dtype=np.uint8))
print("frame length with CRC:", frame.size, "bits | clean frame passes:", CRC16.check(frame))
missed = 0
for pos in range(frame.size):                      # every possible single bit error
    bad = frame.copy(); bad[pos] ^= 1
    missed += CRC16.check(bad)
print("single bit errors missed:", missed, "of", frame.size)
missed = 0
for _ in range(2000):                              # random bursts up to 16 bits long
    L = int(rng.integers(2, 17)); start = int(rng.integers(0, frame.size - L))
    bad = frame.copy(); bad[start] ^= 1; bad[start + L - 1] ^= 1
    bad[start + 1:start + L - 1] ^= rng.integers(0, 2, L - 2, dtype=np.uint8)
    missed += CRC16.check(bad)
print("burst errors (length <= 16) missed:", missed, "of 2000")
assert missed == 0

# ------------------------------------------------------------ channels
section("3. Channel models : measured BER versus requested BER")
n = 2_000_000
zeros = np.zeros(n, dtype=np.uint8)
for ber in (1e-4, 1e-3, 1e-2):
    ch = BSCChannel(ber, seed=7)
    print(f"BSC  requested {ber:.0e} -> measured {count_bit_errors(zeros, ch.transmit(zeros)) / n:.2e}")
    ge = GilbertElliottChannel.from_average_ber(ber, mean_burst=20, seed=7)
    print(f"GE   requested {ber:.0e} -> measured {count_bit_errors(zeros, ge.transmit(zeros)) / n:.2e}")

section("4. Why burst errors matter : same average BER, different frame error rate")
frame_bits, frames = 1040, 5000
for ber in (1e-3, 3e-3):
    results = {}
    for name, ch in (("random (BSC)", BSCChannel(ber, seed=3)),
                     ("bursty (GE) ", GilbertElliottChannel.from_average_ber(ber, mean_burst=20, seed=3))):
        bad = sum(count_bit_errors(np.zeros(frame_bits, np.uint8), ch.transmit(np.zeros(frame_bits, np.uint8))) > 0
                  for _ in range(frames))
        results[name] = bad / frames
    print(f"average BER {ber:.0e} : " + " | ".join(f"{k} frames hit = {v:.1%}" for k, v in results.items()))
print("\nAll checks passed.")
