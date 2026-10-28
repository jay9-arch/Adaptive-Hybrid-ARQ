"""test_part7.py : comparison with the reference paper's rule. Run: python test_part7.py  (about 1 minute)"""
import numpy as np
from frame import FrameCodec, CRC_ONLY, HAMMING74
from policy import StaticPolicy
from arq import simulate_arq
from rto import AdaptiveRTO
from adaptive import AdaptivePolicy
from goga import GogaPolicy
from scenarios import make_channel, four_phase_channel

codec = FrameCodec()
SEEDS = (1, 2, 3)


def section(title):
    print("\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80)


section("1. The ACK/NAK counting rule, step by step")
g = GogaPolicy(fail_threshold=2, window=8, success_threshold=4)
steps = [True, True, False, True, False]          # two failures within the last 8 -> go hybrid
for ok in steps:
    g._outcome(ok)
print("after outcomes", ["ACK" if o else "NAK" for o in steps], "-> mode:", "hybrid" if g.mode == HAMMING74 else "pure ARQ")
assert g.mode == HAMMING74
for _ in range(3):
    g._outcome(True)
print("after 3 more ACKs -> mode:", "hybrid" if g.mode == HAMMING74 else "pure ARQ", "(needs 4 in a row)")
assert g.mode == HAMMING74
g._outcome(True)
print("after the 4th ACK  -> mode:", "hybrid" if g.mode == HAMMING74 else "pure ARQ")
assert g.mode == CRC_ONLY

section("2. Stationary channels (throughput, 1.0 = perfect, average of 3 runs)")
schemes = {
    "GBN + CRC":                 lambda r, s: ("gbn", 16, StaticPolicy(CRC_ONLY, 1024)),
    "SR + CRC":                  lambda r, s: ("sr", 32, StaticPolicy(CRC_ONLY, 1024)),
    "SR + Hamming (fixed)":      lambda r, s: ("sr", 32, StaticPolicy(HAMMING74, 1024)),
    "Reference rule, GBN":       lambda r, s: ("gbn", 16, GogaPolicy()),
    "Reference rule, SR":        lambda r, s: ("sr", 32, GogaPolicy()),
    "PROPOSED ADAPTIVE (SR)":    lambda r, s: ("sr", 128, AdaptivePolicy(codec, rto=r, seed=s)),
}
cases = [("rand", 1e-4), ("rand", 1e-3), ("rand", 3e-3), ("rand", 1e-2), ("burst", 1e-3), ("burst", 1e-2)]
table = {}
for name, make in schemes.items():
    row = []
    for kind, ber in cases:
        vals = []
        for seed in SEEDS:
            r_obj = AdaptiveRTO(5082)
            proto, W, pol = make(r_obj, seed)
            vals.append(simulate_arq(proto, W, 1024 * 200, make_channel(kind, ber, seed), policy=pol,
                                     rto=r_obj, seed=seed, abort_factor=20).throughput)
        row.append(np.mean(vals))
    table[name] = row
head = "".join(f"{('random' if k == 'rand' else 'bursty')[:4]} {b:.0e}".rjust(12) for k, b in cases)
print(f"{'':<24}|{head} |   mean")
for name, row in table.items():
    print(f"{name:<24}|" + "".join(f"{v:>12.3f}" for v in row) + f" | {np.mean(row):.3f}")
ours, ref = table["PROPOSED ADAPTIVE (SR)"], table["Reference rule, GBN"]
ref_sr = table["Reference rule, SR"]
print(f"\nproposed vs reference rule (GBN, as in the paper): {np.mean(ours) / np.mean(ref) - 1:+.0%} on average")
print(f"proposed vs reference rule (SR, same protocol)   : {np.mean(ours) / np.mean(ref_sr) - 1:+.0%} on average")
wins = sum(o > r for o, r in zip(ours, ref_sr))
print(f"proposed beats the reference rule with SR on {wins} of {len(cases)} channels")
assert np.mean(ours) > np.mean(ref_sr) > np.mean(ref)

section("3. Channel that changes during the transfer (clean, noisy, bursty, light)")
dyn = {k: v for k, v in schemes.items() if k in ("SR + CRC", "SR + Hamming (fixed)", "Reference rule, GBN",
                                                    "Reference rule, SR", "PROPOSED ADAPTIVE (SR)")}
res = {}
for name, make in dyn.items():
    vals = []
    for seed in SEEDS:
        r_obj = AdaptiveRTO(5082)
        proto, W, pol = make(r_obj, seed)
        r = simulate_arq(proto, W, 1024 * 900, four_phase_channel(seed * 10), policy=pol, rto=r_obj,
                         seed=seed, abort_factor=20)
        vals.append(r.throughput)
    res[name] = np.mean(vals)
    print(f"{name:<24} throughput {res[name]:.3f}")
assert res["PROPOSED ADAPTIVE (SR)"] == max(res.values())
print("\nAll checks passed.")
