"""test_part6.py : the adaptive controller. Run: python test_part6.py   (takes about a minute)"""
import collections
import numpy as np
from frame import FrameCodec, CRC_ONLY, HAMMING74, MODE_NAMES
from policy import StaticPolicy
from arq import simulate_arq
from rto import AdaptiveRTO
from adaptive import AdaptivePolicy
from scenarios import make_channel, four_phase_channel

codec = FrameCodec()
SEEDS = (1, 2, 3)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# ------------------------------------------------------------------------------------------
section("1. What the model recommends when it KNOWS the channel (no simulation)")
import math
rto = AdaptiveRTO(3082); rto.srtt, rto.base = 3032, 3042
pol = AdaptivePolicy(codec, rto=rto)
print(f"{'random BER':>10} | recommended configuration | predicted efficiency | window")
expected = {1e-5: (CRC_ONLY, 2048), 1e-3: (CRC_ONLY, 128), 3e-3: (HAMMING74, 1024), 1e-2: (HAMMING74, 256)}
for ber, want in expected.items():
    hc = -math.log(1 - ber)
    qh = pol._crc_to_ham(hc)
    mode, size = pol._best(pol.modes, hc, qh)
    pol.current = (mode, size)
    print(f"{ber:>10.0e} | {MODE_NAMES[mode]:<17} {size:>5} bits | {pol.predicted_efficiency(mode, size, hc, qh):>20.3f} | {pol.window_limit():>6}")
    assert (mode, size) == want
print("These match the best fixed settings found by exhaustive search (oracle_grid.py).")

# ------------------------------------------------------------------------------------------
section("2. Ablation: which part of the idea does the work? (throughput, 1.0 = perfect)")
variants = {
    "fixed CRC 1024":  lambda r, s: StaticPolicy(CRC_ONLY, 1024),
    "fixed Ham 1024":  lambda r, s: StaticPolicy(HAMMING74, 1024),
    "mode switch only": lambda r, s: AdaptivePolicy(codec, rto=r, seed=s, adapt_size=False, fixed_size=1024),
    "size only (CRC)": lambda r, s: AdaptivePolicy(codec, rto=r, seed=s, adapt_mode=False, fixed_mode=CRC_ONLY),
    "size only (Ham)": lambda r, s: AdaptivePolicy(codec, rto=r, seed=s, adapt_mode=False, fixed_mode=HAMMING74),
    "FULL ADAPTIVE":   lambda r, s: AdaptivePolicy(codec, rto=r, seed=s),
}
cases = [("rand", 1e-4), ("rand", 1e-3), ("rand", 3e-3), ("rand", 1e-2), ("burst", 1e-3), ("burst", 1e-2)]
reference = [0.888, 0.677, 0.531, 0.452, 0.889, 0.646]          # best fixed setting per channel (oracle_grid.py)
table, undetected = {}, {}
for name, make in variants.items():
    row, undetected[name] = [], 0
    for kind, ber in cases:
        vals = []
        for seed in SEEDS:
            r_obj = AdaptiveRTO(5082)
            res = simulate_arq("sr", 32 if name.startswith("fixed") else 128, 1024 * 300, make_channel(kind, ber, seed),
                               policy=make(r_obj, seed), rto=r_obj, seed=seed, abort_factor=60)
            undetected[name] += res.undetected_errors
            if name == "FULL ADAPTIVE":                          # settings that cannot adapt may give up, that is the point
                assert res.completed and res.undetected_errors == 0
            vals.append(res.throughput)
        row.append(np.mean(vals))
    table[name] = row
header = "".join(f"{('random' if k == 'rand' else 'bursty')[:4]} {b:.0e}".rjust(12) for k, b in cases)
print(f"{'':<17}|{header} | worst case*")
best_in_col = [max(table[n][i] for n in table) for i in range(len(cases))]
for name, row in table.items():
    worst = min(v / b for v, b in zip(row, best_in_col))
    print(f"{name:<17}|" + "".join(f"{v:>12.3f}" for v in row) + f" | {worst:>9.0%}")
print(f"{'best fixed (grid)':<17}|" + "".join(f"{v:>12.3f}" for v in reference) + " |  reference")
print("* worst case = lowest value of (variant / best variant on that channel) over all six channels")
bad = {n: u for n, u in undetected.items() if u}
print("frames wrongly accepted by the CRC (residual errors):", bad if bad else "none")
print("A fixed CRC with large frames on a very noisy channel resends thousands of garbage frames,")
print("and about 1 in 65,000 of them passes CRC-16 by chance. Adaptive schemes avoid that situation.")
full = table["FULL ADAPTIVE"]
assert all(f >= 0.85 * b for f, b in zip(full, best_in_col)), "adaptive should stay close to the best variant everywhere"
assert min(v / b for v, b in zip(full, best_in_col)) > max(min(v / b for v, b in zip(row, best_in_col))
                                                           for n, row in table.items() if n != "FULL ADAPTIVE")

# ------------------------------------------------------------------------------------------
section("3. A channel that changes during the transfer (1.0 = perfect)")
print("phase 1: random 1e-5 | phase 2: random 3e-3 | phase 3: BURSTY 1e-2 | phase 4: random 1e-4\n")


class Logged(AdaptivePolicy):
    def __init__(self, *a, chan=None, **k):
        super().__init__(*a, **k)
        self.chan, self.log = chan, []

    def tx_mode(self, idx, attempt):
        m = super().tx_mode(idx, attempt)
        if attempt == 0:
            self.log.append((self.chan.index, m, self._cur_bits))
        return m


results, kept = {}, None
for name in ("fixed CRC 1024", "fixed Ham 1024", "fixed CRC 256", "fixed Ham 512", "FULL ADAPTIVE"):
    vals = []
    for seed in SEEDS:
        ch, r_obj = four_phase_channel(seed * 10), AdaptiveRTO(5082)
        if name == "FULL ADAPTIVE":
            pol = Logged(codec, rto=r_obj, seed=seed, chan=ch)
        else:
            pol = StaticPolicy(CRC_ONLY if "CRC" in name else HAMMING74, int(name.split()[-1]))
        res = simulate_arq("sr", 128, 1024 * 900, ch, policy=pol, rto=r_obj, seed=seed, abort_factor=60)
        assert res.completed and res.undetected_errors == 0
        vals.append(res.throughput)
        if name == "FULL ADAPTIVE" and seed == 1:
            kept = pol
    results[name] = np.mean(vals)
    print(f"{name:<16} throughput {results[name]:.3f}")
best_fixed = max(v for n, v in results.items() if n != "FULL ADAPTIVE")
fixed_vals = [v for n, v in results.items() if n != "FULL ADAPTIVE"]
print(f"\nadaptive vs best fixed setting    : {results['FULL ADAPTIVE'] / best_fixed - 1:+.1%}")
print(f"adaptive vs average fixed setting : {results['FULL ADAPTIVE'] / np.mean(fixed_vals) - 1:+.1%}")
print("The best fixed setting here (CRC 256) happens to suit three of the four phases. Someone who does not")
print("know the channel in advance cannot pick it, the adaptive scheme matches it without being told.")
assert results["FULL ADAPTIVE"] >= 0.97 * best_fixed
print("\nWhat the controller did in each phase (share of new frames):")
for ph, label in enumerate(("1 clean", "2 noisy random", "3 bursty", "4 light noise")):
    c = collections.Counter((MODE_NAMES[m].split("(")[0].strip(), b) for p, m, b in kept.log if p == ph)
    tot = sum(c.values())
    print(f"  phase {label:<15}", ", ".join(f"{k[0]} {k[1]} bits {v / tot:.0%}" for k, v in c.most_common(3)))
print("\nAll checks passed.")
