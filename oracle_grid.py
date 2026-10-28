"""
oracle_grid.py : tries EVERY fixed (mode, payload size) pair on each channel and reports the best.
This is the ceiling a controller could reach if it knew the channel in advance.
Run:  python oracle_grid.py        (takes about a minute)
"""
import numpy as np
from frame import CRC_ONLY, HAMMING74
from policy import StaticPolicy
from arq import simulate_arq
from rto import AdaptiveRTO
from adaptive import SIZES
from scenarios import make_channel

M = 1024 * 300
for kind in ("rand", "burst"):
    print(kind)
    print(f"{'BER':>7} | " + " ".join(f"C{p:>5}" for p in SIZES) + " | " + " ".join(f"H{p:>5}" for p in SIZES) + " | best")
    for ber in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        row = []
        for mode in (CRC_ONLY, HAMMING74):
            for p in SIZES:
                vals = [simulate_arq("sr", 128, M, make_channel(kind, ber, s), policy=StaticPolicy(mode, p),
                                     rto=AdaptiveRTO(5082), seed=s, abort_factor=80).throughput for s in (1, 2)]
                row.append(np.mean(vals))
        names = [f"C{p}" for p in SIZES] + [f"H{p}" for p in SIZES]
        i = int(np.argmax(row))
        print(f"{ber:>7.0e} | " + " ".join(f"{v:>6.3f}" for v in row[:6]) + " | " + " ".join(f"{v:>6.3f}" for v in row[6:])
              + f" | {names[i]} {row[i]:.3f}")
