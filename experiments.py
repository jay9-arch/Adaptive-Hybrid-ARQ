"""
experiments.py : the final experiments for the report. Produces graphs and tables in ./results

Run:   python experiments.py              full run (5 random seeds per point, a few minutes)
       python experiments.py --quick      2 seeds, fewer points (about a minute)
       python experiments.py --workers 4  use 4 processor cores

HOLD OUT SETTINGS: the controller was tuned on BER 1e-5..1e-2, bursts of 20 bits and a 1500 bit delay.
These experiments use DIFFERENT error rates, bursts of 50 bits and a 2500 bit one way delay, so the
results are not simply the numbers the controller was tuned on.
"""
import argparse
import collections
import csv
import os
import time
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel import BSCChannel, GilbertElliottChannel
from frame import FrameCodec, CRC_ONLY, HAMMING74, MODE_NAMES
from policy import StaticPolicy
from arq import simulate_arq, ACK_BITS
from rto import AdaptiveRTO
from adaptive import AdaptivePolicy
from goga import GogaPolicy
from scenarios import PhasedChannel

PROP, BURST, MESSAGE = 2500, 50, 1024 * 200
RTO0 = 2 * PROP + ACK_BITS + 3000                  # generous first guess, the adaptive timer corrects it
SCHEMES = ["GBN + CRC", "SR + CRC", "SR + Hamming (fixed)", "Reference rule, GBN", "Reference rule, SR", "PROPOSED ADAPTIVE"]
STYLE = {"GBN + CRC": ("tab:gray", "x", ":"), "SR + CRC": ("tab:blue", "s", "--"),
         "SR + Hamming (fixed)": ("tab:green", "^", "--"), "Reference rule, GBN": ("tab:orange", "v", "-."),
         "Reference rule, SR": ("tab:red", "D", "-."), "PROPOSED ADAPTIVE": ("black", "o", "-")}


def channel(kind, ber, seed):
    if kind == "random":
        return BSCChannel(ber, seed=seed)
    return GilbertElliottChannel.from_average_ber(ber, mean_burst=BURST, seed=seed)


def build(scheme, seed, rto):
    codec = FrameCodec()
    return {
        "GBN + CRC": lambda: ("gbn", 16, StaticPolicy(CRC_ONLY, 1024)),
        "SR + CRC": lambda: ("sr", 32, StaticPolicy(CRC_ONLY, 1024)),
        "SR + Hamming (fixed)": lambda: ("sr", 32, StaticPolicy(HAMMING74, 1024)),
        "Reference rule, GBN": lambda: ("gbn", 16, GogaPolicy()),
        "Reference rule, SR": lambda: ("sr", 32, GogaPolicy()),
        "PROPOSED ADAPTIVE": lambda: ("sr", 128, AdaptivePolicy(codec, rto=rto, rtt_hint=2 * PROP + ACK_BITS, seed=seed)),
    }[scheme]()


def run_one(job):
    scheme, kind, ber, seed = job
    rto = AdaptiveRTO(RTO0)
    proto, window, policy = build(scheme, seed, rto)
    r = simulate_arq(proto, window, MESSAGE, channel(kind, ber, seed), policy=policy, rto=rto,
                     prop=PROP, seed=seed, abort_factor=20)
    return dict(scheme=scheme, kind=kind, ber=ber, seed=seed, completed=int(r.completed),
                throughput=r.throughput, bit_efficiency=r.bit_efficiency,
                tx_per_frame=r.transmissions / max(r.frames, 1),
                delay=r.avg_delay if r.completed else float("nan"), undetected=r.undetected_errors)


class Logged(AdaptivePolicy):
    """Adaptive policy that records what it chose, for the timeline figure."""
    def __init__(self, *a, chan=None, **k):
        super().__init__(*a, **k)
        self.chan, self.log = chan, []

    def new_frame_payload(self):
        p = super().new_frame_payload()
        self.log.append((self.chan.index, self._pending, p))
        return p


def phased(seed, n=250_000):
    return PhasedChannel([(n, channel("random", 1e-5, seed)), (n, channel("random", 4e-3, seed + 1)),
                          (n, channel("burst", 1.5e-2, seed + 2)), (n, channel("random", 2e-4, seed + 3))])


def run_dynamic(job):
    scheme, seed = job
    rto = AdaptiveRTO(RTO0)
    ch = phased(seed * 10)
    if scheme == "PROPOSED ADAPTIVE":
        pol = Logged(FrameCodec(), rto=rto, rtt_hint=2 * PROP + ACK_BITS, seed=seed, chan=ch)
        proto, W = "sr", 128
    else:
        proto, W, pol = build(scheme, seed, rto)
    r = simulate_arq(proto, W, 1024 * 900, ch, policy=pol, rto=rto, prop=PROP, seed=seed, abort_factor=20)
    return scheme, seed, r.throughput, (pol.log if scheme == "PROPOSED ADAPTIVE" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    seeds = range(1, 3) if args.quick else range(1, 6)
    bers = [2e-4, 2e-3, 8e-3] if args.quick else [2e-5, 2e-4, 6e-4, 2e-3, 5e-3, 8e-3]
    os.makedirs("results", exist_ok=True)
    t0 = time.time()

    jobs = [(s, k, b, sd) for s in SCHEMES for k in ("random", "bursty") for b in bers for sd in seeds]
    print(f"{len(jobs)} simulations ...")
    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = pool.map(run_one, jobs)
    else:
        rows = []
        for i, j in enumerate(jobs):
            rows.append(run_one(j))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    with open("results/all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- aggregate: mean and standard deviation over seeds
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["scheme"], r["kind"], r["ber"])].append(r)
    def stat(key, field):
        v = np.array([r[field] for r in agg[key]], dtype=float)
        return np.nanmean(v) if not np.all(np.isnan(v)) else float("nan"), np.nanstd(v) if not np.all(np.isnan(v)) else float("nan")
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "channel", "ber", "throughput_mean", "throughput_std", "tx_per_frame_mean", "delay_mean",
                    "bit_efficiency_mean", "undetected_errors_total", "runs_completed"])
        for (s, k, b), rs in sorted(agg.items()):
            w.writerow([s, k, b, *(f"{x:.4f}" for x in stat((s, k, b), "throughput")), f"{stat((s, k, b), 'tx_per_frame')[0]:.3f}",
                        f"{stat((s, k, b), 'delay')[0]:.0f}", f"{stat((s, k, b), 'bit_efficiency')[0]:.4f}",
                        sum(r["undetected"] for r in rs), sum(r["completed"] for r in rs)])

    # ---- figures 1 to 4: metric versus BER
    for fname, field, ylabel, log in (("fig1_throughput", "throughput", "Throughput (fraction of link rate)", False),
                                      ("fig2_retransmissions", "tx_per_frame", "Transmissions per frame (1 = no retransmission)", True),
                                      ("fig3_delay", "delay", "Average frame delay (bit times)", True),
                                      ("fig4_bit_efficiency", "bit_efficiency", "Useful bits / bits sent", False)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, kind in zip(axes, ("random", "bursty")):
            for s in SCHEMES:
                c, m, ls = STYLE[s]
                mu = np.array([stat((s, kind, b), field)[0] for b in bers])
                sd = np.array([stat((s, kind, b), field)[1] for b in bers])
                ax.errorbar(bers, mu, yerr=sd, color=c, marker=m, linestyle=ls, capsize=2, lw=2 if s.startswith("PROP") else 1.2, label=s)
            ax.set_xscale("log"); ax.set_xlabel("Average bit error rate"); ax.set_title(f"{kind} errors"); ax.grid(alpha=.3)
            if log:
                ax.set_yscale("log")
        axes[0].set_ylabel(ylabel); axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"results/{fname}.png", dpi=150); plt.close(fig)

    # ---- changing channel experiment
    dyn_jobs = [(s, sd) for s in SCHEMES if s != "GBN + CRC" for sd in seeds]
    dyn = [run_dynamic(j) for j in dyn_jobs]
    dyn_tp = {s: np.mean([t for sc, sd, t, _ in dyn if sc == s]) for s in SCHEMES if s != "GBN + CRC"}
    log = next(l for sc, sd, t, l in dyn if sc == "PROPOSED ADAPTIVE")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(dyn_tp)
    a1.bar(range(len(names)), [dyn_tp[n] for n in names], color=[STYLE[n][0] for n in names])
    a1.set_xticks(range(len(names))); a1.set_xticklabels([n.replace(", ", ",\n").replace(" (", "\n(") for n in names], fontsize=7)
    a1.set_ylabel("Throughput"); a1.set_title("Channel changing during the transfer")
    xs = np.arange(len(log)); pay = [p for _, _, p in log]; mode = [m for _, m, _ in log]
    a2.scatter(xs, pay, c=["tab:red" if m == HAMMING74 else "tab:blue" for m in mode], s=4)
    a2.set_yscale("log", base=2); a2.set_xlabel("Frame number"); a2.set_ylabel("Payload chosen (bits)")
    a2.set_title("What the controller chose (blue = CRC only, red = Hamming)")
    last = None
    for i, (ph, _, _) in enumerate(log):
        if ph != last and last is not None:
            a2.axvline(i, color="gray", ls=":")
        last = ph
    fig.tight_layout(); fig.savefig("results/fig5_changing_channel.png", dpi=150); plt.close(fig)

    # ---- console summary
    print("\nTHROUGHPUT (mean over seeds)   hold out settings: bursts 50 bits, one way delay 2500")
    for kind in ("random", "bursty"):
        print(f"\n{kind} errors\n{'scheme':<24}" + "".join(f"{b:>10.0e}" for b in bers) + "     mean")
        for s in SCHEMES:
            v = [stat((s, kind, b), "throughput")[0] for b in bers]
            print(f"{s:<24}" + "".join(f"{x:>10.3f}" for x in v) + f"   {np.mean(v):.3f}")
    ours = np.mean([stat(("PROPOSED ADAPTIVE", k, b), "throughput")[0] for k in ("random", "bursty") for b in bers])
    for s in ("Reference rule, GBN", "Reference rule, SR", "SR + Hamming (fixed)", "SR + CRC"):
        other = np.mean([stat((s, k, b), "throughput")[0] for k in ("random", "bursty") for b in bers])
        print(f"proposed vs {s:<22}: {ours / other - 1:+.0%}")
    print("\nchanging channel:", {k: round(v, 3) for k, v in dyn_tp.items()})
    und = {s: sum(r['undetected'] for r in rows if r['scheme'] == s) for s in SCHEMES}
    print("frames wrongly accepted by the CRC:", und)
    print(f"\nSaved to ./results/ : all_runs.csv, summary.csv, fig1..fig5 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
