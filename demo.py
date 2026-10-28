"""
demo.py : command line demo (the backup if the dashboard misbehaves).

Examples
  python demo.py --channel random --ber 1e-3
  python demo.py --channel bursty --ber 3e-3 --burst 50 --scheme "PROPOSED ADAPTIVE"
  python demo.py --channel changing
  python demo.py --channel random --ber 1e-2 --frames 400 --seeds 3
"""
import argparse
import pandas as pd

from runner import SCHEMES, run_many, summarize_choices

ap = argparse.ArgumentParser()
ap.add_argument("--channel", choices=["random", "bursty", "changing"], default="random")
ap.add_argument("--ber", type=float, default=1e-3, help="average bit error rate (ignored for 'changing')")
ap.add_argument("--burst", type=int, default=20, help="mean burst length in bits for bursty errors")
ap.add_argument("--delay", type=int, default=1500, help="one way propagation delay in bit times")
ap.add_argument("--frames", type=int, default=200, help="message size in frames of 1024 bits")
ap.add_argument("--seeds", type=int, default=1, help="number of random runs to average")
ap.add_argument("--scheme", default="all", help='one of: all, ' + ", ".join(f'"{s}"' for s in SCHEMES))
args = ap.parse_args()

schemes = SCHEMES if args.scheme == "all" else [args.scheme]
frames = max(args.frames, 900) if args.channel == "changing" else args.frames
print(f"channel={args.channel} ber={args.ber:g} burst={args.burst} delay={args.delay} frames={frames} seeds={args.seeds}\n")
df, choices = run_many(schemes, args.channel, args.ber, args.burst, args.delay, frames, args.seeds)
pd.set_option("display.width", 200)
print(df.round(3).to_string())
if choices:
    print("\nWhat the adaptive controller chose (share of new frames):")
    print((summarize_choices(choices).head(6) * 100).round(0).astype(int).astype(str).add("%").to_string())
