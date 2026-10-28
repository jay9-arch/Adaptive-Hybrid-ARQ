"""test_part5.py : adaptive timeout (RFC 6298) versus fixed timeouts. Run: python test_part5.py"""
import numpy as np
from channel import BSCChannel
from arq import simulate_arq, ACK_BITS
from rto import FixedRTO, AdaptiveRTO

M = 1024 * 400            # message size in bits
SEEDS = range(5)          # every number below is an average over 5 random runs


def section(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


def run(proto, W, make_rto, **link):
    rows = []
    for s in SEEDS:
        r = simulate_arq(proto, W, M, BSCChannel(1e-4, seed=s), rto=make_rto(), seed=s, **link)
        rows.append((r.throughput, r.timeouts, r.spurious))
    return np.mean(rows, axis=0)


section("1. The RFC 6298 estimator, checked by hand")
r = AdaptiveRTO(initial=1000, granularity=10)
r.on_sample(100)
print(f"sample 100 -> SRTT {r.srtt:.1f}, RTTVAR {r.rttvar:.1f}, RTO {r.value():.1f}   (hand: 100, 50, 300)")
assert (r.srtt, r.rttvar, r.value()) == (100, 50, 300)
r.on_sample(100)
print(f"sample 100 -> SRTT {r.srtt:.1f}, RTTVAR {r.rttvar:.1f}, RTO {r.value():.1f}   (hand: 100, 37.5, 250)")
assert (r.srtt, r.rttvar, r.value()) == (100, 37.5, 250)
r.on_sample(180)
print(f"sample 180 -> SRTT {r.srtt:.2f}, RTTVAR {r.rttvar:.2f}, RTO {r.value():.2f}  (hand: 110, 48.125, 302.5)")
assert abs(r.srtt - 110) < 1e-9 and abs(r.rttvar - 48.125) < 1e-9
r.on_timeout(); r.on_timeout()
print(f"after 2 timeouts RTO = {r.value():.1f}  (4 x 302.5 = 1210)")
assert abs(r.value() - 1210) < 1e-9
r.on_sample(110)
print(f"after a fresh sample the backoff is gone: RTO = {r.value():.1f}")
assert r.backoff == 0

section("2. Scenario A: delay varies a lot, but within known limits (RTT 3032 to 5032)")
print("Selective Repeat, window 16, BER 1e-4. 'spurious' = retransmissions of frames that were NOT lost.\n")
print(f"{'timeout':<26} | {'throughput':>10} | {'timeouts':>8} | {'spurious':>8}")
res_a = {}
for name, f in (("fixed 3300 (too tight)", lambda: FixedRTO(3300)),
                ("fixed 5082 (worst case)", lambda: FixedRTO(5082)),
                ("fixed 15000 (very safe)", lambda: FixedRTO(15000)),
                ("adaptive RFC 6298", lambda: AdaptiveRTO(5082))):
    x = run("sr", 16, f, prop=1500, jitter=1000)
    res_a[name] = x
    print(f"{name:<26} | {x[0]:>10.3f} | {x[1]:>8.0f} | {x[2]:>8.0f}")
assert res_a["adaptive RFC 6298"][0] > res_a["fixed 3300 (too tight)"][0]
assert res_a["adaptive RFC 6298"][0] > res_a["fixed 15000 (very safe)"][0]

section("3. Scenario B: heavy tailed delay (no true worst case exists)")
print(f"{'timeout':<26} | {'throughput':>10} | {'timeouts':>8} | {'spurious':>8}")
res_b = {}
for name, f in (("fixed 4500", lambda: FixedRTO(4500)),
                ("fixed 5500", lambda: FixedRTO(5500)),
                ("fixed 8000", lambda: FixedRTO(8000)),
                ("fixed 15000", lambda: FixedRTO(15000)),
                ("adaptive RFC 6298", lambda: AdaptiveRTO(5082))):
    x = run("sr", 16, f, prop=1500, jitter=1500, jitter_dist="exp")
    res_b[name] = x
    print(f"{name:<26} | {x[0]:>10.3f} | {x[1]:>8.0f} | {x[2]:>8.0f}")

section("4. Scenario C: the path gets slower in the middle (one way delay 1500 -> 3500)")
jump = lambda t: 1500 if t < 150000 else 3500
print(f"{'timeout':<34} | {'throughput':>10} | {'timeouts':>8} | {'spurious':>8}")
res_c = {}
for name, f in (("fixed 3082 (tuned for old delay)", lambda: FixedRTO(3082)),
                ("fixed 7082 (tuned for new delay)", lambda: FixedRTO(7082)),
                ("fixed 15000 (very safe)", lambda: FixedRTO(15000)),
                ("adaptive RFC 6298", lambda: AdaptiveRTO(3082))):
    x = run("sr", 32, f, prop=jump)
    res_c[name] = x
    print(f"{name:<34} | {x[0]:>10.3f} | {x[1]:>8.0f} | {x[2]:>8.0f}")
assert res_c["adaptive RFC 6298"][0] > 1.3 * res_c["fixed 3082 (tuned for old delay)"][0]

section("5. Go-Back-N is far more sensitive to timer mistakes (scenario A)")
print(f"{'timeout':<26} | {'throughput':>10}")
for name, f in (("fixed 3300 (too tight)", lambda: FixedRTO(3300)),
                ("fixed 5082 (worst case)", lambda: FixedRTO(5082)),
                ("adaptive RFC 6298", lambda: AdaptiveRTO(5082))):
    x = run("gbn", 16, f, prop=1500, jitter=1000)
    print(f"{name:<26} | {x[0]:>10.3f}")
print("\nAll checks passed.")
