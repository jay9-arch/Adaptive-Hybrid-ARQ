"""
runner.py : one place that runs a scheme on a channel and returns the numbers.
Used by the dashboard (app.py), the command line demo (demo.py) and test_part9.py.
"""
import numpy as np
import pandas as pd

from channel import BSCChannel, GilbertElliottChannel
from frame import FrameCodec, CRC_ONLY, HAMMING74, MODE_NAMES
from policy import StaticPolicy
from arq import simulate_arq, ACK_BITS
from rto import AdaptiveRTO
from adaptive import AdaptivePolicy
from goga import GogaPolicy
from scenarios import PhasedChannel

SCHEMES = ["GBN + CRC", "SR + CRC", "SR + Hamming (fixed)", "Reference rule, GBN", "Reference rule, SR",
           "PROPOSED ADAPTIVE"]
SCENARIOS = ["random", "bursty", "changing"]


def make_channel(kind, ber, burst, seed):
    if kind == "random":
        return BSCChannel(ber, seed=seed)
    return GilbertElliottChannel.from_average_ber(ber, mean_burst=burst, seed=seed)


def changing_channel(burst, seed, phase_bits=250_000):
    """clean -> noisy random -> bursty -> light noise"""
    return PhasedChannel([(phase_bits, make_channel("random", 1e-5, burst, seed)),
                          (phase_bits, make_channel("random", 4e-3, burst, seed + 1)),
                          (phase_bits, make_channel("bursty", 1.5e-2, burst, seed + 2)),
                          (phase_bits, make_channel("random", 2e-4, burst, seed + 3))])


class _LoggedAdaptive(AdaptivePolicy):
    """Adaptive policy that remembers what it chose for every new frame."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.choices = []

    def new_frame_payload(self):
        p = super().new_frame_payload()
        self.choices.append((self._pending, p))
        return p


def run_scheme(scheme, scenario, ber, burst, prop, frames, seed):
    """Returns (metrics dict, list of (mode, payload) chosen by the adaptive policy or None)."""
    rto0 = 2 * prop + ACK_BITS + 3000
    rto = AdaptiveRTO(rto0)
    codec = FrameCodec()
    if scheme == "GBN + CRC":
        proto, window, pol = "gbn", 16, StaticPolicy(CRC_ONLY, 1024)
    elif scheme == "SR + CRC":
        proto, window, pol = "sr", 32, StaticPolicy(CRC_ONLY, 1024)
    elif scheme == "SR + Hamming (fixed)":
        proto, window, pol = "sr", 32, StaticPolicy(HAMMING74, 1024)
    elif scheme == "Reference rule, GBN":
        proto, window, pol = "gbn", 16, GogaPolicy()
    elif scheme == "Reference rule, SR":
        proto, window, pol = "sr", 32, GogaPolicy()
    elif scheme == "PROPOSED ADAPTIVE":
        proto, window = "sr", 128
        pol = _LoggedAdaptive(codec, rto=rto, rtt_hint=2 * prop + ACK_BITS, seed=seed)
    else:
        raise ValueError(scheme)
    channel = changing_channel(burst, seed * 10) if scenario == "changing" else make_channel(scenario, ber, burst, seed)
    r = simulate_arq(proto, window, frames * 1024, channel, policy=pol, rto=rto, prop=prop, seed=seed, abort_factor=20)
    metrics = {"Throughput": r.throughput, "Useful bits / bits sent": r.bit_efficiency,
               "Transmissions per frame": r.transmissions / max(r.frames, 1),
               "Avg delay (bit times)": r.avg_delay if r.completed else float("nan"),
               "Retransmissions": r.retransmissions, "Wrong frames accepted": r.undetected_errors,
               "Finished": float(r.completed)}
    return metrics, (pol.choices if scheme == "PROPOSED ADAPTIVE" else None)


def run_many(schemes, scenario, ber, burst, prop, frames, seeds):
    """Mean of every metric over `seeds` runs. Returns (DataFrame, adaptive choices of the first seed)."""
    rows, choices = {}, None
    for s in schemes:
        per_seed = []
        for seed in range(1, seeds + 1):
            m, ch = run_scheme(s, scenario, ber, burst, prop, frames, seed)
            per_seed.append(m)
            if ch is not None and seed == 1:
                choices = ch
        rows[s] = pd.DataFrame(per_seed).mean(numeric_only=True)
    return pd.DataFrame(rows).T, choices


def summarize_choices(choices):
    """Share of new frames per (mode, payload size)."""
    counts = pd.Series([f"{MODE_NAMES[m].split('(')[0].strip()} {p} bits" for m, p in choices]).value_counts(normalize=True)
    return counts
