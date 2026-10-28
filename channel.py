"""
channel.py : noisy channel models for the Adaptive Hybrid ARQ project.

Bits are stored as numpy arrays of dtype uint8 containing only 0 or 1.
A channel takes a bit array and returns a copy in which some bits are flipped.

Two models are provided:
  1. BSCChannel           : every bit is flipped independently (random errors)
  2. GilbertElliottChannel: errors come in bursts (like fading on a wireless link)
"""
import numpy as np


class BSCChannel:
    """Binary Symmetric Channel: each bit flips independently with probability `ber`."""

    def __init__(self, ber, seed=None):
        self.rng = np.random.default_rng(seed)
        self.set_ber(ber)

    def set_ber(self, ber):
        if not 0.0 <= ber <= 0.5:
            raise ValueError("ber must be between 0 and 0.5")
        self.ber = ber

    @property
    def average_ber(self):
        return self.ber

    def transmit(self, bits):
        flips = (self.rng.random(bits.size) < self.ber).astype(np.uint8)
        return bits ^ flips


class GilbertElliottChannel:
    """
    Two state burst error model.

        GOOD state : bit error probability e_good (usually 0 or very small)
        BAD  state : bit error probability e_bad  (for example 0.5)

    p_gb = probability of moving GOOD -> BAD after each bit
    p_bg = probability of moving BAD  -> GOOD after each bit
    Mean burst length (bits spent in BAD) = 1 / p_bg
    """

    def __init__(self, p_gb, p_bg, e_good=0.0, e_bad=0.5, seed=None):
        self.rng = np.random.default_rng(seed)
        self.p_gb, self.p_bg = p_gb, p_bg
        self.e_good, self.e_bad = e_good, e_bad
        self.in_bad = False

    @classmethod
    def from_average_ber(cls, avg_ber, mean_burst=20, e_good=0.0, e_bad=0.5, seed=None):
        """Build a channel whose long run average BER equals `avg_ber`."""
        if not e_good <= avg_ber < e_bad:
            raise ValueError("avg_ber must lie between e_good and e_bad")
        frac_bad = (avg_ber - e_good) / (e_bad - e_good)       # time spent in BAD state
        p_bg = 1.0 / mean_burst
        p_gb = p_bg * frac_bad / (1.0 - frac_bad)
        return cls(p_gb, p_bg, e_good, e_bad, seed)

    @property
    def average_ber(self):
        frac_bad = self.p_gb / (self.p_gb + self.p_bg)
        return frac_bad * self.e_bad + (1 - frac_bad) * self.e_good

    def _state_sequence(self, n):
        """Return a boolean array (True = BAD) of length n, continuing from the current state."""
        states = np.empty(n, dtype=bool)
        pos, state = 0, self.in_bad
        while pos < n:
            p = self.p_bg if state else self.p_gb
            if p <= 0:                                  # state never changes
                states[pos:] = state
                break
            run = int(self.rng.geometric(p))            # how long we stay in this state
            end = min(pos + run, n)
            states[pos:end] = state
            if pos + run <= n:                          # run finished inside this block
                state = not state
            pos = end
        self.in_bad = state
        return states

    def transmit(self, bits):
        bad = self._state_sequence(bits.size)
        p_err = np.where(bad, self.e_bad, self.e_good)
        flips = (self.rng.random(bits.size) < p_err).astype(np.uint8)
        return bits ^ flips


def count_bit_errors(sent, received):
    """Number of positions where the two bit arrays differ."""
    return int(np.count_nonzero(sent != received))
