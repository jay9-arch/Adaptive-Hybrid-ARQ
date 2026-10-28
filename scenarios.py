"""
scenarios.py : channel setups shared by the tests and experiments.

  make_channel(kind, ber, seed)  : "rand" = independent errors, "burst" = Gilbert-Elliott with mean burst 20 bits
  PhasedChannel                  : a channel that changes character while a transfer is running
  four_phase_channel(seed)       : clean -> noisy random -> bursty -> lightly noisy
"""
from channel import BSCChannel, GilbertElliottChannel


def make_channel(kind, ber, seed):
    if kind == "rand":
        return BSCChannel(ber, seed=seed)
    return GilbertElliottChannel.from_average_ber(ber, mean_burst=20, seed=seed)


class PhasedChannel:
    """Uses channel number i while the number of bits sent so far lies in phase i."""

    def __init__(self, phases):                  # phases: list of (length in bits, channel)
        self.phases = phases
        self.sent = 0

    @property
    def index(self):
        total = 0
        for i, (length, _) in enumerate(self.phases):
            total += length
            if self.sent < total:
                return i
        return len(self.phases) - 1

    def transmit(self, bits):
        channel = self.phases[self.index][1]
        self.sent += bits.size
        return channel.transmit(bits)


def four_phase_channel(seed, phase_bits=250_000):
    return PhasedChannel([
        (phase_bits, make_channel("rand", 1e-5, seed)),
        (phase_bits, make_channel("rand", 3e-3, seed + 1)),
        (phase_bits, make_channel("burst", 1e-2, seed + 2)),
        (phase_bits, make_channel("rand", 1e-4, seed + 3)),
    ])
