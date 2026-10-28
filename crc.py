"""
crc.py : Cyclic Redundancy Check (error DETECTION) for the HARQ project.

Two versions of the same maths are provided:
  crc_remainder_bitwise : textbook modulo 2 long division, easy to understand and explain
  CRC class             : fast table driven version used in the simulations
"""
import numpy as np


def crc_remainder_bitwise(data_bits, generator_bits):
    """
    Modulo 2 long division, exactly as taught in class.
    Example: data 100100, generator 1101 -> remainder 001
    """
    g = list(generator_bits)
    r = len(g) - 1
    work = list(data_bits) + [0] * r          # append r zeros
    for i in range(len(data_bits)):
        if work[i] == 1:                      # divide only when the leading bit is 1
            for j in range(len(g)):
                work[i + j] ^= g[j]           # modulo 2 subtraction is XOR
    return work[-r:]


def _reverse_bits(value, width):
    out = 0
    for _ in range(width):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


class CRC:
    """Generic CRC calculator working on bit arrays (length must be a multiple of 8)."""

    def __init__(self, width, poly, init, xorout, reflect, name=""):
        self.width, self.poly, self.init = width, poly, init
        self.xorout, self.reflect, self.name = xorout, reflect, name
        self.mask = (1 << width) - 1
        self.table = self._build_table()

    def _build_table(self):
        table = []
        for i in range(256):
            if self.reflect:
                rpoly = _reverse_bits(self.poly, self.width)
                crc = i
                for _ in range(8):
                    crc = (crc >> 1) ^ rpoly if crc & 1 else crc >> 1
            else:
                top = 1 << (self.width - 1)
                crc = i << (self.width - 8)
                for _ in range(8):
                    crc = ((crc << 1) ^ self.poly) if crc & top else crc << 1
                crc &= self.mask
            table.append(crc)
        return table

    def compute(self, data):
        """CRC value (an int) of a bytes object."""
        if self.reflect:
            crc = _reverse_bits(self.init, self.width)
            for b in data:
                crc = (crc >> 8) ^ self.table[(crc ^ b) & 0xFF]
        else:
            crc = self.init
            shift = self.width - 8
            for b in data:
                crc = ((crc << 8) & self.mask) ^ self.table[((crc >> shift) ^ b) & 0xFF]
        return crc ^ self.xorout

    # ---- bit array interface used by the rest of the project ----
    def encode(self, bits):
        """Return bits followed by their CRC bits."""
        if bits.size % 8:
            raise ValueError("payload length must be a multiple of 8 bits")
        value = self.compute(np.packbits(bits).tobytes())
        crc_bits = np.unpackbits(np.frombuffer(value.to_bytes(self.width // 8, "big"), dtype=np.uint8))
        return np.concatenate([bits, crc_bits])

    def check(self, bits_with_crc):
        """True if the frame passes the CRC check (no error detected)."""
        data, tail = bits_with_crc[:-self.width], bits_with_crc[-self.width:]
        received = int.from_bytes(np.packbits(tail).tobytes(), "big")
        return self.compute(np.packbits(data).tobytes()) == received


CRC16 = CRC(16, 0x1021, 0xFFFF, 0x0000, reflect=False, name="CRC-16/CCITT")
CRC32 = CRC(32, 0x04C11DB7, 0xFFFFFFFF, 0xFFFFFFFF, reflect=True, name="CRC-32/Ethernet")
