"""
hamming.py : Hamming error CORRECTION (FEC) for the Adaptive Hybrid ARQ project.

Contents
  hamming74_encode / hamming74_decode : Hamming(7,4), corrects 1 error per 7 bit block
  hamming84_encode / hamming84_decode : Extended Hamming(8,4) "SECDED",
                                        corrects 1 error AND detects 2 errors per 8 bit block
  interleave / deinterleave           : spreads a burst of errors over many blocks

Bits are numpy uint8 arrays holding 0 or 1.

Hamming(7,4) layout (positions are numbered 1 to 7, position = binary number):
    position :  1   2   3   4   5   6   7
    role     :  p1  p2  d1  p4  d2  d3  d4
  p1 checks positions 1,3,5,7   (positions whose binary form has the 1s bit set)
  p2 checks positions 2,3,6,7   (2s bit set)
  p4 checks positions 4,5,6,7   (4s bit set)
At the receiver, re-computing the three checks gives a 3 bit SYNDROME.
Read as a binary number, the syndrome is the POSITION of the flipped bit (0 = no error).
"""
import numpy as np

DATA_POS = [2, 4, 5, 6]            # 0 based indices of d1..d4 inside a 7 bit codeword

# Parity check matrix: column j is the binary form of position j+1 (row 0 = 1s bit)
H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)


def _syndrome_position(cw7):
    """Syndrome of each 7 bit row, as a position number 0..7."""
    s = (cw7 @ H.T) % 2
    return s[:, 0] + 2 * s[:, 1] + 4 * s[:, 2]


# --------------------------------------------------------------------------- (7,4)
def hamming74_encode(bits):
    """4 data bits -> 7 coded bits. Input length must be a multiple of 4."""
    if bits.size % 4:
        raise ValueError("input length must be a multiple of 4")
    d = bits.reshape(-1, 4)
    cw = np.zeros((d.shape[0], 7), dtype=np.uint8)
    cw[:, DATA_POS] = d
    cw[:, 0] = cw[:, 2] ^ cw[:, 4] ^ cw[:, 6]      # p1 = d1 ^ d2 ^ d4
    cw[:, 1] = cw[:, 2] ^ cw[:, 5] ^ cw[:, 6]      # p2 = d1 ^ d3 ^ d4
    cw[:, 3] = cw[:, 4] ^ cw[:, 5] ^ cw[:, 6]      # p4 = d2 ^ d3 ^ d4
    return cw.reshape(-1)


def hamming74_decode(bits):
    """
    7 coded bits -> 4 data bits.
    Returns (data_bits, number_of_blocks_where_a_correction_was_made).
    """
    if bits.size % 7:
        raise ValueError("input length must be a multiple of 7")
    cw = bits.reshape(-1, 7).copy()
    pos = _syndrome_position(cw)
    rows = np.nonzero(pos)[0]
    cw[rows, pos[rows] - 1] ^= 1                    # flip the bit the syndrome points to
    return cw[:, DATA_POS].reshape(-1), int(rows.size)


# --------------------------------------------------------------------------- (8,4)
def hamming84_encode(bits):
    """Hamming(7,4) plus one overall parity bit -> 8 coded bits per 4 data bits."""
    cw7 = hamming74_encode(bits).reshape(-1, 7)
    overall = cw7.sum(axis=1, dtype=np.uint8) % 2
    return np.column_stack([cw7, overall]).reshape(-1).astype(np.uint8)


def hamming84_decode(bits):
    """
    Returns (data_bits, blocks_corrected, blocks_with_double_error_detected).

      syndrome = 0, overall parity OK  -> no error
      overall parity wrong             -> exactly one error: correct it
      syndrome != 0, parity OK         -> two errors: DETECTED, cannot correct
    """
    if bits.size % 8:
        raise ValueError("input length must be a multiple of 8")
    cw8 = bits.reshape(-1, 8).copy()
    cw7 = cw8[:, :7]
    pos = _syndrome_position(cw7)
    parity_bad = (cw8.sum(axis=1, dtype=np.uint8) % 2) == 1

    single = np.nonzero(parity_bad & (pos > 0))[0]
    cw7[single, pos[single] - 1] ^= 1               # fix single errors inside the 7 bits
    corrected = int(parity_bad.sum())               # includes an error in the overall parity bit
    double = int(np.count_nonzero(~parity_bad & (pos > 0)))
    return cw7[:, DATA_POS].reshape(-1), corrected, double


# ----------------------------------------------------------------------- interleaver
def interleave(bits, block_len, depth=None):
    """
    Write `depth` codewords as rows, send the bits COLUMN by column.
    A burst of up to `depth` bits then touches each codeword at most once.
    depth=None uses every codeword in the frame (no extra overhead, only needs the whole frame).
    """
    blocks = bits.size // block_len
    depth = blocks if depth is None else depth
    if blocks % depth:
        raise ValueError("number of blocks must be a multiple of depth")
    x = bits.reshape(blocks // depth, depth, block_len)
    return x.transpose(0, 2, 1).reshape(-1)


def deinterleave(bits, block_len, depth=None):
    blocks = bits.size // block_len
    depth = blocks if depth is None else depth
    if blocks % depth:
        raise ValueError("number of blocks must be a multiple of depth")
    x = bits.reshape(blocks // depth, block_len, depth)
    return x.transpose(0, 2, 1).reshape(-1)
