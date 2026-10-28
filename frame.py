"""
frame.py : frame format and the three transmission modes of the Adaptive Hybrid ARQ project.

On the wire a frame looks like this:

  +----------------+--------------------------------------------------------------+
  |  MODE FIELD    |  BODY (Hamming coded + interleaved, or plain, depending on mode)|
  |  6 bits        |  [ SEQ 8 bits | PAYLOAD n bits | CRC 16 bits ]                |
  +----------------+--------------------------------------------------------------+

MODE FIELD : 2 bit mode number sent three times (repetition code, majority vote), so the
             receiver can learn the mode BEFORE it knows how to decode the body.
SEQ        : sequence number (needed by Selective Repeat and Go-Back-N), wraps around.
CRC        : computed over SEQ + PAYLOAD. It is the final judge: a frame is accepted only
             if the CRC passes, whatever Hamming did.

Modes
  CRC_ONLY  : body sent as is. Lowest overhead. Any error means retransmission.
  HAMMING74 : body Hamming(7,4) coded and interleaved. Corrects 1 error per 7 bit block.
  HAMMING84 : body Hamming(8,4) coded and interleaved. Also reports blocks with 2 errors.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from crc import CRC16
from hamming import (hamming74_encode, hamming74_decode, hamming84_encode, hamming84_decode,
                     interleave, deinterleave)

CRC_ONLY, HAMMING74, HAMMING84 = 0, 1, 2
MODE_NAMES = {CRC_ONLY: "CRC only", HAMMING74: "Hamming(7,4)+CRC", HAMMING84: "Hamming(8,4)+CRC"}
MODE_REPEAT = 3                      # repetition factor of the mode field
MODE_FIELD_BITS = 2 * MODE_REPEAT


@dataclass
class RxResult:
    ok: bool                         # True if the frame was accepted (CRC passed)
    seq: Optional[int]               # sequence number if accepted
    payload: Optional[np.ndarray]    # payload bits if accepted
    mode: Optional[int]              # mode as read by the receiver
    corrected: int = 0               # Hamming blocks in which one error was fixed
    double_detected: int = 0         # Hamming(8,4) blocks in which two errors were detected
    reason: str = ""                 # why the frame was rejected: "crc", "mode" or "length"


class FrameCodec:
    def __init__(self, crc=CRC16, seq_bits=8):
        if seq_bits % 8:
            raise ValueError("seq_bits must be a multiple of 8")
        self.crc, self.seq_bits = crc, seq_bits

    # --------------------------------------------------------------- sender side
    def encode(self, seq, payload, mode):
        if payload.size % 8:
            raise ValueError("payload length must be a multiple of 8 bits")
        seq_arr = np.unpackbits(np.frombuffer(
            (seq % (1 << self.seq_bits)).to_bytes(self.seq_bits // 8, "big"), dtype=np.uint8))
        body = self.crc.encode(np.concatenate([seq_arr, payload]))

        if mode == CRC_ONLY:
            coded = body
        elif mode == HAMMING74:
            coded = interleave(hamming74_encode(body), 7)
        elif mode == HAMMING84:
            coded = interleave(hamming84_encode(body), 8)
        else:
            raise ValueError("unknown mode")

        mode_bits = np.array([(mode >> 1) & 1, mode & 1], dtype=np.uint8)
        return np.concatenate([np.repeat(mode_bits, MODE_REPEAT), coded])

    # ------------------------------------------------------------- receiver side
    def decode(self, rx):
        if rx.size <= MODE_FIELD_BITS:
            return RxResult(False, None, None, None, reason="length")
        votes = rx[:MODE_FIELD_BITS].reshape(2, MODE_REPEAT).sum(axis=1)
        b = (votes * 2 > MODE_REPEAT).astype(int)           # majority vote
        mode = int(b[0]) * 2 + int(b[1])
        body_rx = rx[MODE_FIELD_BITS:]
        corrected = double = 0

        if mode == CRC_ONLY:
            body = body_rx
        elif mode == HAMMING74:
            if body_rx.size % 7:
                return RxResult(False, None, None, mode, reason="length")
            body, corrected = hamming74_decode(deinterleave(body_rx, 7))
        elif mode == HAMMING84:
            if body_rx.size % 8:
                return RxResult(False, None, None, mode, reason="length")
            body, corrected, double = hamming84_decode(deinterleave(body_rx, 8))
        else:                                                # mode field decoded to the unused value 3
            return RxResult(False, None, None, mode, reason="mode")

        if body.size % 8 or body.size <= self.seq_bits + self.crc.width:
            return RxResult(False, None, None, mode, corrected, double, "length")
        if not self.crc.check(body):
            return RxResult(False, None, None, mode, corrected, double, "crc")
        seq = int.from_bytes(np.packbits(body[:self.seq_bits]).tobytes(), "big")
        payload = body[self.seq_bits:-self.crc.width]
        return RxResult(True, seq, payload, mode, corrected, double)

    # ---------------------------------------------------------------- bookkeeping
    def frame_bits(self, payload_len, mode):
        """Number of bits actually sent over the channel for one frame."""
        body = self.seq_bits + payload_len + self.crc.width
        coded = {CRC_ONLY: body, HAMMING74: body * 7 // 4, HAMMING84: body * 2}[mode]
        return MODE_FIELD_BITS + coded

    def overhead_efficiency(self, payload_len, mode):
        """Useful bits divided by bits sent, for ONE error free transmission."""
        return payload_len / self.frame_bits(payload_len, mode)
