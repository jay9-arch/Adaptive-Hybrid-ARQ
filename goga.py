"""
goga.py : the baseline from the reference paper, re-implemented inside our simulator.

Goga, Polec and Kotuliakova (Radioengineering, 2003) adapt between hybrid ARQ and pure ARQ by
COUNTING ACKs and NAKs. This is our version of that rule. IMPORTANT differences from the paper:
  * we use Hamming(7,4)+CRC as the hybrid mode (the paper uses a convolutional code),
  * the thresholds below are our own choice (the paper's exact settings were not used),
  * the frame size stays fixed, as in the paper.
So treat it as "the counting rule applied to our setup", not as a reproduction of their numbers.

RULE
  pure ARQ (CRC only)  -> hybrid : when `fail_threshold` of the last `window` transmissions failed
  hybrid -> pure ARQ             : after `success_threshold` ACKs in a row
It only sees ACK or no ACK. It does not use the Hamming correction counts, the frame size or any model.
"""
from collections import deque

from frame import CRC_ONLY, HAMMING74


class GogaPolicy:
    name = "ACK/NAK counting rule"

    def __init__(self, payload_len=1024, fail_threshold=2, window=8, success_threshold=16):
        self.payload_len = payload_len
        self.fail_threshold, self.window, self.success_threshold = fail_threshold, window, success_threshold
        self.mode = CRC_ONLY
        self.recent = deque(maxlen=window)
        self.run = 0
        self.switches = 0

    def new_frame_payload(self):
        return self.payload_len

    def tx_mode(self, idx, attempt):
        return self.mode

    def _outcome(self, ok):
        if self.mode == CRC_ONLY:
            self.recent.append(ok)
            if self.recent.count(False) >= self.fail_threshold:
                self.mode, self.run, self.switches = HAMMING74, 0, self.switches + 1
                self.recent.clear()
        else:
            self.run = self.run + 1 if ok else 0
            if self.run >= self.success_threshold:
                self.mode, self.switches = CRC_ONLY, self.switches + 1
                self.recent.clear()

    def on_ack(self, mode, corrected, double, bits=0, attempt=0):
        self._outcome(True)

    def on_timeout(self, mode, bits=0, attempt=0):
        self._outcome(False)
