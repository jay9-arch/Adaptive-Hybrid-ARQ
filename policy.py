"""
policy.py : decides HOW each frame is sent (payload size and error control mode).

The ARQ simulator asks the policy three questions and gives it two kinds of feedback.
StaticPolicy always answers the same. In Part 6 an AdaptivePolicy will answer based on
what it has learned from the feedback, and the simulator will not need any change.
"""


class StaticPolicy:
    name = "static"

    def __init__(self, mode, payload_len=1024):
        self.mode = mode
        self.payload_len = payload_len

    def new_frame_payload(self):
        """Payload size (bits, multiple of 8) for the next NEW frame."""
        return self.payload_len

    def tx_mode(self, idx, attempt):
        """Mode for transmission number `attempt` (0 = first try) of frame `idx`."""
        return self.mode

    def on_ack(self, mode, corrected, double, bits=0, attempt=0):
        """
        Feedback: a frame sent in `mode` was accepted.
        corrected / double : Hamming statistics that came back inside the ACK
        bits               : length of that transmission on the wire
        attempt            : 0 for the first transmission of the frame, 1 for the first retry, ...
        """

    def on_timeout(self, mode, bits=0, attempt=0):
        """Feedback: a transmission in `mode` was not acknowledged in time (lost, corrupted, or ACK lost)."""

    # Optional method, used by arq.py if present:
    #   window_limit() -> maximum number of outstanding frames the policy wants right now
