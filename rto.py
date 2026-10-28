"""
rto.py : retransmission timeout (RTO) strategies used by the ARQ simulator.

Every strategy offers the same four methods, so they can be swapped without touching arq.py:
    value()          current timeout in bit times (measured from the END of the frame transmission)
    on_sample(rtt)   called with a measured round trip time
    on_timeout(now)  called when a timer expired (now = current time, optional)
    on_progress()    called when an ACK acknowledged new data but gave no usable RTT sample
"""


class FixedRTO:
    """A timeout that never changes. It must be set above the worst case round trip time."""
    name = "fixed"

    def __init__(self, value):
        self.v = float(value)

    def value(self):
        return self.v

    def on_sample(self, rtt):
        pass

    def on_timeout(self, now=None):
        pass

    def on_progress(self):
        pass


class AdaptiveRTO:
    """
    Retransmission timer of RFC 6298 (the one TCP uses).

        first sample R :  SRTT = R,  RTTVAR = R / 2
        later samples  :  RTTVAR = (1 - beta) * RTTVAR + beta * |SRTT - R|
                          SRTT   = (1 - alpha) * SRTT + alpha * R
        RTO = SRTT + max(G, K * RTTVAR)        alpha = 1/8, beta = 1/4, K = 4

      Karn's rule : the simulator never feeds in samples from retransmitted frames
                    (their ACK is ambiguous), see arq.py.
      Backoff     : a timeout doubles the RTO (up to max_rto) until a fresh sample arrives. RFC 6298
                    has ONE timer, so several frames timing out within the same RTO interval are one
                    loss event and double it only once (when `now` is given).
                    We also cancel the backoff when an ACK acknowledges new data without a
                    sample, because that proves the path works again. (Without this, Go-Back-N,
                    which resends everything and so produces no clean samples, can spiral.)
    """
    name = "adaptive (RFC 6298)"

    def __init__(self, initial, alpha=1 / 8, beta=1 / 4, k=4, granularity=10, min_rto=0.0, max_rto=None):
        self.alpha, self.beta, self.k, self.g = alpha, beta, k, granularity
        self.min_rto = min_rto
        self.max_rto = max_rto if max_rto is not None else 16.0 * initial
        self.initial = float(initial)
        self.srtt = None
        self.rttvar = None
        self.base = float(initial)
        self.backoff = 0                         # number of consecutive timeouts
        self.last_backoff = None                 # time of the last doubling

    def value(self):
        return min(self.max_rto, max(self.min_rto, self.base * (2 ** self.backoff)))

    def on_sample(self, rtt):
        if self.srtt is None:
            self.srtt, self.rttvar = rtt, rtt / 2
        else:
            self.rttvar = (1 - self.beta) * self.rttvar + self.beta * abs(self.srtt - rtt)
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * rtt
        self.base = self.srtt + max(self.g, self.k * self.rttvar)
        self.backoff, self.last_backoff = 0, None   # a fresh measurement cancels the backoff

    def on_timeout(self, now=None):
        if now is not None:
            if self.last_backoff is not None and now - self.last_backoff < self.value():
                return                           # same loss event, already backed off
            self.last_backoff = now
        if self.value() < self.max_rto:
            self.backoff += 1

    def on_progress(self):
        self.backoff, self.last_backoff = 0, None
