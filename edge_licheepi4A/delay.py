import time

class TokenBucket(object):
    def __init__(self, tokens, fill_rate):
        """tokens is the total tokens in the bucket. fill_rate is the rate in tokens/second that the bucket will be refilled."""
        self.capacity = float(tokens)
        self._tokens = float(tokens)
        self.fill_rate = float(fill_rate)
        self.timestamp = time.time()

    def consume(self, tokens):
        """Consume tokens from the bucket. Returns 0 if there were sufficient tokens, otherwise the expected time until enough tokens become available."""
        if tokens <= self.tokens:
            self._tokens -= tokens
            return 0
        else:
            return (tokens - self.tokens) / self.fill_rate

    @property
    def tokens(self):
        if self._tokens < self.capacity:
            now = time.time()
            delta = self.fill_rate * (now - self.timestamp)
            self._tokens = min(self.capacity, self._tokens + delta)
            self.timestamp = now
        return self._tokens
