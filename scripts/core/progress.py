"""One progress line, the same shape in every long running script.

    p = Progress(len(work), "task sets")
    for i, item in enumerate(work, start=1):
        ...
        p.tick(i)
    p.close()

Reports on a timer rather than every N items, since these scripts range from
milliseconds to minutes per item. The remaining estimate is a running average.
"""

import sys
import time


def hms(seconds):
    s = int(max(seconds, 0))
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


class Progress:
    def __init__(self, total, unit="", every=10.0, stream=sys.stdout):
        self.total = max(int(total), 1)
        self.unit = unit
        self.every = every
        self.stream = stream
        self.t0 = time.time()
        self.last = self.t0
        self.n = 0

    def tick(self, done=None, extra=""):
        """Advance, printing if the interval has passed or this is the last one."""
        self.n = self.n + 1 if done is None else done
        now = time.time()
        if now - self.last < self.every and self.n < self.total:
            return
        self.last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        left = (self.total - self.n) / rate if rate > 0 else 0.0
        line = (f"  {self.n}/{self.total} {self.unit}".rstrip()
                + f"  {100 * self.n / self.total:.0f}%"
                + f"  {hms(el)} elapsed"
                + f"  {hms(left)} left"
                + (f"  {rate:.1f}/s" if rate >= 1 else f"  {60 * rate:.1f}/min"))
        if extra:
            line += f"  {extra}"
        print(line, file=self.stream, flush=True)

    def close(self, note=""):
        el = time.time() - self.t0
        print(f"  done in {hms(el)}" + (f"  {note}" if note else ""),
              file=self.stream, flush=True)
        return el
