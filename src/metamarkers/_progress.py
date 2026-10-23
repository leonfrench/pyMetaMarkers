"""Progress messages (verbose mode)."""

from __future__ import annotations

import time
from datetime import timedelta

#: Minimum number of seconds between two progress messages of a stage.
PROGRESS_INTERVAL = 60


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


class Log:
    """Prints messages prefixed with the time elapsed since its creation (nothing if not enabled)."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start = time.perf_counter()

    def __call__(self, message: str):
        if self.enabled:
            print(f"[{format_duration(time.perf_counter() - self.start)}] {message}", flush=True)

    def stage(self, message: str, total: int, unit: str) -> Stage:
        """Announce a long stage that processes ``total`` items (``unit``), to report its progress."""
        self(message)
        return Stage(self, total, unit)


#: Log that prints nothing (default of functions that accept a log).
QUIET = Log(enabled=False)


class Stage:
    """Progress of a long stage, reported at most every PROGRESS_INTERVAL seconds."""

    def __init__(self, log: Log, total: int, unit: str):
        self.log, self.total, self.unit = log, total, unit
        self.start = self.last_report = time.perf_counter()

    def update(self, done: int):
        """Report that ``done`` of the ``total`` items have been processed."""
        now = time.perf_counter()
        if not self.log.enabled or now - self.last_report < PROGRESS_INTERVAL or not 0 < done < self.total:
            return
        self.last_report = now
        left = (now - self.start) * (self.total - done) / done
        self.log(f"  {done:,} of {self.total:,} {self.unit} ({done / self.total:.1%}), "
                 f"about {format_duration(left)} left")

    def done(self, details: str = ""):
        self.log(f"  done in {format_duration(time.perf_counter() - self.start)}{details}")
