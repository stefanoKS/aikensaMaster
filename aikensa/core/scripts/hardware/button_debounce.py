# debounce.py (Python 3.8 safe)
import time
from typing import List

class DebouncedButton:
    def __init__(self, debounce_ms=40, require_release=True):
        self.debounce_s = debounce_ms / 1000.0
        self.require_release = require_release
        self._last_raw = 0
        self._last_change = time.monotonic()
        self._stable = 0
        self._armed = True

    def update(self, raw_level: int) -> bool:
        now = time.monotonic()

        if raw_level != self._last_raw:
            self._last_raw = raw_level
            self._last_change = now

        if (now - self._last_change) >= self.debounce_s:
            if self._stable != self._last_raw:
                self._stable = self._last_raw
                if self._stable == 1:
                    if self._armed:
                        if self.require_release:
                            self._armed = False
                        return True
                else:
                    if self.require_release:
                        self._armed = True

        return False
