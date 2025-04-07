import time
import math
from threading import Lock


class FPSMonitor:
    def __init__(self):
        self.lock = Lock()
        self.reset()

    def reset(self):
        self.stats = {
            "current": 0.0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "count": 0,
            "total": 0.0,
        }

    def update(self, start_time):
        end_time = time.time()
        delta = end_time - start_time
        if delta == 0:
            return
        fps = 1.0 / delta
        with self.lock:
            s = self.stats
            s["current"] = round(fps, 2)
            s["min"] = fps if s["count"] == 0 else min(s["min"], fps)
            s["max"] = max(s["max"], fps)
            s["total"] += fps
            s["count"] += 1
            s["avg"] = s["total"] / s["count"]

    def get_stats(self):
        with self.lock:
            return {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in self.stats.items()
            }
