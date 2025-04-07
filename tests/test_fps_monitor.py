import time
import pytest
from core.fps_monitor import FPSMonitor


def test_fps_monitor_stats_update():
    fps = FPSMonitor()
    for _ in range(5):
        start = time.time()
        time.sleep(0.01)
        fps.update(start)

    stats = fps.get_stats()
    assert stats["count"] == 5
    assert stats["avg"] > 0
    assert stats["min"] > 0
    assert stats["max"] >= stats["min"]
    assert stats["current"] > 0
