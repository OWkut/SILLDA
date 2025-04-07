import time
from threading import Lock
from core.fps_monitor import FPSMonitor
from core.sources import WebcamStream, FileVideoStream


class StreamManager:
    def __init__(self):
        self.fps_monitor = FPSMonitor()
        self.stream = None
        self.source = None
        self.is_streaming = True
        self.lock = Lock()

    def start_webcam(self):
        self.fps_monitor.reset()
        self.stream = WebcamStream(self.fps_monitor)
        self.source = "webcam"

    def start_video(self, path):
        self.fps_monitor.reset()
        self.stream = FileVideoStream(path, self.fps_monitor)
        self.source = path

    def pause(self):
        with self.lock:
            self.is_streaming = False

    def resume(self):
        with self.lock:
            self.is_streaming = True

    def get_fps(self):
        return self.fps_monitor.get_stats()

    def generate(self):
        while True:
            with self.lock:
                if not self.is_streaming:
                    time.sleep(0.1)
                    continue
            for frame in self.stream.frames():
                with self.lock:
                    if not self.is_streaming:
                        break
                yield frame


stream_manager = StreamManager()
