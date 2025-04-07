import cv2
import time


class BaseStream:
    def __init__(self, fps_monitor):
        self.fps_monitor = fps_monitor

    def frames(self):
        raise NotImplementedError


class WebcamStream(BaseStream):
    def frames(self):
        camera = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            success, frame = camera.read()
            if not success:
                break
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            self.fps_monitor.update(start_time)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / 30)
        camera.release()


class FileVideoStream(BaseStream):
    def __init__(self, path, fps_monitor):
        super().__init__(fps_monitor)
        self.path = path

    def frames(self):
        cap = cv2.VideoCapture(self.path)
        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            self.fps_monitor.update(start_time)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / 30)
        cap.release()