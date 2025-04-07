# Désactiver les erreurs liées à Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Masque les messages supplémentaires

import cv2
import time
import tensorflow as tf
import numpy as np 

from core.models.VisualizeLip import LipTracking
from core.models.LipReadingModel import LipReadingModel

class BaseStream:
    def __init__(self, fps_monitor):
        self.fps_monitor = fps_monitor
        self.lip_tracker = LipTracking()
        self.lip_reader = LipReadingModel()
        self.frames_buffer = []
        self.transcription_buffer = ""

    def frames(self):
        raise NotImplementedError
    
    def process_frame(self, frame):
        """Traite une frame avec lip tracking et lip reading"""
        # Convertir l'image en RGB pour le traitement
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Appliquer le lip tracking
        processed_frame, lip_coordinates, lip_status = self.lip_tracker.process_frame(frame_rgb)
        
        # Extraire et préparer la région des lèvres pour le lip reading
        if lip_coordinates is not None:
            x, y, w, h = lip_coordinates
            mouth_region = frame_rgb[y:y+h, x:x+w]
            mouth_resized = cv2.resize(mouth_region, (140, 46))
            mouth_resized_grey = tf.image.rgb_to_grayscale(mouth_resized)
            self.frames_buffer.append(mouth_resized_grey)
        else:
            self.frames_buffer.append(tf.zeros((46, 140, 1), dtype=tf.float32))

    def predict_text(self):
        """Prédit le texte à partir des frames de lèvres"""
        transcription = ""
        if len(self.frames_buffer) >= 75:
            frames_np = np.array(self.frames_buffer[-75:])  # Convertir la liste en tableau numpy
            normalized_frames = self.lip_reader.normalized_frames(frames_np)
            predicted_text = self.lip_reader.predict_text(normalized_frames)
            self.transcription_buffer += predicted_text + " "
            self.frames_buffer = []
            transcription = self.transcription_buffer
        
        return transcription


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

            # Appliquer la retranscription
            self.process_frame(frame)
            transcription = self.predict_text()
            if transcription:
                print(f"Transcription: {transcription}")
            
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

            # Appliquer la retranscription
            self.process_frame(frame)
            transcription = self.predict_text()
            if transcription:
                print(f"Transcription: {transcription}")

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / 30)
        cap.release()