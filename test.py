import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit
from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import time

from src.lip_tracking.VisualizeLip import LipTracking

class LipReadingModel:
    def __init__(self):
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)
        self.model = self.build_model()
        self.model.load_weights('./models/pretrained/checkpoint_2').expect_partial()
    
    def build_model(self):
        model = Sequential([
            Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same', activation='relu'),
            MaxPool3D((1, 2, 2)),
            Conv3D(256, 3, padding='same', activation='relu'),
            MaxPool3D((1, 2, 2)),
            Conv3D(75, 3, padding='same', activation='relu'),
            MaxPool3D((1, 2, 2)),
            TimeDistributed(Flatten()),
            Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            Dropout(.5),
            Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)),
            Dropout(.5),
            Dense(41, kernel_initializer='he_normal', activation='softmax')
        ])
        return model

    def preprocess_frame(self, frame, lip_coordinates):
        x, y, w, h = lip_coordinates
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[y:y+h, x:x+w]
        frame = cv2.resize(frame, (140, 46)) # Resize to match model input size
        frame = frame / 255.0
        return frame

    def predict(self, frames):
        # frames doit avoir la forme (75, 46, 140, 1)
        frames = np.expand_dims(frames, axis=0) # Ajouter une dimension pour le batch (1, 75, 46, 140, 1)
        yhat = self.model.predict(frames)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        return "".join([tf.strings.reduce_join([self.num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded])

class PredictionThread(QThread):
    result_signal = Signal(str)
    def __init__(self, transcriber, frames):
        super().__init__()
        self.transcriber = transcriber
        self.frames = frames
    def run(self):
        transcription = self.transcriber.predict(self.frames)
        self.result_signal.emit(transcription)

class LipReadingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SILLDA - Transcription Labiale")
        self.setGeometry(100, 100, 900, 600)
        
        self.layout = QVBoxLayout()
        self.webcam_label = QLabel(self)
        self.text_output = QPlainTextEdit(self)
        self.text_output.setReadOnly(True)
        self.toggle_webcam_button = QPushButton("Activer la Webcam", self)
        
        self.layout.addWidget(self.webcam_label)
        self.layout.addWidget(self.text_output)
        self.layout.addWidget(self.toggle_webcam_button)
        
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.lip_tracker = LipTracking()
        self.transcriber = LipReadingModel()
        self.frame_buffer = [] # Pour stocker les frames
        self.full_decoded_text = "" # Texte transcrit complet
        self.prev_time = time.time()
        self.fram_count = 0
        self.fps = 0
        
        self.toggle_webcam_button.clicked.connect(self.toggle_webcam)
    
    def toggle_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            if not self.cap.isOpened():
                self.text_output.appendPlainText("❌ ERREUR : Impossible d'accéder à la webcam")
                self.cap = None
                return
            self.timer.start(33)
            self.toggle_webcam_button.setText("Désactiver la Webcam")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.webcam_label.clear()
            self.toggle_webcam_button.setText("Activer la Webcam")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            processed_frame, lip_coordinates, _ = self.lip_tracker.process_frame(frame)
            self.fram_count += 1
            current_time = time.time()
            self.fps = self.fram_count / (current_time - self.prev_time)
            
            if lip_coordinates is not None:
                cropped_frame = self.transcriber.preprocess_frame(processed_frame, lip_coordinates)
                self.frame_buffer.append(cropped_frame)
                
                if len(self.frame_buffer) == 75:
                    frames = np.array(self.frame_buffer)
                    frames = np.expand_dims(frames, axis=-1)
                    
                    self.prediction_thread = PredictionThread(self.transcriber, frames)
                    self.prediction_thread.result_signal.connect(self.update_transcription)
                    self.prediction_thread.start()
                    
                    self.frame_buffer = []
            
            if self.fram_count % 3 == 0:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = processed_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.webcam_label.setPixmap(QPixmap.fromImage(qt_image))
            
            self.setWindowTitle(f"SILLDA - FPS: {self.fps:.2f}")

    def update_transcription(self, transcription):
        self.full_decoded_text += transcription + " "
        self.text_output.setPlainText(self.full_decoded_text)

if __name__ == "__main__":
    app = QApplication([])
    window = LipReadingApp()
    window.show()
    app.exec()
