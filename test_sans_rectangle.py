import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit, QHBoxLayout
from PySide6.QtCore import QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import time

# from src.lip_tracking.VisualizeLip import LipTracking

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

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
        Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax')
    ])
        return model

    def preprocess_frame(self, frame):
        frame = tf.image.rgb_to_grayscale(frame)
        frame = frame[190:236, 80:220, :]
        frame = tf.image.resize(frame, (46, 140))  # Redimensionner pour correspondre à l'entrée du modèle

       # Normalisation
        # mean = tf.math.reduce_mean(frame)
        # std = tf.math.reduce_std(tf.cast(frame, tf.float32))
        # frame = tf.cast((frame - mean), tf.float32) / std
        return frame.numpy()  # Convertir en numpy array pour compatibilité avec le reste du code

    def predict(self, frames):
        # frames doit avoir la forme (75, 46, 140, 1)

        frames = np.expand_dims(frames, axis=0)  # Ajouter une dimension pour le batch (1, 75, 46, 140, 1)
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
        self.setWindowTitle("SILLDA - Transcription Labiale (Vidéo de Test)")
        self.setGeometry(100, 100, 1200, 600)
        
        # Layout principal
        self.main_layout = QHBoxLayout()
        
        # Section vidéo originale
        self.video_layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setText("Vidéo originale")
        self.video_layout.addWidget(self.video_label)
        
        # Section vidéo prétraitée (ce que voit le modèle)
        self.processed_layout = QVBoxLayout()
        self.processed_label = QLabel(self)
        self.processed_label.setText("Ce que voit le modèle")
        self.processed_layout.addWidget(self.processed_label)
        
        # Section texte
        self.text_output = QPlainTextEdit(self)
        self.text_output.setReadOnly(True)
        self.start_button = QPushButton("Démarrer la Transcription", self)
        
        # Ajouter les sections au layout principal
        self.main_layout.addLayout(self.video_layout)
        self.main_layout.addLayout(self.processed_layout)
        self.main_layout.addWidget(self.text_output)
        self.main_layout.addWidget(self.start_button)
        
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # self.lip_tracker = LipTracking()
        self.transcriber = LipReadingModel()
        self.frame_buffer = []  # Pour stocker les frames
        self.full_decoded_text = ""  # Texte transcrit complet
        self.prev_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        self.start_button.clicked.connect(self.start_video_processing)

    def start_video_processing(self):
        # Charger la vidéo de test
        video_path = "data/raw/lip_reading_dlib/bbaf2n.mpg"  # Remplacez par le chemin de votre vidéo de test
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.text_output.appendPlainText("❌ ERREUR : Impossible d'ouvrir la vidéo de test")
            return
        self.timer.start(33)  # Mettre à jour la frame toutes les 33 ms (~30 FPS)
        self.start_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            #frame = cv2.resize(frame, (320, 240))
            # processed_frame, lip_coordinates, _ = self.lip_tracker.process_frame(frame)
            self.frame_count += 1
            current_time = time.time()
            self.fps = self.frame_count / (current_time - self.prev_time)
            
            # if lip_coordinates is not None:
            # Prétraiter la frame pour le modèle
            cropped_frame = self.transcriber.preprocess_frame(frame)
            self.frame_buffer.append(cropped_frame)
            
            if len(self.frame_buffer) == 75:
                frames = np.array(self.frame_buffer)
                frames = np.expand_dims(frames, axis=-1)

                # Normalisation
                mean = tf.math.reduce_mean(frames)
                std = tf.math.reduce_std(tf.cast(frames, tf.float32))
                normalized_frames = tf.cast((frames - mean), tf.float32) / std
                normalized_frames = normalized_frames.numpy()  # Convertir en numpy pour affichage

                # Affichage séquentiel des 75 frames normalisées
                for i in range(75):
                    display_frame = normalized_frames[i, :, :, 0]  # Extraire la i-ème frame normalisée
                    display_frame = ((display_frame) * 255).astype(np.uint8)  
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)

                    h, w, ch = display_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                    self.processed_label.setPixmap(QPixmap.fromImage(qt_image))
                    QApplication.processEvents()  # Force l'affichage immédiat de chaque frame
                    time.sleep(0.03)  # Pause courte pour simuler un affichage fluide

                self.prediction_thread = PredictionThread(self.transcriber, normalized_frames)
                self.prediction_thread.result_signal.connect(self.update_transcription)
                self.prediction_thread.start()

                self.frame_buffer = []


            
            if self.frame_count % 3 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
            self.setWindowTitle(f"SILLDA - FPS: {self.fps:.2f}")
        else:
            self.timer.stop()
            self.cap.release()
            self.text_output.appendPlainText("✅ Transcription terminée.")
            self.start_button.setEnabled(True)

    def update_transcription(self, transcription):
        self.full_decoded_text += transcription + " "
        self.text_output.setPlainText(self.full_decoded_text)

if __name__ == "__main__":
    app = QApplication([])
    window = LipReadingApp()
    window.show()
    app.exec()