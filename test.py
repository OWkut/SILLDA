import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit
from PySide6.QtCore import QTimer
from src.lip_tracking.VisualizeLip import LipTracking
from PySide6.QtGui import QImage, QPixmap


class LipReadingModel:
    def __init__(self):
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)
        self.model = self.build_model()
        self.model.load_weights('./models/pretrained/checkpoint_2').expect_partial()
    
    def build_model(self):
        model = Sequential()

        model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(256, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(75, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(TimeDistributed(Flatten()))

        model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
        model.add(Dropout(.5))

        model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
        return model

    def preprocess_frame(self, frame, lip_coordinates):
        x, y, w, h = lip_coordinates
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[y:y+h, x:x+w]
        frame = cv2.resize(frame, (140, 46))  # Resize to match model input size
        frame = frame / 255.0  # Normalisation simple au lieu de z-score
        return frame

    def predict(self, frames):
        # frames doit avoir la forme (75, 46, 140, 1)
        frames = np.expand_dims(frames, axis=0)  # Ajouter une dimension pour le batch (1, 75, 46, 140, 1)
        yhat = self.model.predict(frames)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        return "".join([tf.strings.reduce_join([self.num_to_char(word) for word in sentence]).numpy().decode('utf-8') for sentence in decoded])


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
        self.frame_buffer = []  # Buffer pour stocker les frames
        self.full_decoded_text = ""  # Texte transcrit complet
        
        self.toggle_webcam_button.clicked.connect(self.toggle_webcam)
    
    def toggle_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.text_output.appendPlainText("❌ ERREUR : Impossible d'accéder à la webcam")
                self.cap = None
                return
            self.timer.start(50)
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
            # Traitement du frame avec le lip tracking
            processed_frame, lip_coordinates, lip_status = self.lip_tracker.process_frame(frame)
            
            if lip_coordinates is not None:
                # Prétraitement du frame pour le lip reading
                cropped_frame = self.transcriber.preprocess_frame(processed_frame, lip_coordinates)
                
                # Ajouter le frame prétraité au buffer
                self.frame_buffer.append(cropped_frame)
                
                # Si le buffer contient 75 frames, prédire le texte
                if len(self.frame_buffer) == 75:
                    # Convertir le buffer en un tableau numpy de forme (75, 46, 140, 1)
                    frames = np.array(self.frame_buffer)  # Forme (75, 46, 140)
                    frames = np.expand_dims(frames, axis=-1)  # Ajouter une dimension pour le canal (75, 46, 140, 1)
                    
                    # Prédire le texte
                    transcription = self.transcriber.predict(frames)
                    self.full_decoded_text += transcription + " "
                    self.text_output.setPlainText(self.full_decoded_text)
                    
                    # Réinitialiser le buffer
                    self.frame_buffer = []
            
            if processed_frame is not None:
                # Convertir l'image BGR en RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Vérifier la forme de l'image
                h, w, ch = processed_frame.shape
                if h == 0 or w == 0:
                    print("Erreur: L'image traitée est vide")
                    return  # Sortir si l'image est vide
                
                # Afficher l'image traitée dans l'interface
                bytes_per_line = ch * w
                qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.webcam_label.setPixmap(QPixmap.fromImage(qt_image))


if __name__ == "__main__":
    app = QApplication([])
    window = LipReadingApp()
    window.show()
    app.exec()