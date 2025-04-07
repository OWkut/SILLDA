import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed # type: ignore
import numpy as np
import cv2
from typing import List
from core.models.VisualizeLip import LipTracking

# ==================== Classe pour le modèle de transcription labiale ==================== #
class LipReadingModel:
    def __init__(self):
        # Définir le vocabulaire et les mappages
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)
        
        # Construire et charger le modèle de transcription labiale
        self.model = self.build_model()
        self.model.load_weights('./core/models/pretrained/checkpoint')
    
    def build_model(self) -> Sequential: 
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


    # Charger et prétraiter une vidéo en utilisant LipTracking
    def load_video_with_liptracking(self, path: str) -> List[float]:
        cap = cv2.VideoCapture(path)
        lip_tracker = LipTracking()  # Initialiser le LipTracking
        frames = []                  # Initialiser liste pour stocker les frames prétraitées
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Nombre total de frames dans la vidéo

        # Parcourir chaque frame de la vidéo
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Traiter la frame avec LipTracking
            processed_frame, lip_coordinates, lip_status = lip_tracker.process_frame(frame)

            if lip_coordinates is not None:
                x, y, w, h = lip_coordinates
                mouth_region = frame[y:y+h, x:x+w]  # Extraire la région des lèvres
                mouth_resized = cv2.resize(mouth_region, (140, 46))  # Redimensionner à la taille attendue par le modèle
                mouth_resized_grey = tf.image.rgb_to_grayscale(mouth_resized)  # Convertir en niveaux de gris

                frames.append(mouth_resized_grey)
                # frame = tf.image.rgb_to_grayscale(frame)
                # frames.append(frame[190:236,80:220,:])
            else:
                # Si aucune bouche n'est détectée, ajouter une frame vide (noire)
                frames.append(tf.zeros((46, 140, 1), dtype=tf.float32))
                print("frame vide")

        cap.release()

        return frames

    # Normaliser les frames
    def normalized_frames(self, frames):
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        return tf.cast((frames - mean), tf.float32) / std

    # Prédire le texte à partir des frames
    def predict_text(self, frames):
        frames = self.normalized_frames(frames)  # Normaliser les frames
        frames = np.expand_dims(frames, axis=0)  # Ajouter une dimension pour le batch
        yhat = self.model.predict(frames)
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        return tf.strings.reduce_join(self.num_to_char(decoder)).numpy().decode('utf-8')


    # Charger et prétraiter une vidéo de + de 75 frames en utilisant LipTracking
    def load_long_video_with_liptracking(self, path: str) -> List[List[float]]:
        cap = cv2.VideoCapture(path)
        lip_tracker = LipTracking()  # Initialiser le LipTracking
        frames = []  # Liste pour stocker toutes les frames prétraitées
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Nombre total de frames dans la vidéo

        # Parcourir chaque frame de la vidéo
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Traiter la frame avec LipTracking
            processed_frame, lip_coordinates, lip_status = lip_tracker.process_frame(frame)

            if lip_coordinates is not None:
                x, y, w, h = lip_coordinates
                mouth_region = frame[y:y+h, x:x+w]  # Extraire la région des lèvres
                mouth_resized = cv2.resize(mouth_region, (140, 46))  # Redimensionner à la taille attendue par le modèle
                mouth_resized_grey = tf.image.rgb_to_grayscale(mouth_resized)  # Convertir en niveaux de gris
                mouth_resized_grey = tf.cast(mouth_resized_grey, tf.float32)  # S'assurer du bon type
                frames.append(mouth_resized_grey)
            else:
                # Ajouter une frame noire si aucune bouche n'est détectée
                frames.append(tf.zeros((46, 140, 1), dtype=tf.float32))
                print("frame vide")

        cap.release()

        # Découper en segments de 75 frames
        num_segments = (len(frames) + 74) // 75  # Nombre total de segments
        frame_segments = []

        for i in range(num_segments):
            start = i * 75
            end = start + 75
            segment = frames[start:end]

            # Compléter avec des frames noires si le segment est incomplet
            while len(segment) < 75:
                segment.append(tf.zeros((46, 140, 1), dtype=tf.float32))

            frame_segments.append(segment)
            # frame_segments.append(tf.stack(segment))  # Convertir la liste en un tenseur homogène avant stockage

        return frame_segments  # Retourne une liste de segments, chaque segment contenant 75 frames

    # Prédire le texte à partir des segments de frames
    def predict_long_text(self, frame_segments):
        full_text = ""

        for segment in frame_segments:
            segment = self.normalized_frames(segment)  # Normaliser
            segment = np.expand_dims(segment, axis=0)  # Ajouter la dimension batch
            yhat = self.model.predict(segment)
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            text = tf.strings.reduce_join(self.num_to_char(decoder)).numpy().decode('utf-8')

            full_text += text + " "  # Ajouter un espace entre chaque segment

        return full_text.strip()
