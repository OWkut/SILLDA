# Désactiver les logs de Qt (évite les messages QObject::moveToThread)
import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.info=false;*.warning=false"

# Désactiver les erreurs liées à Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Masque les messages supplémentaires

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed # type: ignore
from typing import List

# Permet de charger des modules depuis nimporte quel endroit du projet
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Récupérer le chemin absolu du dossier racine du projet
sys.path.append(project_root) # Ajouter ce chemin au sys.path

from src.lip_tracking.VisualizeLip import LipTracking

# Définir le vocabulaire et les mappages
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Charger le modèle de transcription labiale
def load_model():
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
    model.load_weights('./models/pretrained/lipread_tensorflow/checkpoint')
    return model

# Charger et prétraiter une vidéo en utilisant LipTracking
def load_video_with_liptracking(path: str) -> List[float]:
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
        processed_frame, lip_coordinates, _ = lip_tracker.process_frame(frame)

        if lip_coordinates is not None:
            x, y, w, h = lip_coordinates
            mouth_region = frame[y:y+h, x:x+w]  # Extraire la région des lèvres
            mouth_resized = cv2.resize(mouth_region, (140, 46))  # Redimensionner à la taille attendue par le modèle
            mouth_resized_grey = tf.image.rgb_to_grayscale(mouth_resized)  # Convertir en niveaux de gris
            # _____ Problème ici _____ #
            frames.append(mouth_resized_grey)
            # frame = tf.image.rgb_to_grayscale(frame)
            # frames.append(frame[190:236,80:220,:])
        else:
            # Si aucune bouche n'est détectée, ajouter une frame vide (noire)
            frames.append(tf.zeros((46, 140, 1), dtype=tf.float32))
            print("frame vide")

    cap.release()

    # Normaliser les frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

# Prédire le texte à partir des frames
def predict_text(model, frames):
    frames = np.expand_dims(frames, axis=0)  # Ajouter une dimension pour le batch
    yhat = model.predict(frames)
    decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    return tf.strings.reduce_join([num_to_char(word) for word in decoded[0]]).numpy().decode('utf-8')

# Traiter une vidéo et afficher les résultats
def process_video(video_path):
    # Charger le modèle
    model = load_model()

    # Charger la vidéo avec LipTracking
    frames = load_video_with_liptracking(video_path)

    # Découper la vidéo en paquets de 75 frames
    for i in range(0, len(frames), 75):
        batch_frames = frames[i:i + 75]
        if len(batch_frames) < 75:
            continue  # Ignorer les paquets incomplets

        # Prédire le texte pour ce paquet de frames
        predicted_text = predict_text(model, batch_frames)
        print(f"Traduction paquet {i // 75 + 1}: {predicted_text}")

        # Visualisation : afficher les frames
        for frame in batch_frames:
            frame_np = (frame.numpy() * 255).astype(np.uint8)
            frame_np_grey = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Frame mouth region', frame_np_grey)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Appuyer sur 'q' pour quitter
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chemin de la vidéo à traiter
    video_path = "data/raw/lip_reading_dlib/bbaf2n.mpg"  # Remplacez par le chemin de la vidéo

    # Traiter la vidéo
    process_video(video_path)

