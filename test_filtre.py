import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, Flatten, TimeDistributed
from typing import List
import imageio

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
    model.load_weights('./models/pretrained/checkpoint_2').expect_partial()
    return model

# Charger et prétraiter une vidéo
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])  # Recadrage spécifique
    cap.release()
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

    # Charger la vidéo
    frames = load_video(video_path)

    # Découper la vidéo en paquets de 75 frames
    for i in range(0, len(frames), 75):
        batch_frames = frames[i:i + 75]
        if len(batch_frames) < 75:
            continue  # Ignorer les paquets incomplets

        # Prédire le texte pour ce paquet de frames
        predicted_text = predict_text(model, batch_frames)
        print(f"Paquet {i // 75 + 1}: {predicted_text}")

        # Afficher les frames (optionnel)
        for frame in batch_frames:
            frame = (frame.numpy() * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Appuyer sur 'q' pour quitter
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Chemin de la vidéo à traiter
    video_path = "data/raw/lip_reading_dlib/bbaf2n.mpg"  # Remplacez par le chemin de votre vidéo

    # Traiter la vidéo
    process_video(video_path)