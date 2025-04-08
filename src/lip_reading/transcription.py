import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

class LipReadingModel:
    def __init__(self, model_path='./models/pretrained/checkpoint'):
        self.vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, oov_token="")
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True)
        
        self.model = self.build_model()
        self.model.load_weights(model_path).expect_partial()

    def build_model(self):
        model = Sequential()
        model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D(pool_size=(1, 2, 2)))

        model.add(Conv3D(256, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D(pool_size=(1, 2, 2)))

        model.add(Conv3D(75, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPool3D(pool_size=(1, 2, 2)))

        model.add(TimeDistributed(Flatten()))
        
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        
        model.add(Dense(self.char_to_num.vocabulary_size() + 1, activation='softmax'))
        return model

    def preprocess_frame(self, frame):
        """Prétraite une frame pour l'adapter au modèle."""
        frame = tf.image.rgb_to_grayscale(frame)
        frame = frame[190:236, 80:220, :]
        return frame

    def predict(self, frames):
        """Prédit le texte à partir d'une séquence de frames."""
        frames = np.expand_dims(frames, axis=0)  # Ajouter la dimension batch
        yhat = self.model.predict(frames)
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        transcription = tf.strings.reduce_join([self.num_to_char(word) for word in decoded])
        return transcription.numpy().decode('utf-8')
