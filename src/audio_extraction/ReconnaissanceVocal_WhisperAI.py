# conda install -c conda-forge ffmpeg

# pip install openai-whisper
# Si ca ne marche pas, installer Rust : https://rustup.rs/; cliquer sur 1, attendre, recliquer sur 1; fermer tt les terminales puis refaire la commande openai-whisper
# conda install -c anaconda portaudio
# pip install pyaudio       via conda si ca marche pas: conda install -c anaconda pyaudio
# Verifier installation :
    # whisper --help
    # import pyaudio
    # print(pyaudio.PyAudio().get_device_count())

#si ca marche toujours aps: installer FFmpeg : https://ffmpeg.org/download.html


import whisper
import pyaudio
import wave
import numpy as np
import threading
import queue

# Charger le modèle Whisper
model = whisper.load_model("base")  # Vous pouvez choisir un autre modèle comme "small", "medium", ou "large"

# Paramètres pour l'enregistrement audio
FORMAT = pyaudio.paInt16  # Format audio
CHANNELS = 1  # Mono
RATE = 16000  # Taux d'échantillonnage
CHUNK = 1024  # Taille des chunks audio
RECORD_SECONDS = 5  # Durée de chaque enregistrement

# Queue pour stocker les chunks audio
audio_queue = queue.Queue()

# Fonction pour capturer l'audio en temps réel
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Enregistrement en cours...")

    while True:
        data = stream.read(CHUNK)
        audio_queue.put(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    p.terminate()

# Fonction pour transcrire l'audio en temps réel
def transcribe_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalisation
            result = model.transcribe(audio_data)
            print("Transcription:", result['text'])

# Démarrer les threads pour l'enregistrement et la transcription
record_thread = threading.Thread(target=record_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)

record_thread.start()
transcribe_thread.start()

record_thread.join()
transcribe_thread.join()

## A EXECUTER DANS UN TERMINAL AVEC LA COMMANDE :
#python c:\Users\le reste du chemin\\ReconnaissanceVocal_WhisperAI.py 