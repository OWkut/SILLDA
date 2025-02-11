import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import queue
import threading

# Configuration du modèle
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Charger le modèle Whisper
model_id = "openai/whisper-large-v3"  # Remplacez par "openai/whisper-large-v3-turbo" si disponible
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

# Charger le processeur
processor = AutoProcessor.from_pretrained(model_id)

# Créer le pipeline de transcription
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Longueur des chunks audio (en secondes)
    batch_size=4,  # Taille du batch pour l'inférence
    torch_dtype=torch_dtype,
    device=device,
)

# Paramètres audio
RATE = 16000  # Taux d'échantillonnage (16 kHz)
CHUNK = 1024 * 4  # Taille des chunks audio (en échantillons)
BUFFER_SIZE = 16000 * 10  # 10 secondes de buffer

# Queue pour stocker les chunks audio
audio_queue = queue.Queue()

# Buffer pour accumuler les chunks audio
audio_buffer = np.array([], dtype=np.float32)

# Fonction pour capturer l'audio en temps réel
def record_audio():
    print("Enregistrement en cours... Parlez maintenant !")
    with sd.InputStream(samplerate=RATE, channels=1, dtype=np.float32, blocksize=CHUNK) as stream:
        while True:
            audio_data, _ = stream.read(CHUNK)
            audio_queue.put(audio_data.flatten())

# Fonction pour transcrire l'audio en temps réel
def transcribe_audio():
    global audio_buffer
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()

            # Normalisation et formatage des données audio
            audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))  # Normalisation
            audio_data = np.squeeze(audio_data)  # Supprimer les dimensions inutiles

            # Ajouter les données audio au buffer
            audio_buffer = np.append(audio_buffer, audio_data)

            # Transcrire uniquement si le buffer est suffisamment grand
            if len(audio_buffer) >= BUFFER_SIZE:
                result = pipe(
                    {"raw": audio_buffer, "sampling_rate": RATE},  # Format d'entrée correct
                    return_timestamps=True,
                    generate_kwargs={"language": "fr"}  # Spécifier la langue avec generate_kwargs
                )
                print("Transcription:", result["text"])
                audio_buffer = np.array([], dtype=np.float32)  # Réinitialiser le buffer

# Démarrer les threads pour l'enregistrement et la transcription
record_thread = threading.Thread(target=record_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)

record_thread.start()
transcribe_thread.start()

record_thread.join()
transcribe_thread.join()