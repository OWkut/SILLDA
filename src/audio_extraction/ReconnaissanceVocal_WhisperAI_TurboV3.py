import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import numpy as np
import queue
import threading
import signal
import sys
import webrtcvad
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Initialisation de Rich pour un affichage amélioré
console = Console()

# Configuration du modèle
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Charger le modèle Whisper TurboV3
model_id = "openai/whisper-large-v3"
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
    chunk_length_s=30,
    batch_size=4,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "en", "task": "transcribe"}  # Forcer la transcription en anglais
)

# Paramètres audio
RATE = 16000  # Taux d'échantillonnage (16 kHz)
CHUNK = 1024 * 4  # Taille des chunks audio (en échantillons)
BUFFER_SIZE = 16000 * 5  # 5 secondes de buffer

# Queue pour stocker les chunks audio
audio_queue = queue.Queue()

# Buffer pour accumuler les chunks audio
audio_buffer = np.array([], dtype=np.float32)

# Initialiser le VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # Mode moins strict

# Variable pour contrôler l'exécution des threads
running = True

# Fonction pour détecter la parole
def is_speech(audio_data, sample_rate=RATE):
    return vad.is_speech(audio_data.tobytes(), sample_rate)

# Fonction pour filtrer les sons de faible amplitude
def filter_low_amplitude(audio_data, threshold=0.005):  # Seuil réduit
    if np.max(np.abs(audio_data)) < threshold:
        return None
    return audio_data

# Fonction pour capturer l'audio en temps réel
def record_audio():
    console.print("[bold green]Recording in progress... Speak now.[/bold green]")
    with sd.InputStream(samplerate=RATE, channels=1, dtype=np.float32, blocksize=CHUNK) as stream:
        while running:
            try:
                audio_data, _ = stream.read(CHUNK)
                console.print(f"[bold blue]Audio data captured: {np.max(np.abs(audio_data))}[/bold blue]")  # Log pour déboguer
                audio_queue.put(audio_data.flatten())
            except Exception as e:
                console.print(f"[bold red]Error in recording: {e}[/bold red]")
                break

# Fonction pour transcrire l'audio en temps réel
def transcribe_audio():
    global audio_buffer
    while running:
        if not audio_queue.empty():
            try:
                audio_data = audio_queue.get()

                # Normalisation et formatage des données audio
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalisation pour un signal 16 bits
                audio_data = np.squeeze(audio_data)  # Supprimer les dimensions inutiles

                # Appliquer le filtre et le VAD
                audio_data = filter_low_amplitude(audio_data)
                if audio_data is not None and is_speech(audio_data):
                    # Ajouter les données audio au buffer
                    audio_buffer = np.append(audio_buffer, audio_data)

                    # Transcrire uniquement si le buffer est suffisamment grand
                    if len(audio_buffer) >= BUFFER_SIZE:
                        console.print(f"[bold blue]Processing audio buffer (size: {len(audio_buffer)})...[/bold blue]")
                        result = pipe(
                            {"raw": audio_buffer, "sampling_rate": RATE},
                            return_timestamps=True,
                            generate_kwargs={"language": "en"}
                        )
                        transcription = result["text"].strip()

                        if transcription:
                            # Affichage de la transcription avec Rich
                            rich_text = Text(transcription, style="bold yellow")
                            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
                            console.print(panel)

                        audio_buffer = np.array([], dtype=np.float32)  # Réinitialiser le buffer
            except Exception as e:
                console.print(f"[bold red]Error in transcription: {e}[/bold red]")

# Gestion de l'arrêt propre avec Ctrl+C
def signal_handler(sig, frame):
    global running
    console.print("[bold red]Stopping transcription...[/bold red]")
    running = False
    sys.exit(0)

# Démarrer les threads pour l'enregistrement et la transcription
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)  # Capturer Ctrl+C

    record_thread = threading.Thread(target=record_audio)
    transcribe_thread = threading.Thread(target=transcribe_audio)

    record_thread.start()
    transcribe_thread.start()

    record_thread.join()
    transcribe_thread.join()