import whisper
import pyaudio
import numpy as np
import queue
import threading
import signal
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Charger le modèle Whisper
model = whisper.load_model("base")

# Paramètres pour l'enregistrement audio
FORMAT = pyaudio.paInt16  # Format audio
CHANNELS = 1  # Mono
RATE = 16000  # Taux d'échantillonnage
CHUNK = 1024  # Taille des chunks audio

# Queue pour stocker les chunks audio
audio_queue = queue.Queue()

# Initialisation de Rich pour un affichage amélioré
console = Console()

# Variable pour contrôler l'exécution des threads
running = True

# Historique des transcriptions
transcription_history = []

# Fonction pour capturer l'audio en temps réel
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    console.print("[bold green]Recording in progress... Speak now.[/bold green]")

    while running:
        try:
            data = stream.read(CHUNK)
            audio_queue.put(np.frombuffer(data, dtype=np.int16))
        except Exception as e:
            console.print(f"[bold red]Error in recording: {e}[/bold red]")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

# Fonction pour transcrire l'audio en temps réel
def transcribe_audio():
    global transcription_history
    audio_buffer = []
    buffer_size = 5 * RATE  # 5 secondes d'audio

    while running:
        if not audio_queue.empty():
            try:
                audio_data = audio_queue.get()
                audio_buffer.extend(audio_data)

                if len(audio_buffer) >= buffer_size:
                    audio_segment = np.array(audio_buffer[:buffer_size], dtype=np.float32) / 32768.0
                    audio_buffer = audio_buffer[buffer_size:]  # Réinitialiser le buffer

                    # Transcrire le segment audio
                    try:
                        result = model.transcribe(audio_segment, language="en", fp16=False)
                        text = result['text'].strip()

                        if text:
                            transcription_history.append(text)
                            if len(transcription_history) > 5:  # Garder les 5 dernières transcriptions
                                transcription_history.pop(0)

                            # Affichage de l'historique
                            rich_text = Text("\n".join(transcription_history), style="bold yellow")
                            panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
                            console.print(panel)
                    except Exception as e:
                        console.print(f"[bold red]Error in transcription: {e}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]Error in processing audio: {e}[/bold red]")

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