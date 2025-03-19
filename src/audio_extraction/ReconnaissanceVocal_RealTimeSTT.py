from RealtimeSTT import AudioToTextRecorder
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import pyautogui
import os

# Configuration globale
EXTENDED_LOGGING = False
WRITE_TO_KEYBOARD_INTERVAL = 0.002

# Initialisation de Rich
console = Console()

# Variables globales pour la gestion du texte
full_sentences = []
displayed_text = ""
prev_text = ""

# Délais de détection de pause
end_of_sentence_detection_pause = 0.3  # Réduit pour une meilleure réactivité
unknown_sentence_detection_pause = 0.5
mid_sentence_detection_pause = 1.5

def clean_text(text):
    return ''.join(char for char in text if char.isprintable())

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def preprocess_text(text):
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:]
    text = text.lstrip()
    if text:
        text = text[0].upper() + text[1:]
    return clean_text(text)

def text_detected(text):
    global prev_text, displayed_text, full_sentences

    text = preprocess_text(text)

    sentence_end_marks = ['.', '!', '?', '。']
    if text.endswith("..."):
        recorder.post_speech_silence_duration = mid_sentence_detection_pause
    elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
        recorder.post_speech_silence_duration = end_of_sentence_detection_pause
    else:
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

    prev_text = text

    rich_text = Text()
    for i, sentence in enumerate(full_sentences):
        rich_text.append(sentence, style="yellow" if i % 2 == 0 else "cyan")
        rich_text.append(" ")

    if text:
        rich_text.append(text, style="bold yellow")

    new_displayed_text = rich_text.plain
    if new_displayed_text != displayed_text:
        displayed_text = new_displayed_text
        panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
        console.print(panel)

def process_text(text):
    global full_sentences, prev_text

    text = preprocess_text(text)
    text = text.rstrip()
    if text.endswith("..."):
        text = text[:-2]

    if not text:
        return

    if not text.isascii():
        return

    full_sentences.append(text)
    prev_text = ""
    text_detected("")

    if WRITE_TO_KEYBOARD_INTERVAL:
        pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)

if __name__ == '__main__':
    print("System initializing, please wait...")

    recorder_config = {
        'model': 'medium',  # Utilisation d'un modèle plus léger
        'language': 'en',
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'post_speech_silence_duration': unknown_sentence_detection_pause,
        'on_realtime_transcription_update': text_detected,
        'no_log_file': True,
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        )
    }

    recorder = AudioToTextRecorder(**recorder_config)

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
    finally:
        print("Cleaning up and stopping the program.")