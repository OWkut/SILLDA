import speech_recognition as sr

# Initialiser le reconnaisseur de la parole
recognizer = sr.Recognizer()

# Initialiser le microphone
microphone = sr.Microphone()

# Fonction pour la transcription en temps réel
def transcribe_in_real_time():
    print("Commencez à parler...")
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            print("Enregistrement...")
            audio_data = recognizer.listen(source)
            
            try:
                text = recognizer.recognize_google(audio_data, language="fr-FR")
                print("Transcription : ", text)
                
                # Arrêter si le mot "stop" est détecté
                if "stop" in text.lower():
                    print("Arrêt de la transcription.")
                    break
            except sr.UnknownValueError:
                print("L'audio n'a pas pu etre compris.")
            except sr.RequestError as e:
                print(f"Erreur de requête avec le service de reconnaissance vocale Google : {e}")
                break

# Appeler la fonction pour commencer la transcription en temps réel
transcribe_in_real_time()





















''' Premiere version, peut difficilement être adapté pour du temps reel

import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import io

# Charger la vidéo
video_path = "C:/Users/leapo/OneDrive/Documents/GitHub/SILLDA/Interface/videomp4.mp4"
video_clip = VideoFileClip(video_path)

# Extraire l'audio de la vidéo
audio_path = "C:/Users/leapo/OneDrive/Documents/GitHub/SILLDA/Interface/audio_extrait.wav"
video_clip.audio.write_audiofile(audio_path)

# Charger l'audio dans pydub
audio = AudioSegment.from_wav(audio_path)

# Initialiser le reconnaisseur de la parole
recognizer = sr.Recognizer()


# Utiliser l'audio pour la reconnaissance vocale
with sr.AudioFile(audio_path) as source:
    audio_data = recognizer.record(source)

    try:
        # Utilisation de l'API Google pour la transcription
        text = recognizer.recognize_google(audio_data, language = "fr-FR")
        print("Transcription : ", text)
    except sr.UnknownValueError:
        print("L'audio n'a pas pu être compris")
    except sr.RequestError as e:
        print(f"Erreur de requête avec le service de reconnaissance vocale Google : {e}")

'''