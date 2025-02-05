import speech_recognition as sr
from PySide6.QtCore import QObject, Signal, QThread


class SpeechTranscriber(QThread):
    transcribed_text = Signal(str)  # ✅ Signal qui envoie la transcription

    def __init__(self, language="fr-FR", stop_word="stop", interval=5):
        """
        Initialise le transcripteur vocal.

        :param language: Langue utilisée pour la reconnaissance (ex: "fr-FR").
        :param stop_word: Mot clé pour arrêter la transcription.
        :param interval: Intervalle entre chaque écoute (en secondes).
        """
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language
        self.stop_word = stop_word
        self.is_running = False
        self.interval = interval  # Intervalle entre chaque écoute

    def run(self):
        """Exécute la reconnaissance vocale en arrière-plan."""
        self.is_running = True
        print("🎤 Démarrage de la transcription... Parlez !")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("✅ Bruit ambiant ajusté. Commencez à parler...")

            while self.is_running:
                print("🎙️ Enregistrement...")

                try:
                    audio_data = self.recognizer.listen(
                        source, timeout=self.interval, phrase_time_limit=3
                    )
                    text = self.recognizer.recognize_google(
                        audio_data, language=self.language
                    )
                    print("📝 Transcription :", text)

                    self.transcribed_text.emit(text)  # ✅ Émettre la transcription

                    if self.stop_word and self.stop_word.lower() in text.lower():
                        print("🛑 Mot d'arrêt détecté. Arrêt de la transcription.")
                        self.stop_transcription()
                        break

                except sr.WaitTimeoutError:
                    print("⏳ Aucune parole détectée, réessayons...")
                except sr.UnknownValueError:
                    print("⚠️ L'audio n'a pas été compris.")
                except sr.RequestError as e:
                    print(f"❌ Erreur de connexion au service Google : {e}")
                    self.stop_transcription()
                    break

    def stop_transcription(self):
        """Arrête proprement la transcription."""
        self.is_running = False
        self.quit()
        print("🛑 Transcription arrêtée.")
