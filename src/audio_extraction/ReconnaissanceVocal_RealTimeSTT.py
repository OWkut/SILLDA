#pip install RealtimeSTT
from RealtimeSTT import AudioToTextRecorder

def traiter_texte(texte):
    """
    Fonction de traitement du texte transcrit.
    """
    if texte.strip():  # Vérifie que le texte n'est pas vide
        print(f"Texte transcrit: {texte}")
        if "arrêt" in texte.lower() or "stop" in texte.lower():
            print("Commande d'arrêt détectée. Arrêt du programme.")
            return True
    return False

if __name__ == '__main__':
    print("Attendez jusqu'à ce que 'parlez maintenant' s'affiche")

    # Configuration de l'enregistreur avec la langue française
    enregistreur = AudioToTextRecorder(
        model="tiny",  # Modèle de transcription (tiny, base, small, medium, large)
        language="fr",  # Langue française
        enable_realtime_transcription=True,  # Transcription en temps réel
        realtime_processing_pause=0.2,  # Pause entre les traitements (en secondes)
    )

    try:
        while True:
            # Récupère le texte transcrit et le traite
            texte = enregistreur.text()
            if traiter_texte(texte):
                break
    except KeyboardInterrupt:
        print("Interruption manuelle détectée. Arrêt du programme.")
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
    finally:
        print("Nettoyage et arrêt du programme.")


## A EXECUTER DANS UN TERMINAL AVEC LA COMMANDE :
#python c:\Users\le reste du chemin\\ReconnaissanceVocal_RealTimeSTT.py