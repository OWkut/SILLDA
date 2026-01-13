# Référence API - SILLDA

## Vue d'ensemble

Ce document fournit une référence complète de l'API du projet SILLDA.

## Modules principaux

### 1. `src.lip_tracking.VisualizeLip`

#### Classe `LipTracking`

Classe principale pour la détection et le suivi des lèvres.

##### `__init__(self)`

Initialise le détecteur de visage et le prédicteur de points clés.

**Paramètres** : Aucun

**Raises** :
- `FileNotFoundError` : Si le fichier `shape_predictor_68_face_landmarks.dat` n'est pas trouvé

**Exemple** :
```python
from src.lip_tracking.VisualizeLip import LipTracking

tracker = LipTracking()
```

##### `process_frame(self, frame)`

Traite une frame vidéo pour détecter et extraire la région labiale.

**Paramètres** :
- `frame` (numpy.ndarray) : Frame vidéo en format BGR (OpenCV) ou RGB, shape (H, W, 3)

**Retourne** :
- `tuple` : (processed_frame, lip_status)
  - `processed_frame` (numpy.ndarray) : Frame avec rectangle de détection dessiné
  - `lip_status` (str) : Statut de détection ("Détecté", "Non détecté", "Hors champ")

**Exemple** :
```python
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

processed_frame, status = tracker.process_frame(frame)
print(f"Statut : {status}")
```

---

### 2. `src.lip_reading.transcription`

#### Variables globales

##### `vocab`
Liste des caractères du vocabulaire : `"abcdefghijklmnopqrstuvwxyz'?!123456789 "`

##### `char_to_num`
`tf.keras.layers.StringLookup` : Mappage caractère → nombre

##### `num_to_char`
`tf.keras.layers.StringLookup` : Mappage nombre → caractère (inverse de char_to_num)

##### `model`
Modèle TensorFlow/Keras pré-entraîné pour la reconnaissance labiale.

**Architecture** : Conv3D + Bidirectional LSTM

**Input shape** : (75, 46, 140, 1)

**Output shape** : (75, vocab_size + 1)

#### `load_video(path: str) -> List[float]`

Charge et prétraite une vidéo pour l'inférence.

**Paramètres** :
- `path` (str) : Chemin vers le fichier vidéo

**Retourne** :
- `List[float]` : Séquences de frames normalisées, shape (num_frames, 46, 140, 1)

**Exemple** :
```python
from src.lip_reading.transcription import load_video

frames = load_video("data/raw/sample_video.mp4")
```

#### `load_alignments(path: str) -> List[str]`

Charge les alignements depuis un fichier texte.

**Paramètres** :
- `path` (str) : Chemin vers le fichier d'alignements

**Retourne** :
- `List[str]` : Tokens encodés numériquement

**Note** : Fonction utilisée principalement pour l'entraînement.

#### `load_data(path: str)`

Charge les données vidéo depuis un chemin TensorFlow.

**Paramètres** :
- `path` : TensorFlow tensor contenant le chemin

**Retourne** :
- Frames vidéo prétraitées

**Note** : Utilisé dans les pipelines TensorFlow (tf.data).

---

### 3. `src.audio_extraction.ReconnaissanceVocal_RealTimeSTT`

#### Variables globales

##### `EXTENDED_LOGGING`
`bool` : Active/désactive les logs étendus (défaut: False)

##### `WRITE_TO_KEYBOARD_INTERVAL`
`float` : Intervalle d'écriture au clavier en secondes (défaut: 0.002)

##### `end_of_sentence_detection_pause`
`float` : Pause pour détecter la fin de phrase (défaut: 0.3)

##### `unknown_sentence_detection_pause`
`float` : Pause par défaut (défaut: 0.5)

##### `mid_sentence_detection_pause`
`float` : Pause pour phrases incomplètes (défaut: 1.5)

#### `clean_text(text: str) -> str`

Nettoie le texte en supprimant les caractères non imprimables.

**Paramètres** :
- `text` (str) : Texte à nettoyer

**Retourne** :
- `str` : Texte nettoyé

#### `preprocess_text(text: str) -> str`

Prétraite le texte : suppression des ellipses, capitalisation.

**Paramètres** :
- `text` (str) : Texte à prétraiter

**Retourne** :
- `str` : Texte prétraité

#### `text_detected(text: str)`

Callback appelé lors de la détection de texte en temps réel.

**Paramètres** :
- `text` (str) : Texte détecté (peut être incomplet)

**Effets de bord** :
- Affiche le texte dans la console avec Rich
- Met à jour l'affichage en temps réel

#### `process_text(text: str)`

Traite le texte finalisé (phrase complète).

**Paramètres** :
- `text` (str) : Texte finalisé

**Effets de bord** :
- Ajoute le texte à la liste des phrases complètes
- Écrit le texte au clavier si `WRITE_TO_KEYBOARD_INTERVAL` > 0

#### Configuration du recorder

```python
recorder_config = {
    'model': 'medium',  # Modèle Whisper
    'language': 'en',   # Langue
    'enable_realtime_transcription': True,
    'realtime_processing_pause': 0.02,
    'post_speech_silence_duration': unknown_sentence_detection_pause,
    'on_realtime_transcription_update': text_detected,
    'no_log_file': True,
    'initial_prompt': "..."  # Prompt pour guider le modèle
}
```

---

### 4. `Interface.View.MainWindoUi`

#### Classe `MainWindowUi(QMainWindow)`

Fenêtre principale de l'interface graphique.

##### `__init__(self)`

Initialise la fenêtre principale et tous ses composants.

**Effets de bord** :
- Crée les widgets de l'interface
- Charge les styles QSS
- Initialise le lip tracker

##### `load_styles(self)`

Charge et applique les styles depuis le fichier QSS.

**Raises** :
- Affiche un message d'erreur si le fichier QSS n'est pas trouvé

##### `toggle_menu(self)`

Affiche ou masque le menu latéral avec animation.

##### `handle_toggle_webcam(self)`

Active ou désactive la capture vidéo depuis la webcam.

**Effets de bord** :
- Démarre/arrête le timer de mise à jour
- Ouvre/ferme la capture vidéo

##### `update_frame(self)`

Met à jour la frame affichée et applique le lip tracking.

**Appelé par** : Timer Qt (toutes les 30ms)

**Effets de bord** :
- Capture une nouvelle frame
- Applique le lip tracking
- Met à jour l'affichage

---

## Utilisation avancée

### Pipeline complet de reconnaissance labiale

```python
import cv2
import tensorflow as tf
from src.lip_tracking.VisualizeLip import LipTracking
from src.lip_reading.transcription import model, load_video

# 1. Initialiser le tracker
tracker = LipTracking()

# 2. Capturer une vidéo
cap = cv2.VideoCapture(0)
frames = []
for _ in range(75):
    ret, frame = cap.read()
    if ret:
        processed_frame, status = tracker.process_frame(frame)
        # Extraire la région labiale (à implémenter)
        frames.append(lip_region)
cap.release()

# 3. Prétraiter les frames
frames = tf.stack(frames)
frames = normalize(frames)  # Normalisation

# 4. Prédire
predictions = model.predict(tf.expand_dims(frames, axis=0))

# 5. Décoder
decoded = tf.keras.backend.ctc_decode(
    predictions, 
    input_length=[75], 
    greedy=True
)[0][0].numpy()
```

### Intégration avec l'interface

```python
from PySide6.QtWidgets import QApplication
from Interface.View.MainWindoUi import MainWindowUi
import sys

app = QApplication(sys.argv)
window = MainWindowUi()
window.show()
app.exec()
```

---

## Notes importantes

1. **Chemins de fichiers** : Les chemins vers les modèles et données doivent être configurés correctement
2. **GPU** : Le modèle de reconnaissance labiale bénéficie grandement d'un GPU
3. **Threading** : Pour le traitement en temps réel, considérer l'utilisation de threads séparés
4. **Mémoire** : Les modèles peuvent être gourmands en mémoire, surveiller l'utilisation

---

## Exemples complets

Voir les fichiers dans `examples/` (à créer) pour des exemples d'utilisation complets.

