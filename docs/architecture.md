# Architecture du Système SILLDA

## Vue d'ensemble

SILLDA est un système multimodale qui combine la reconnaissance labiale visuelle et la transcription vocale pour créer un pipeline de transcription robuste.

## Architecture générale

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Utilisateur                    │
│                      (PySide6 / Qt)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐          ┌─────────▼──────────┐
│  Pipeline      │          │  Pipeline           │
│  Visuel        │          │  Audio             │
│                │          │                    │
│  • Webcam      │          │  • Microphone      │
│  • Détection   │          │  • VAD             │
│    faciale     │          │  • Whisper/STT     │
│  • Extraction  │          │  • Post-traitement │
│    labiale     │          │                    │
│  • Modèle      │          │                    │
│    Conv3D+LSTM │          │                    │
└───────┬────────┘          └─────────┬──────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
              ┌────────▼────────┐
              │  Fusion         │
              │  Multimodale    │
              │  (Futur)        │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Transcription  │
              │  Finale         │
              └─────────────────┘
```

## Composants détaillés

### 1. Module de Suivi Labial

**Fichier** : `src/lip_tracking/VisualizeLip.py`

**Responsabilités** :
- Détection des visages dans les frames vidéo
- Extraction des points clés du visage (68 points dlib)
- Focus sur les 20 points labiaux (indices 48-68)
- Calcul adaptatif de la région d'intérêt (ROI)
- Normalisation et recadrage de la région labiale

**Flux de traitement** :
```
Frame vidéo
    ↓
Détection faciale (dlib)
    ↓
Extraction des 68 points du visage
    ↓
Sélection des 20 points labiaux
    ↓
Calcul de la bounding box avec bordure
    ↓
Recadrage et normalisation
    ↓
Frame labiale normalisée
```

### 2. Module de Reconnaissance Labiale

**Fichier** : `src/lip_reading/transcription.py`

**Architecture du modèle** :

```
Input: (batch_size, 75, 46, 140, 1)
    ↓
Conv3D(filters=128, kernel_size=3, padding='same')
    + Activation('relu')
    + MaxPool3D(pool_size=(1, 2, 2))
    ↓
Conv3D(filters=256, kernel_size=3, padding='same')
    + Activation('relu')
    + MaxPool3D(pool_size=(1, 2, 2))
    ↓
Conv3D(filters=75, kernel_size=3, padding='same')
    + Activation('relu')
    + MaxPool3D(pool_size=(1, 2, 2))
    ↓
TimeDistributed(Flatten())
    ↓
Bidirectional(LSTM(128, return_sequences=True))
    + Dropout(0.5)
    ↓
Bidirectional(LSTM(128, return_sequences=True))
    + Dropout(0.5)
    ↓
Dense(vocab_size + 1, activation='softmax')
    ↓
Output: Prédictions de caractères
```

**Décodage CTC** :
- Utilisation de `tf.keras.backend.ctc_decode` pour convertir les prédictions en texte
- Gestion des alignements temporels
- Décodage greedy pour la prédiction

### 3. Module d'Extraction Audio

**Fichiers** :
- `src/audio_extraction/ReconnaissanceVocal_RealTimeSTT.py` : Transcription en temps réel
- `src/audio_extraction/ReconnaissanceVocal_WhisperAI.py` : Transcription par batch

**Pipeline audio** :
```
Capture audio (microphone/fichier)
    ↓
Prétraitement (normalisation, filtrage)
    ↓
VAD (Voice Activity Detection)
    ↓
Segmentation en segments de parole
    ↓
Transcription Whisper/RealtimeSTT
    ↓
Post-traitement (nettoyage, ponctuation)
    ↓
Texte transcrit
```

**Gestion des pauses** :
- Détection adaptative des pauses selon le contexte
- Gestion des phrases incomplètes (ellipses)
- Segmentation intelligente des phrases

### 4. Interface Utilisateur

**Fichier** : `Interface/View/MainWindoUi.py`

**Composants** :
- **Webcam Widget** : Affichage du flux vidéo en temps réel
- **Lip Tracking Display** : Overlay des détections labiales
- **Transcription Panel** : Affichage de la transcription audio
- **Menu latéral** : Panneau rétractable pour l'historique complet

**Architecture Qt** :
```
QMainWindow
    └── QWidget (central)
        └── QHBoxLayout
            ├── QFrame (menu latéral)
            │   └── QPlainTextEdit (transcription)
            ├── QPushButton (toggle menu)
            └── QVBoxLayout (contenu principal)
                ├── QLabel (webcam)
                ├── QPushButton (toggle webcam)
                ├── QLabel (transcription)
                └── QLabel (statut lip tracking)
```

## Flux de données

### Mode temps réel

1. **Capture vidéo** : Webcam → OpenCV → Frame RGB
2. **Traitement visuel** : Frame → LipTracking → ROI labiale → Modèle → Texte
3. **Capture audio** : Microphone → PyAudio → Buffer audio
4. **Traitement audio** : Buffer → VAD → Whisper → Texte
5. **Affichage** : Texte visuel + Texte audio → Interface Qt

### Mode batch

1. **Chargement vidéo** : Fichier vidéo → OpenCV → Séquences de frames
2. **Prétraitement** : Normalisation, découpage en séquences de 75 frames
3. **Prédiction** : Séquences → Modèle → Prédictions CTC
4. **Décodage** : Prédictions → CTC decode → Texte final

## Gestion de la mémoire

- **GPU** : Configuration automatique de la croissance de mémoire pour TensorFlow
- **CPU** : Fallback automatique si GPU non disponible
- **Buffers** : Gestion des buffers audio/vidéo pour le traitement en temps réel

## Extensibilité

### Points d'extension

1. **Fusion multimodale** : Module à ajouter pour combiner les signaux visuels et audio
2. **Nouveaux modèles** : Architecture modulaire permettant l'ajout de nouveaux modèles
3. **Nouvelles sources** : Support pour fichiers vidéo, streams réseau, etc.
4. **Export** : Formats d'export multiples (texte, sous-titres, etc.)

### Interfaces

- **API REST** : Pour l'intégration avec d'autres systèmes (futur)
- **Plugin system** : Architecture modulaire pour les extensions (futur)

