# SILLDA - Système Intelligent de Lecture Labiale et Détection Audio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.5-green.svg)

**Un système avancé de reconnaissance labiale (lip reading) combinant vision par ordinateur et traitement du signal audio pour la transcription en temps réel**

[Documentation](#documentation) • [Installation](#installation) • [Utilisation](#utilisation) • [Méthodologie](#méthodologie) • [Résultats](#résultats)

</div>

---

## Table des matières

- [Résumé exécutif](#résumé-exécutif)
- [Introduction](#introduction)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Méthodologie](#méthodologie)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)
- [Résultats](#résultats)
- [Références](#références)

---

## Résumé exécutif

**SILLDA** (Système Intelligent de Lecture Labiale et Détection Audio) est une application développée dans le cadre d'un projet technologique (Projet I2, 2024-2025) visant à améliorer l'accessibilité à la communication pour les personnes malentendantes. Le système combine la reconnaissance labiale basée sur la vision par ordinateur et la transcription vocale en temps réel pour créer une solution de retranscription multimodale.

### Objectifs principaux

- **Reconnaissance labiale visuelle** : Détection et suivi des mouvements des lèvres à partir de flux vidéo en temps réel via une webcam
- **Transcription audio** : Transcription vocale en temps réel utilisant des modèles de deep learning (Whisper AI)
- **Intégration multimodale** : Combinaison des signaux visuels et audio pour améliorer la robustesse de la transcription
- **Interface utilisateur intuitive** : Application graphique moderne pour l'interaction en temps réel

### Contexte du projet

Ce projet répond à un besoin sociétal fort : améliorer l'accessibilité à la communication pour les personnes malentendantes. La solution vise à être déployée dans divers contextes tels que les conférences, les échanges interpersonnels ou encore les supports multimédias.

---

## Introduction

### Contexte

Dans le cadre d'un projet technologique, notre équipe a été sollicitée pour concevoir une application innovante de retranscription labiale en temps réel. Ce projet répond à un besoin sociétal fort : améliorer l'accessibilité à la communication pour les personnes malentendantes. La solution vise à être déployée dans divers contextes tels que les conférences, les échanges interpersonnels ou encore les supports multimédias.

### Objectif

L'objectif principal est de développer une interface capable de retranscrire la parole en temps réel en combinant plusieurs sources d'information grâce à l'IA : les mouvements des lèvres (via un script de liptracking), l'audio (via un modèle de transcription vocale) et la vidéo (via un modèle de lecture labiale). Ce projet repose sur l'intégration cohérente de ces modules avec une interface graphique intuitive. Il s'inscrit dans une démarche exploratoire mêlant intelligence artificielle, traitement du signal et interaction homme-machine.

### Grandes étapes du projet

**1. Reconnaissance labiale**

L'un des principaux défis techniques de ce projet réside dans l'intégration en temps réel de deux modèles d'intelligence artificielle complémentaires. Le premier, un modèle de suivi des lèvres, capte en direct les mouvements buccaux via une webcam. Il localise précisément la position des lèvres image par image. Le second, un modèle de retranscription labiale, exploite ces données visuelles pour interpréter les mouvements et en déduire les mots correspondants, sans support audio. Cette étape nécessite une synchronisation fluide entre les deux modèles, ainsi qu'une capacité de traitement rapide et précise, afin d'assurer une retranscription fidèle et instantanée.

**2. Reconnaissance audio**

Pour cette partie, nous avons exploré plusieurs solutions de transcription vocale en temps réel, notamment RealTimeSTT, Whisper AI Turbo et Whisper AI. C'est finalement Whisper AI qui s'est révélée la plus performante, avec une bonne réactivité et une précision élevée. Le système capte la voix via un microphone et la convertit automatiquement en texte, permettant ainsi une transcription instantanée des propos. Elle constitue ainsi un soutien essentiel au système global, notamment lorsque la lecture labiale atteint ses limites. Elle joue donc un rôle clé dans la robustesse de l'application, notamment en environnement réel.

---

## Fonctionnalités

### Reconnaissance labiale visuelle

Le système intègre deux modèles complémentaires pour la reconnaissance labiale :

- **Liptracking** : Modèle de suivi des lèvres qui capture en direct les mouvements buccaux via une webcam et localise précisément la position des lèvres image par image
- **Lipreading** : Modèle de retranscription labiale qui exploite les données visuelles pour interpréter les mouvements et en déduire les mots correspondants, sans support audio
- **Synchronisation** : Intégration fluide entre les deux modèles pour assurer une retranscription fidèle et instantanée
- **Traitement vidéo** : Utilisation de dlib pour la détection des points clés du visage et extraction des régions labiales

### Transcription vocale

- **Transcription en temps réel** : Utilisation de Whisper AI (solution retenue après comparaison avec RealTimeSTT et Whisper AI Turbo)
- **Capture audio** : Système qui capte la voix via un microphone et la convertit automatiquement en texte
- **Précision et réactivité** : Bonne réactivité et précision élevée pour une transcription instantanée
- **Soutien au système global** : Rôle clé dans la robustesse de l'application, notamment lorsque la lecture labiale atteint ses limites

### Interface utilisateur

- **Application graphique moderne** : Interface PySide6 avec design personnalisé
- **Visualisation en direct** : Affichage du flux vidéo avec overlay des détections labiales
- **Transcription en temps réel** : Affichage simultané de la transcription audio et visuelle
- **Menu latéral interactif** : Panneau rétractable pour l'affichage des transcriptions complètes

---

## Architecture

### Vue d'ensemble du système

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Utilisateur (PySide6)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐          ┌─────────▼──────────┐
│  Pipeline      │          │  Pipeline           │
│  Visuel        │          │  Audio             │
├────────────────┤          ├────────────────────┤
│ • Détection    │          │ • Capture audio    │
│   faciale      │          │ • VAD (Voice       │
│ • Extraction   │          │   Activity Det.)   │
│   labiale      │          │ • Transcription     │
│ • Prétraitement│          │   Whisper/STT      │
│ • Modèle       │          │ • Post-traitement  │
│   Conv3D+LSTM  │          │   du texte         │
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

### Composants principaux

#### 1. Module de suivi labial (`src/lip_tracking/`)

- **LipTracking** : Classe principale pour la détection et le suivi des lèvres
  - Détection faciale avec dlib
  - Extraction des 20 points labiaux (indices 48-68)
  - Calcul adaptatif de la région d'intérêt (ROI)
  - Normalisation et recadrage

#### 2. Module de reconnaissance labiale (`src/lip_reading/`)

- **Modèle de deep learning** :
  - **Entrée** : Séquences vidéo de 75 frames (46×140 pixels, niveaux de gris)
  - **Architecture** :
    - 3 couches Conv3D (128, 256, 75 filtres)
    - MaxPooling3D pour la réduction dimensionnelle
    - 2 couches LSTM bidirectionnelles (128 unités)
    - Dense layer avec softmax pour la classification de caractères
  - **Vocabulaire** : 39 caractères (a-z, ponctuation, chiffres)
  - **Décodage** : CTC (Connectionist Temporal Classification)

#### 3. Module d'extraction audio (`src/audio_extraction/`)

- **ReconnaissanceVocal_RealTimeSTT** : Transcription en temps réel
  - Modèle Whisper (medium par défaut)
  - Détection adaptative des pauses
  - Gestion des phrases incomplètes
  - Intégration avec l'interface système

- **ReconnaissanceVocal_WhisperAI** : Transcription par batch
  - Traitement de fichiers audio
  - Support VAD (Voice Activity Detection)

#### 4. Interface utilisateur (`Interface/`)

- **MainWindowUi** : Fenêtre principale PySide6
  - Gestion de la webcam
  - Affichage vidéo en temps réel
  - Panneau de transcription
  - Styles personnalisés (QSS)

---

## Installation

### Prérequis

- **Python** : 3.7 ou supérieur
- **Système d'exploitation** : Windows, Linux, ou macOS
- **GPU** : Optionnel mais recommandé pour l'entraînement (CUDA compatible)
- **Webcam** : Pour l'utilisation en temps réel

### Installation avec Conda (Recommandé)

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/SILLDA.git
cd SILLDA

# Créer l'environnement conda
conda env create -f environnement_SILLDA.yaml

# Activer l'environnement
conda activate SILLDA
```

### Installation avec pip

```bash
# Cloner le dépôt
git clone https://github.com/votre-username/SILLDA.git
cd SILLDA

# Installer les dépendances
pip install -r requirements.txt
```

### Téléchargement des modèles pré-entraînés

Les modèles pré-entraînés doivent être placés dans le dossier `models/pretrained/`. 

**Note** : Les modèles ne sont pas inclus dans le dépôt en raison de leur taille. Consultez la section [Configuration](#configuration) pour plus d'informations.

### Téléchargement des fichiers dlib

Le fichier `shape_predictor_68_face_landmarks.dat` doit être téléchargé depuis le site officiel de dlib et placé dans `data/raw/lip_reading_dlib/`.

```bash
# Téléchargement automatique (si script disponible)
python scripts/download_models.py
```

---

## Utilisation

### Lancement de l'application

```bash
# Activer l'environnement conda (si utilisé)
conda activate SILLDA

# Lancer l'interface graphique
python main.py
```

### Utilisation de la transcription vocale en temps réel

```bash
# Lancer la transcription vocale standalone
python src/audio_extraction/ReconnaissanceVocal_RealTimeSTT.py
```

### Utilisation du module de reconnaissance labiale

```python
from src.lip_reading.transcription import model, load_video

# Charger une vidéo
video_path = "data/raw/sample_video.mp4"
frames = load_video(video_path)

# Prédire la transcription
trimmed_frames = frames[:75]
yhat = model.predict(tf.expand_dims(trimmed_frames, axis=0))
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0]
```

### Utilisation du suivi labial

```python
from src.lip_tracking.VisualizeLip import LipTracking
import cv2

# Initialiser le tracker
tracker = LipTracking()

# Traiter une frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
processed_frame, lip_status = tracker.process_frame(frame)

print(f"Statut de détection : {lip_status}")
```

---

## Méthodologie

### Pipeline de traitement visuel

1. **Acquisition vidéo** : Capture de flux vidéo (webcam ou fichier)
2. **Détection faciale** : Utilisation de dlib pour localiser le visage
3. **Extraction des points clés** : Détection des 68 points du visage, focus sur les 20 points labiaux
4. **Recadrage et normalisation** :
   - Calcul de la région d'intérêt (ROI) avec bordure adaptative
   - Redimensionnement à 46×140 pixels
   - Conversion en niveaux de gris
5. **Prétraitement** :
   - Normalisation (moyenne et écart-type)
   - Découpage en séquences de 75 frames
6. **Prédiction** : Passage à travers le modèle Conv3D+LSTM
7. **Décodage CTC** : Conversion des prédictions en texte

### Pipeline de traitement audio

1. **Capture audio** : Enregistrement en temps réel ou depuis fichier
2. **VAD (Voice Activity Detection)** : Détection de la présence de parole
3. **Segmentation** : Découpage en segments basé sur les pauses
4. **Transcription** : Utilisation de Whisper/RealtimeSTT
5. **Post-traitement** :
   - Nettoyage du texte
   - Détection des fins de phrases
   - Gestion des phrases incomplètes

### Modèle de deep learning

#### Architecture du modèle

```
Input: (75, 46, 140, 1)
    ↓
Conv3D(128, kernel=3) + ReLU + MaxPool3D(1,2,2)
    ↓
Conv3D(256, kernel=3) + ReLU + MaxPool3D(1,2,2)
    ↓
Conv3D(75, kernel=3) + ReLU + MaxPool3D(1,2,2)
    ↓
TimeDistributed(Flatten())
    ↓
Bidirectional(LSTM(128)) + Dropout(0.5)
    ↓
Bidirectional(LSTM(128)) + Dropout(0.5)
    ↓
Dense(40, activation='softmax')
    ↓
Output: Prédictions de caractères
```

#### Hyperparamètres

- **Taille de séquence** : 75 frames
- **Résolution d'entrée** : 46×140 pixels
- **Vocabulaire** : 39 caractères
- **Optimiseur** : Adam (legacy)
- **Taux d'abandon** : 0.5
- **Initialisation** : Orthogonal pour LSTM, He normal pour Dense

### Métriques d'évaluation

- **Word Error Rate (WER)** : Taux d'erreur au niveau des mots
- **Character Error Rate (CER)** : Taux d'erreur au niveau des caractères
- **Temps de traitement** : Latence pour le traitement en temps réel
- **Précision de détection** : Taux de détection correcte des lèvres

---

## Structure du projet

```
SILLDA/
├── data/                          # Données du projet
│   ├── raw/                       # Données brutes
│   │   └── lip_reading_dlib/      # Vidéos et fichiers dlib
│   └── processed/                 # Données prétraitées
│
├── models/                        # Modèles de deep learning
│   ├── pretrained/                # Modèles pré-entraînés
│   └── trained/                   # Modèles entraînés localement
│
├── src/                           # Code source principal
│   ├── audio_extraction/          # Modules de transcription vocale
│   │   ├── ReconnaissanceVocal_RealTimeSTT.py
│   │   ├── ReconnaissanceVocal_WhisperAI.py
│   │   └── ReconnaissanceVocal_WhisperAI_TurboV3.py
│   ├── feature_extraction/        # Extraction de caractéristiques
│   ├── lip_reading/               # Module de reconnaissance labiale
│   │   └── transcription.py
│   ├── lip_tracking/              # Suivi et détection labiale
│   │   └── VisualizeLip.py
│   ├── utils/                     # Fonctions utilitaires
│   ├── webcam/                    # Gestion de la webcam
│   │   └── webcam_temps_reel.py
│   └── app.py                     # Application principale
│
├── Interface/                     # Interface graphique
│   ├── View/                      # Vues de l'interface
│   │   └── MainWindoUi.py
│   ├── Ressources/                # Ressources (styles, images)
│   │   └── QSS/
│   │       └── style.qss
│   └── README.md
│
├── docs/                          # Documentation détaillée
│   ├── architecture.md            # Architecture du système
│   ├── methodology.md             # Méthodologie détaillée
│   └── api_reference.md           # Référence API
│
├── config/                        # Fichiers de configuration
│   ├── model_config.yaml          # Configuration des modèles
│   └── training_config.yaml       # Configuration d'entraînement
│
├── scripts/                       # Scripts utilitaires
│   ├── setup_environment.sh       # Script d'installation
│   └── download_models.sh        # Téléchargement des modèles
│
├── experiments/                   # Expérimentations
│   ├── results/                   # Résultats des expériences
│   └── logs/                      # Logs d'entraînement
│
├── tests/                         # Tests unitaires et d'intégration
├── notebooks/                     # Notebooks Jupyter pour l'expérimentation
│
├── main.py                        # Point d'entrée principal
├── setup.py                        # Script d'installation
├── requirements.txt                # Dépendances Python
├── environnement_SILLDA.yaml      # Environnement Conda
├── .gitignore                     # Fichiers ignorés par Git
└── README.md                      # Ce fichier
```

---

## Configuration

### Configuration des modèles

Les paramètres des modèles peuvent être ajustés dans `config/model_config.yaml` :

```yaml
lip_reading:
  sequence_length: 75
  input_height: 46
  input_width: 140
  vocab_size: 39
  
audio:
  model: "medium"  # Options: tiny, base, small, medium, large
  language: "en"
  realtime_processing_pause: 0.02
```

### Configuration de l'interface

Les paramètres de l'interface peuvent être modifiés dans le code source ou via des fichiers de configuration.

### Variables d'environnement

```bash
# Configuration GPU (optionnel)
export CUDA_VISIBLE_DEVICES=0

# Configuration des chemins
export SILLDA_DATA_PATH=/path/to/data
export SILLDA_MODELS_PATH=/path/to/models
```

---

## Résultats

### Bilan

Le projet a permis de valider l'intégration de plusieurs modules d'intelligence artificielle dans une même application. Si la retranscription audio en temps réel est plutôt fluide et précise, la composante labiale reste plus complexe à stabiliser, bien qu'elle soit satisfaisante sur des vidéos préenregistrées. L'utilisation en temps réel soulève encore des défis techniques, notamment au niveau de la synchronisation et de la réactivité des modèles.

### Performances observées

- **Transcription audio** : Fluide et précise en temps réel avec Whisper AI
- **Reconnaissance labiale** : Satisfaisante sur vidéos préenregistrées
- **Temps réel** : Défis techniques liés à la synchronisation et à la réactivité

### Limitations actuelles

- Performance réduite en conditions d'éclairage faible
- Nécessite une vue frontale du visage
- Latence accrue sans GPU pour le traitement vidéo
- Synchronisation entre les modules à améliorer pour le temps réel

### Perspectives

**Objectifs à court terme :**
- Améliorer la synchronisation et la réactivité des modèles en temps réel
- Obtenir une expérience plus fluide et utilisable en situation réelle

**Objectif à moyen terme :**
- Proposer une solution robuste et accessible, capable d'aider les personnes malentendantes dans des échanges spontanés
- Atteindre une qualité proche de celle obtenue sur vidéo préenregistrée

---

## Technologies utilisées

### Bibliothèques principales

- **TensorFlow** : Bibliothèque open-source développée par Google, utilisée pour l'entraînement et l'exécution de modèles d'intelligence artificielle, notamment en traitement de données audio et visuelles.

- **RealTimeSTT & Whisper** : Outils de reconnaissance vocale automatique permettant la transcription en temps réel de la parole, facilitant l'interaction homme-machine.

- **Lipreading & Liptracking** : Deux dépôts Git intégrant des modèles de lecture labiale automatique et de suivi des mouvements des lèvres à partir de vidéos.

- **Jira & GitHub** : Plateformes de gestion de projet et de versioning utilisées pour la coordination de l'équipe, le suivi des tâches et le contrôle du code source.

- **PySide6** : Framework d'interface graphique pour le développement de l'application.

- **OpenCV** : Bibliothèque de traitement d'images et vidéo.

- **dlib** : Bibliothèque de vision par ordinateur pour la détection faciale.

### Ressources et dépôts utilisés

- **RealTimeSTT** : https://github.com/KoljaB/RealtimeSTT
- **Lipreading** : https://github.com/skaty5678/lipread_tensorflow
- **Liptracking** : https://github.com/astorfi/lip-reading-deeplearning/tree/master?tab=readme-ov-file#lip-tracking

### Documentation du projet

- **Affiche de présentation** : [docs/affiche_projet_sillda.pdf](docs/affiche_projet_sillda.pdf) - Présentation complète du projet (Projet I2, 2024-2025)
- **Architecture** : [docs/architecture.md](docs/architecture.md) - Architecture détaillée du système
- **Méthodologie** : [docs/methodology.md](docs/methodology.md) - Méthodologie et approches techniques
- **Référence API** : [docs/api_reference.md](docs/api_reference.md) - Documentation de l'API

---

## Équipe

**Projet I2 - 2024-2025**

- **Floriant RIOTTE**
- **Théo BEN SADOUN**
- **Maël GOAREGUER**
- **Enzo MAGGIO**
- **Léa POTIN**

### Enseignant référent

- **Jesus FRANCO ROBLES** - Enseignant référent et client - 3iL Ingénieurs

---

## Remerciements

- **Jesus FRANCO ROBLES** pour son encadrement et son soutien tout au long du projet
- Équipe d'OpenAI pour Whisper
- Développeurs des projets open source utilisés (RealTimeSTT, Lipreading, Liptracking)
- Communauté open source pour les outils et bibliothèques utilisés

---

<div align="center">

**Projet I2 - 2024-2025 - 3iL Ingénieurs**

Application de retranscription labiale en temps réel pour l'accessibilité

</div>
