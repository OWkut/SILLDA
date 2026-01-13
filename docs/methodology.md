# Méthodologie - SILLDA

## Vue d'ensemble

Ce document décrit en détail la méthodologie utilisée dans le projet SILLDA pour la reconnaissance labiale et la transcription vocale.

## 1. Reconnaissance Labiale

### 1.1 Détection et Suivi des Lèvres

#### Outils utilisés
- **dlib** : Bibliothèque de vision par ordinateur pour la détection faciale
- **68 points faciaux** : Modèle prédéfini pour la détection des points clés du visage
- **20 points labiaux** : Extraction des points spécifiques à la bouche (indices 48-68)

#### Algorithme de détection

```python
1. Détection faciale avec dlib.get_frontal_face_detector()
2. Pour chaque visage détecté :
   a. Prédiction des 68 points avec shape_predictor
   b. Extraction des 20 points labiaux (indices 48-68)
   c. Calcul des coordonnées min/max pour la bounding box
   d. Ajout d'une bordure adaptative (30 pixels par défaut)
   e. Calcul du centre de la bouche
   f. Mise à jour dynamique des dimensions de crop
3. Recadrage de la région labiale
4. Vérification des limites de l'image
```

#### Normalisation

- **Redimensionnement** : 46×140 pixels (hauteur × largeur)
- **Conversion** : RGB → Niveaux de gris
- **Normalisation statistique** : 
  - Calcul de la moyenne et de l'écart-type sur la séquence
  - Normalisation : `(frame - mean) / std`

### 1.2 Modèle de Deep Learning

#### Architecture

Le modèle utilise une architecture hybride Conv3D + LSTM :

**Partie Convolutionnelle 3D** :
- **Conv3D Layer 1** : 128 filtres, kernel 3×3×3, padding 'same'
- **MaxPool3D** : Pool size (1, 2, 2) - réduction spatiale uniquement
- **Conv3D Layer 2** : 256 filtres, kernel 3×3×3
- **MaxPool3D** : Pool size (1, 2, 2)
- **Conv3D Layer 3** : 75 filtres, kernel 3×3×3
- **MaxPool3D** : Pool size (1, 2, 2)

**Partie Temporelle (LSTM)** :
- **TimeDistributed(Flatten)** : Aplatissement des features spatiales
- **Bidirectional LSTM 1** : 128 unités, return_sequences=True
- **Dropout** : 0.5
- **Bidirectional LSTM 2** : 128 unités, return_sequences=True
- **Dropout** : 0.5

**Couche de sortie** :
- **Dense** : vocab_size + 1 (pour le token blank CTC), activation='softmax'

#### Choix de conception

1. **Conv3D** : Permet de capturer les patterns spatio-temporels dans les séquences vidéo
2. **MaxPool3D avec pool_size (1,2,2)** : Réduction spatiale uniquement, préservation de la dimension temporelle
3. **LSTM Bidirectionnel** : Accès au contexte passé et futur pour améliorer la prédiction
4. **CTC (Connectionist Temporal Classification)** : Gestion des alignements temporels sans nécessiter d'annotations frame-by-frame

#### Vocabulaire

39 caractères : `abcdefghijklmnopqrstuvwxyz'?!123456789 ` (espace inclus)

#### Décodage CTC

```python
# Décodage greedy
decoded = tf.keras.backend.ctc_decode(
    yhat, 
    input_length=[75], 
    greedy=True
)[0][0].numpy()

# Conversion en texte
text = tf.strings.reduce_join([
    num_to_char(word) for word in sentence
])
```

### 1.3 Prétraitement des données

#### Format d'entrée
- **Séquence temporelle** : 75 frames
- **Résolution spatiale** : 46×140 pixels
- **Canaux** : 1 (niveaux de gris)
- **Format** : TensorFlow Tensor, shape (75, 46, 140, 1)

#### Normalisation
- Normalisation par séquence (moyenne et écart-type calculés sur toutes les frames)
- Type de données : float32

## 2. Transcription Vocale

### 2.1 Modèles utilisés

#### Whisper (OpenAI)
- **Versions supportées** : tiny, base, small, medium, large
- **Modèle par défaut** : medium
- **Langue** : Anglais (configurable)
- **Avantages** : Haute précision, robuste au bruit

#### RealtimeSTT
- **Utilisation** : Transcription en temps réel
- **Fonctionnalités** :
  - Détection de pause adaptative
  - Gestion des phrases incomplètes
  - Mise à jour en temps réel

### 2.2 Détection de la voix (VAD)

#### Voice Activity Detection
- **Objectif** : Détecter les segments contenant de la parole
- **Méthode** : Filtrage basé sur l'amplitude et les caractéristiques spectrales
- **Seuil** : 0.005 (configurable)

#### Gestion des pauses

Trois types de pauses détectées :

1. **Pause fin de phrase** (`end_of_sentence_detection_pause`) : 0.3s
   - Détectée après un point, point d'exclamation, ou point d'interrogation
   
2. **Pause phrase incomplète** (`mid_sentence_detection_pause`) : 1.5s
   - Détectée lors de la présence d'ellipses (...)
   
3. **Pause inconnue** (`unknown_sentence_detection_pause`) : 0.5s
   - Pause par défaut pour les autres cas

### 2.3 Post-traitement du texte

#### Nettoyage
- Suppression des caractères non imprimables
- Suppression des ellipses en fin de phrase
- Capitalisation de la première lettre

#### Formatage
- Ajout automatique d'espaces entre les phrases
- Gestion de la ponctuation

## 3. Fusion Multimodale (Futur)

### 3.1 Approches envisagées

1. **Fusion précoce** : Concaténation des features avant la prédiction
2. **Fusion tardive** : Combinaison des prédictions des deux modèles
3. **Attention multimodale** : Mécanisme d'attention pour pondérer les deux sources

### 3.2 Avantages attendus

- **Robustesse** : Compensation des erreurs d'un modalité par l'autre
- **Précision** : Amélioration de la transcription dans des conditions difficiles
- **Redondance** : Système de secours si une modalité échoue

## 4. Évaluation

### 4.1 Métriques

#### Word Error Rate (WER)
```
WER = (S + D + I) / N
```
- S : Substitutions
- D : Déletions
- I : Insertions
- N : Nombre total de mots de référence

#### Character Error Rate (CER)
```
CER = (S + D + I) / N
```
- Même formule au niveau des caractères

### 4.2 Datasets de référence

- **LRW (Lip Reading in the Wild)** : Dataset standard pour la lecture labiale
- **LRS2/LRS3** : Datasets BBC avec alignements audio-vidéo
- **GRID Corpus** : Dataset contrôlé pour l'évaluation

## 5. Limitations et défis

### 5.1 Limitations actuelles

1. **Vue frontale requise** : Performance dégradée avec des angles de vue importants
2. **Éclairage** : Sensibilité aux conditions d'éclairage
3. **Langue** : Modèle optimisé pour l'anglais
4. **Latence** : Traitement GPU recommandé pour le temps réel

### 5.2 Défis techniques

1. **Synchronisation audio-vidéo** : Alignement temporel précis nécessaire
2. **Variabilité inter-locuteurs** : Adaptation aux différents styles de parole
3. **Occlusions** : Gestion des cas où la bouche est partiellement cachée
4. **Résolution** : Balance entre qualité et performance

## 6. Améliorations futures

### 6.1 Court terme

- [ ] Support multilingue pour la reconnaissance labiale
- [ ] Optimisation de la latence
- [ ] Amélioration de la robustesse à l'éclairage

### 6.2 Moyen terme

- [ ] Implémentation de la fusion multimodale
- [ ] Fine-tuning sur datasets spécifiques
- [ ] Interface web pour l'accès distant

### 6.3 Long terme

- [ ] Modèles end-to-end optimisés
- [ ] Support pour mobile/embarqué
- [ ] Intégration avec d'autres systèmes (traduction, etc.)

