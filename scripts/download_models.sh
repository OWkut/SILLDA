#!/bin/bash
# Script de téléchargement des modèles et fichiers nécessaires
# Usage: ./scripts/download_models.sh

set -e

echo "📥 Téléchargement des modèles et fichiers nécessaires..."

# Créer les dossiers si nécessaire
mkdir -p data/raw/lip_reading_dlib
mkdir -p models/pretrained

# Télécharger shape_predictor_68_face_landmarks.dat
if [ ! -f "data/raw/lip_reading_dlib/shape_predictor_68_face_landmarks.dat" ]; then
    echo "📥 Téléchargement de shape_predictor_68_face_landmarks.dat..."
    wget -O data/raw/lip_reading_dlib/shape_predictor_68_face_landmarks.dat \
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    echo "📦 Décompression..."
    bunzip2 data/raw/lip_reading_dlib/shape_predictor_68_face_landmarks.dat.bz2
    echo "✅ shape_predictor téléchargé"
else
    echo "✅ shape_predictor_68_face_landmarks.dat existe déjà"
fi

# Note: Les modèles pré-entraînés doivent être téléchargés manuellement
# ou depuis un dépôt privé selon votre configuration
echo ""
echo "⚠️  Note: Les modèles pré-entraînés (checkpoint) doivent être téléchargés manuellement"
echo "   et placés dans models/pretrained/"
echo ""
echo "✅ Téléchargement terminé!"

