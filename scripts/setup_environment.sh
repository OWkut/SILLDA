#!/bin/bash
# Script d'installation de l'environnement SILLDA
# Usage: ./scripts/setup_environment.sh

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation de l'environnement SILLDA..."

# Vérifier si conda est installé
if command -v conda &> /dev/null; then
    echo "✅ Conda détecté"
    echo "📦 Création de l'environnement conda..."
    conda env create -f environnement_SILLDA.yaml
    echo "✅ Environnement conda créé"
    echo "💡 Activez l'environnement avec: conda activate SILLDA"
else
    echo "⚠️  Conda non détecté, utilisation de pip..."
    echo "📦 Installation des dépendances avec pip..."
    pip install -r requirements.txt
    echo "✅ Dépendances installées"
fi

# Créer les dossiers nécessaires
echo "📁 Création des dossiers..."
mkdir -p data/raw data/processed
mkdir -p models/pretrained models/trained
mkdir -p experiments/results experiments/logs
mkdir -p tests notebooks

# Télécharger les modèles (si script disponible)
if [ -f "scripts/download_models.sh" ]; then
    echo "📥 Téléchargement des modèles..."
    bash scripts/download_models.sh
fi

echo "✅ Installation terminée!"
echo ""
echo "📝 Prochaines étapes:"
echo "   1. Activez l'environnement: conda activate SILLDA (ou votre venv)"
echo "   2. Téléchargez shape_predictor_68_face_landmarks.dat dans data/raw/lip_reading_dlib/"
echo "   3. Placez les modèles pré-entraînés dans models/pretrained/"
echo "   4. Lancez l'application: python main.py"

