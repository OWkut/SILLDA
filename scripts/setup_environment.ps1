# Script d'installation de l'environnement SILLDA pour Windows PowerShell
# Usage: .\scripts\setup_environment.ps1

Write-Host "🚀 Installation de l'environnement SILLDA..." -ForegroundColor Green

# Vérifier si conda est installé
if (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "✅ Conda détecté" -ForegroundColor Green
    Write-Host "📦 Création de l'environnement conda..." -ForegroundColor Yellow
    conda env create -f environnement_SILLDA.yaml
    Write-Host "✅ Environnement conda créé" -ForegroundColor Green
    Write-Host "💡 Activez l'environnement avec: conda activate SILLDA" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  Conda non détecté, utilisation de pip..." -ForegroundColor Yellow
    Write-Host "📦 Installation des dépendances avec pip..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✅ Dépendances installées" -ForegroundColor Green
}

# Créer les dossiers nécessaires
Write-Host "📁 Création des dossiers..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path data\raw, data\processed | Out-Null
New-Item -ItemType Directory -Force -Path models\pretrained, models\trained | Out-Null
New-Item -ItemType Directory -Force -Path experiments\results, experiments\logs | Out-Null
New-Item -ItemType Directory -Force -Path tests, notebooks | Out-Null

# Télécharger les modèles (si script disponible)
if (Test-Path "scripts\download_models.ps1") {
    Write-Host "📥 Téléchargement des modèles..." -ForegroundColor Yellow
    .\scripts\download_models.ps1
}

Write-Host "✅ Installation terminée!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Prochaines étapes:" -ForegroundColor Cyan
Write-Host "   1. Activez l'environnement: conda activate SILLDA (ou votre venv)" -ForegroundColor White
Write-Host "   2. Téléchargez shape_predictor_68_face_landmarks.dat dans data\raw\lip_reading_dlib\" -ForegroundColor White
Write-Host "   3. Placez les modèles pré-entraînés dans models\pretrained\" -ForegroundColor White
Write-Host "   4. Lancez l'application: python main.py" -ForegroundColor White

