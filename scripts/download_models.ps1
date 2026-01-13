# Script de téléchargement des modèles pour Windows PowerShell
# Usage: .\scripts\download_models.ps1

Write-Host "📥 Téléchargement des modèles et fichiers nécessaires..." -ForegroundColor Yellow

# Créer les dossiers si nécessaire
New-Item -ItemType Directory -Force -Path data\raw\lip_reading_dlib | Out-Null
New-Item -ItemType Directory -Force -Path models\pretrained | Out-Null

# Télécharger shape_predictor_68_face_landmarks.dat
$predictorPath = "data\raw\lip_reading_dlib\shape_predictor_68_face_landmarks.dat"
if (-not (Test-Path $predictorPath)) {
    Write-Host "📥 Téléchargement de shape_predictor_68_face_landmarks.dat..." -ForegroundColor Yellow
    $url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    $output = "$predictorPath.bz2"
    
    try {
        Invoke-WebRequest -Uri $url -OutFile $output
        Write-Host "📦 Décompression..." -ForegroundColor Yellow
        # Note: Nécessite 7-Zip ou autre outil de décompression
        # Expand-Archive ne supporte pas .bz2, utiliser 7z ou autre
        Write-Host "⚠️  Décompression manuelle requise pour .bz2" -ForegroundColor Yellow
        Write-Host "   Utilisez 7-Zip ou téléchargez depuis: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Erreur lors du téléchargement: $_" -ForegroundColor Red
    }
} else {
    Write-Host "✅ shape_predictor_68_face_landmarks.dat existe déjà" -ForegroundColor Green
}

Write-Host ""
Write-Host "⚠️  Note: Les modèles pré-entraînés (checkpoint) doivent être téléchargés manuellement" -ForegroundColor Yellow
Write-Host "   et placés dans models\pretrained\" -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ Téléchargement terminé!" -ForegroundColor Green

