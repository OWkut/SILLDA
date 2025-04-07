# 🎥 Flask Video Streaming App

Une application Flask modulaire pour :
- Diffuser la webcam ou une vidéo uploadée avec OpenCV
- Afficher les FPS (actuel / min / max / moyen)
- Mettre en pause ou reprendre le flux en temps réel
- Système de menu dynamique et modulaire en HTML/CSS/JS

---

## 📁 Arborescence du projet

```
WEB_VERSION/
├── app.py                      # Point d'entrée principal Flask
├── uploads/                   # Vidéos uploadées par l'utilisateur
├── templates/
│   └── index.html             # Interface utilisateur principale
├── static/
│   ├── style.css              # Feuilles de style CSS
│   └── script.js              # Logique frontend JS
├── core/                      # Logique métier (modulaire)
│   ├── __init__.py
│   ├── fps_monitor.py         # Calculs de FPS en temps réel
│   ├── sources.py             # Classes WebcamStream et FileVideoStream
│   └── stream_manager.py      # Contrôleur principal du flux (play/pause, FPS)
└── tests/                     # Tests unitaires avec pytest
    ├── test_fps_monitor.py
    └── test_stream_manager.py
```

---

## ⚙️ Installation & lancement

### 1. Installer les dépendances
```bash
pip install flask opencv-python pytest
```

### 2. Lancer le serveur Flask
```bash
python app.py
```

Accéder à l'application sur [http://localhost:5000](http://localhost:5000)

---

## ✅ Fonctionnalités principales

- 📷 Diffusion webcam avec OpenCV
- 📂 Lecture de vidéos uploadées
- ⏸ Play / Pause du flux (côté serveur)
- 📊 Monitoring FPS dynamique (JSON + JS)
- 🎨 Interface modulaire avec onglets/menu
- 🧪 Tests unitaires pour les modules critiques

---
