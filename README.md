# SILLDA

# Lip Reading App  

Une application de reconnaissance labiale utilisant l'intelligence artificielle.  

## Structure du projet  

Voici l'organisation des fichiers et dossiers du projet :  

```plaintext
├── data/
│   ├── raw/                  # Données brutes (vidéos, images)
│   ├── processed/            # Données prétraitées
├── models/
│   ├── pretrained/           # Modèles pré-entraînés
│   └── trained/              # Modèles que vous entraînez
├── src/
│   ├── lip_tracking/         # Code pour le suivi des lèvres
│   ├── feature_extraction/   # Code pour l'extraction des caractéristiques
│   ├── lip_reading/          # Code pour la reconnaissance labiale
│   ├── utils/                # Fonctions utilitaires (prétraitement, post-traitement)
│   ├── webcam/               # Code pour l'utilisation de la webcam en temps reel
│   └── app.py                # Script principal pour l'application
├── tests/                    # Tests unitaires et d'intégration
├── notebooks/                # Notebooks Jupyter pour l'expérimentation
├── requirements.txt          # Liste des dépendances Python
├── README.md                 # Documentation du projet
├── setup.py                  # Script d'installation (si nécessaire)
└── .gitignore                # Fichiers et dossiers à ignorer par Git
