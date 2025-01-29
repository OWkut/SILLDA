# 🎨 Interface Graphique - Projet de Visualisation d'Images

Ce projet fournit une interface graphique pour afficher et naviguer entre les matrices d'images générées par l'objet `ImageProcessor`.  
L'interface est construite avec **PySide6**, et utilise **Matplotlib** pour afficher les images et **OpenCV** pour la gestion des images.

---

## 📌 **Dépendances**
Avant d'exécuter l'interface, assure-toi d'avoir installé les bibliothèques suivantes :

📌 **Dépendances Python**
- `PySide6` → Interface graphique Qt pour Python
- `Matplotlib` → Affichage d'images (`imshow()`) avec support pour les annotations
- `OpenCV` (`opencv-python`) → Traitement d'image et compatibilité avec NumPy

📌 **Installation des dépendances**
Si tu n'as pas encore installé les bibliothèques, exécute la commande suivante :
```sh
pip install pyside6 matplotlib opencv-python
```

---

## 🎯 **Structure du projet**
```
📁 Projet/
│── 📁 Ressources/     # Contient les fichiers principaux du projet
│    ├──
│
│── 📁 View/           # Contient les fichiers d'affichage
│    ├── interface.py  # Interface graphique avec PySide6 et Matplotlib
│
│── main.py            # Point d'entrée principal pour l'interface
└── README.md          # Documentation du projet
```

---

## 🚀 **Utilisation**
### **1️⃣ Lancer l'interface**
Après avoir installé les dépendances, exécute la commande suivante pour ouvrir l'interface :

```sh
python main.py
```


---

## 🖼️ **Fonctionnalités**
✅ **Affichage d'images sous forme de matrices** avec `matplotlib.pyplot.imshow()`  
✅ **Navigation entre les images** avec des boutons `Suivant` / `Précédent`  
✅ **Ajout de rectangles d'annotations** pour mettre en évidence des zones d'intérêt  
✅ **Intégration fluide avec `ImageProcessor`** pour récupérer les matrices  

---

## 📞 **Support**
Si tu rencontres un problème, n'hésite pas à créer une **issue** sur le dépôt GitHub ou à contacter le responsable du projet. 🚀
