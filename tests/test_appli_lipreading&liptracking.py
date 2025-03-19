# Désactiver les erreurs liées à Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Masque les messages supplémentaires

import cv2

# Permet de charger des modules depuis nimporte quel endroit du projet
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Récupérer le chemin absolu du dossier racine du projet
sys.path.append(project_root) # Ajouter ce chemin au sys.path

# Importer les modèles (LipTracking et LipReading)
from src.lip_tracking.VisualizeLip import LipTracking
from src.lip_reading.LipReadingModel import LipReadingModel

# Pour l'interface graphique
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QPlainTextEdit, QHBoxLayout, QFileDialog
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap


# ==================== Classe pour l'interface graphique ==================== #
class MainWindowUi(QMainWindow):
    def __init__(self):
        super().__init__()

        # 🔹 Propriétés de la fenêtre
        self.setWindowTitle("SILLDA - Transcription Labiale")
        self.setGeometry(100, 100, 900, 600)

        # 🔹 Widgets
        self.video_label = QLabel("Aucune vidéo chargée")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        self.transcription_text = QPlainTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setPlaceholderText("La transcription apparaîtra ici...")

        self.load_button = QPushButton("Charger une vidéo")
        self.load_button.clicked.connect(self.load_video)

        self.transcribe_button = QPushButton("Transcrire la vidéo")
        self.transcribe_button.clicked.connect(self.transcribe_video)
        self.transcribe_button.setEnabled(False)  # Désactivé jusqu'à ce qu'une vidéo soit chargée

        # 🔹 Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.transcription_text)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.transcribe_button)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 🔹 Propriétés supplémentaires
        self.video_path = None
        self.lip_reading_model = LipReadingModel()
        self.cap = None  # Pour stocker l'objet VideoCapture
        self.timer = QTimer()  # Timer pour lire la vidéo frame par frame
        self.timer.timeout.connect(self.update_frame)  # Connecter le timer à la méthode update_frame
        self.lip_tracker = LipTracking()  # Initialiser LipTracking

    def transcribe_video(self):
        if not self.video_path:
            return

        # Arrêter l'affichage de la vidéo
        self.timer.stop()
        if self.cap:
            self.cap.release()

        # Charger et prétraiter la vidéo
        frames = self.lip_reading_model.load_video_with_liptracking(self.video_path)

        # Prédire le texte
        transcription = self.lip_reading_model.predict_text(frames)

        # Afficher la transcription
        self.transcription_text.setPlainText(transcription)

    def load_video(self):
        # Ouvrir une boîte de dialogue pour sélectionner une vidéo
        self.video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Charger une vidéo",
            "",
            "Vidéo Files (*.mp4 *.avi *.mpg *.mpeg);;Tous les fichiers (*)"  # Ajouter .mpg et .mpeg
        )
        if self.video_path:
            self.video_label.setText(f"Vidéo chargée : {self.video_path}")
            self.transcribe_button.setEnabled(True)  # Activer le bouton de transcription

            # Ouvrir la vidéo avec OpenCV
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print("Erreur : Impossible d'ouvrir la vidéo.")
                return

            # Démarrer le timer pour lire la vidéo
            self.timer.start(30)  # Mettre à jour toutes les 30 ms (~30 fps)

    def update_frame(self):
        # Lire une frame de la vidéo
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()  # Arrêter le timer si la vidéo est terminée
            self.cap.release()
            return

        # Traiter la frame avec LipTracking
        processed_frame, lip_coordinates, _ = self.lip_tracker.process_frame(frame)

        # Dessiner un rectangle autour des lèvres si elles sont détectées
        if lip_coordinates is not None:
            x, y, w, h = lip_coordinates
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle vert

        # Convertir la frame de BGR (OpenCV) à RGB (Qt)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Convertir la frame en format QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Afficher la frame dans le QLabel
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        # Arrêter le timer et libérer la vidéo lors de la fermeture de l'application
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ==================== Main ==================== #
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindowUi()
    window.show()
    app.exec()