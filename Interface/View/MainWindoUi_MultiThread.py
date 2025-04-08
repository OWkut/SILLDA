# Désactiver les erreurs liées à Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Masque les messages supplémentaires

import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMainWindow,
    QHBoxLayout,
    QPlainTextEdit,
    QFrame,
    QFileDialog,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, QPropertyAnimation, QEasingCurve, QThread, Signal

# Importer les modèles (LipTracking et LipReading)
from src.lip_tracking.VisualizeLip import LipTracking
from src.lip_reading.LipReadingModel import LipReadingModel


class TranscriptionThread(QThread):
    """Thread pour effectuer la transcription en arrière-plan."""
    transcription_ready = Signal(str)  # Signal pour envoyer la transcription terminée

    def __init__(self, video_path, lip_reading_model):
        super().__init__()
        self.video_path = video_path
        self.lip_reading_model = lip_reading_model

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= 75:
            frames = self.lip_reading_model.load_video_with_liptracking(self.video_path)
            transcription = self.lip_reading_model.predict_text(frames)
        else:
            frame_segments = self.lip_reading_model.load_long_video_with_liptracking(self.video_path)
            transcription = self.lip_reading_model.predict_long_text(frame_segments)
        
        self.transcription_ready.emit(transcription) # Envoyer la transcription terminée

class MainWindowUi(QMainWindow):
    def __init__(self):
        super().__init__()

        # 🔹 Propriétés de la fenêtre
        self.setWindowTitle("SILLDA")
        self.setGeometry(100, 100, 900, 600)
        self.load_styles()

        # 🔹 Widget central contenant la webcam et le menu
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()  
        self.main_layout.setObjectName("main_widget")

        # ========================= MENU LATÉRAL =========================
        self.menu_width = 250  
        self.menu = QFrame(self)
        self.menu.setFixedWidth(0)  
        self.menu.setObjectName("menu")

        # Layout du menu
        self.menu_layout = QVBoxLayout()
        self.menu_layout.setObjectName("menu_layout")
        self.menu.setLayout(self.menu_layout)

        # 🔹 Zone de texte (QPlainTextEdit)
        self.text_output = QPlainTextEdit(self)
        self.text_output.setPlaceholderText("Retranscription totale de l'audio")
        self.text_output.setReadOnly(True)
        self.menu_layout.addWidget(self.text_output)

        # 🔹 Bouton pour masquer le menu
        self.toggle_menu_button = QPushButton("➡", self)
        self.toggle_menu_button.setFixedWidth(50)
        self.toggle_menu_button.clicked.connect(self.toggle_menu)

        # ========================= SECTION WEBCAM =========================
        self.content_layout = QVBoxLayout()

        self.webcam = QLabel(self)
        self.webcam.setFixedSize(640, 480)
        self.webcam.setObjectName("webcam")
        self.content_layout.addWidget(self.webcam)
        self.content_layout.setObjectName("content_layout")

        self.toggle_webcam = QPushButton("Activer la webcam", self)
        self.toggle_webcam.setObjectName("toggle_webcam")
        self.toggle_webcam.clicked.connect(self.handle_toggle_webcam)

        self.restranscription_output = QLabel("RETRANSCRIPTION")
        self.lip_tracking_output = QLabel("Analyse des lèvres : Inactif")  # 🔹 Affichage des résultats du lip tracking
        
        self.content_layout.addWidget(self.toggle_webcam)
        self.content_layout.addWidget(self.restranscription_output)
        self.content_layout.addWidget(self.lip_tracking_output)  # 🔹 Ajout du label au layout

        # 🔹 Ajout des sections au layout principal
        self.main_layout.addWidget(self.menu)
        self.main_layout.addWidget(self.toggle_menu_button)
        self.main_layout.addLayout(self.content_layout)

        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 🔹 Timer pour la mise à jour de la vidéo
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 🔹 Animation pour afficher/réduire le menu
        self.animation = QPropertyAnimation(self.menu, b"minimumWidth")
        self.animation.setDuration(300)  
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # État du menu (fermé par défaut)
        self.menu_open = False

        # ========================= RETRANSCRIPTION D'UNE VIDEO ========================= #
        # Bouton pour charger une vidéo
        self.load_video_button = QPushButton("Retranscrire une vidéo", self)
        self.load_video_button.setObjectName("load_video_button")
        self.load_video_button.clicked.connect(self.load_video)
        self.content_layout.addWidget(self.load_video_button)
        
        # 🔹 Propriétés supplémentaires pour la transcription en temps réel
        self.transcription_thread = None

        # ========================= SECTION LIPTRACKING et LIPREADING ========================= #
        # 🔹 Initialisation du modèle de LipTracking et LipReading
        self.lip_tracker = LipTracking()
        self.lip_reading_model = LipReadingModel()

        # 🔹 Charger le modèle de lipreading
        self.model = self.lip_reading_model.build_model()
        self.frames = []

        # 🔹 Ajouter un label pour afficher les frames normalisées
        self.normalized_frames_output = QLabel("Frames normalisées")
        self.normalized_frames_output.setFixedSize(140, 46)  # Taille attendue par le modèle
        self.content_layout.addWidget(self.normalized_frames_output)

    def load_video(self):
        """Ouvre un gestionnaire de fichiers pour sélectionner une vidéo et la retranscrit."""
        video_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner une vidéo", "", "Vidéo Files (*.mp4 *.mpg *.avi)")
        
        if video_path:
            self.process_video(video_path)

    def process_video(self, video_path):
        """Affiche la vidéo en temps réel et lance la transcription en arrière-plan."""
        # Ouvrir la vidéo avec OpenCV
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.text_output.appendPlainText("❌ ERREUR : Impossible d'ouvrir la vidéo")
            return

        # Démarrer le timer pour afficher la vidéo
        self.timer.timeout.disconnect()  # Déconnecter tout slot précédent
        self.timer.timeout.connect(self.update_frame_video)  # Connecter à update_frame_video
        self.timer.start(30)  # Mettre à jour toutes les 30 ms (~30 fps)

        # Lancer la transcription dans un thread séparé
        self.transcription_thread = TranscriptionThread(video_path, self.lip_reading_model)
        self.transcription_thread.transcription_ready.connect(self.update_transcription)
        self.transcription_thread.start()

    def update_transcription(self, transcription):
        """Met à jour la transcription dans l'interface graphique."""
        self.text_output.setPlainText(transcription)

    def update_frame_video(self):
        """Affiche la vidéo frame par frame avec le rectangle du LipTracking."""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Convertir la frame en RGB pour le traitement
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Appliquer le LipTracking
                processed_frame, lip_coordinates, _ = self.lip_tracker.process_frame(frame_rgb)

                # Dessiner un rectangle autour des lèvres si elles sont détectées
                if lip_coordinates is not None:
                    x, y, w, h = lip_coordinates
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle vert

                # Afficher la frame traitée dans le QLabel
                h, w, ch = processed_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.webcam.setPixmap(QPixmap.fromImage(qt_image))
            else:
                # Arrêter le timer lorsque la vidéo est terminée
                self.timer.stop()
                self.cap.release()
                self.cap = None
    
    def transcribe_video(self, video_path):
        """Utilise le modèle de lip reading pour retranscrire la vidéo."""
        frames = self.lip_reading_model.load_video_with_liptracking(video_path)
        transcription = self.lip_reading_model.predict_text(frames)
        self.text_output.setPlainText(transcription)

    def load_styles(self):
        """Charge le fichier QSS et applique les styles"""
        file_path = os.path.dirname(__file__)
        qss_file_path = os.path.join(file_path, "..", "Ressources", "QSS", "style.qss")
        if os.path.exists(qss_file_path):
            with open(qss_file_path, "r") as f:
                style = f.read()
                self.setStyleSheet(style)
        else:
            print("❌ Fichier 'style.qss' introuvable !")

    def toggle_menu(self):
        """Afficher ou réduire le menu latéral"""
        if self.menu_open:
            new_width = 0
            self.toggle_menu_button.setText("➡")
        else:
            new_width = self.menu_width
            self.toggle_menu_button.setText("⬅")

        self.animation.setStartValue(self.menu.width())
        self.animation.setEndValue(new_width)
        self.animation.start()
        self.menu.setFixedWidth(new_width)  

        self.menu_open = not self.menu_open  

    def handle_toggle_webcam(self):
        """Active ou désactive la webcam"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  
            if not self.cap.isOpened():
                self.text_output.appendPlainText("❌ ERREUR : Impossible d'accéder à la webcam")
                self.cap = None
                return
            # Connecter le timer à update_frame pour la webcam
            self.timer.timeout.disconnect()  # Déconnecter tout slot précédent
            self.timer.timeout.connect(self.update_frame)  # Connecter à update_frame
            self.timer.start(30)  
            self.toggle_webcam.setText("Désactiver la webcam")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.webcam.clear()
            self.lip_tracking_output.setText("Analyse des lèvres : Inactif")  # Réinitialisation
            self.toggle_webcam.setText("Activer la webcam")

    def update_frame(self):
        """Capture l'image de la webcam et applique le lip tracking et la traduction"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 🔹 Appliquer le lip tracking
                processed_frame, lip_coordinates, lip_status = self.lip_tracker.process_frame(frame)

                # 🔹 Mettre à jour l'état de la détection des lèvres (détécté ou non détécté)
                self.lip_tracking_output.setText(f"Analyse des lèvres : {lip_status}")

                if lip_coordinates is not None:
                    x, y, w, h = lip_coordinates
                    mouth_region = frame[y:y+h, x:x+w]  # Extraire la région des lèvres
                    mouth_resized = cv2.resize(mouth_region, (140, 46))  # Redimensionner à la taille attendue par le modèle
                    mouth_resized_grey = tf.image.rgb_to_grayscale(mouth_resized)  # Convertir en niveaux de gris
                    self.frames.append(mouth_resized_grey)
                else:
                    # Si aucune bouche n'est détectée, ajouter une frame vide (noire)
                    self.frames.append(tf.zeros((46, 140, 1), dtype=tf.float32))
                    print("frame vide")

                # 🔹 Convertir l'image traitée pour l'affichage dans QLabel
                h, w, ch = processed_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.webcam.setPixmap(QPixmap.fromImage(qt_image))

                # 🔹 Prédire le texte toutes les 75 frames
                if len(self.frames) >= 75:
                    frames_np = np.array(self.frames[-75:])  # Convertir la liste en tableau numpy
                    normalized_frames = self.lip_reading_model.normalized_frames(frames_np) # Normaliser les frames

                    # # 🔹 Afficher la première frame normalisée
                    # normalized_frame = frames_np[0].numpy()  # Convertir en numpy array
                    # normalized_frame = (normalized_frame * 255).astype(np.uint8)  # Convertir en uint8 pour l'affichage
                    # normalized_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR)  # Convertir en BGR pour QImage

                    # # Convertir en QImage pour l'affichage
                    # h, w, ch = normalized_frame.shape
                    # bytes_per_line = ch * w
                    # qt_image = QImage(normalized_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    # self.normalized_frames_output.setPixmap(QPixmap.fromImage(qt_image))

                    # 🔹 Prédire le texte
                    predicted_text = self.lip_reading_model.predict_text(normalized_frames)
                    self.text_output.appendPlainText(predicted_text)
                    self.frames = []  # Réinitialiser les frames après la prédiction
