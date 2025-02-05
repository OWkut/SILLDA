import sys
import cv2
import os
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMainWindow,
    QHBoxLayout,
    QPlainTextEdit,
    QFrame,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, QPropertyAnimation, QEasingCurve
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Classes"))
)

from WebcamHandler import WebcamHandler


class MainWindowUi(QMainWindow):
    def __init__(self):
        super().__init__()

        # 🔹 Propriétés de la fenêtre
        self.setWindowTitle("SILLDA")
        self.setGeometry(100, 100, 900, 600)
        self.load_styles()

        # 🔹 Widget central contenant la webcam et le menu
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()  # Disposition horizontale
        self.main_layout.setObjectName("main_widget")

        # ========================= MENU LATÉRAL =========================
        self.menu_width = 250  # Largeur du menu ouvert
        self.menu = QFrame(self)
        self.menu.setFixedWidth(0)  # Caché au démarrage
        self.menu.setObjectName("menu")

        # Layout du menu
        self.menu_layout = QVBoxLayout()
        self.menu_layout.setObjectName("menu_layout")
        self.menu.setLayout(self.menu_layout)

        # 🔹 Zone de texte (QPlainTextEdit)
        self.text_output = QPlainTextEdit(self)
        self.text_output.setPlaceholderText("Retranscription totale de l'audio")
        self.text_output.setReadOnly(True)  # Empêche l'utilisateur d'éditer
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

        # 🔹 Label pour retranscription
        self.restranscription_output = QLabel("RETRANSCRIPTION")
        self.content_layout.addWidget(self.toggle_webcam)
        self.content_layout.addWidget(self.restranscription_output)
        self.btn_fps_graph = QPushButton("Afficher FPS", self)
        self.content_layout.addWidget(self.btn_fps_graph)

        # 🔹 Ajout des sections au layout principal
        self.main_layout.addWidget(self.menu)
        self.main_layout.addWidget(self.toggle_menu_button)
        self.main_layout.addLayout(self.content_layout)

        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # 🔹 Animation pour afficher/réduire le menu
        self.animation = QPropertyAnimation(self.menu, b"minimumWidth")
        self.animation.setDuration(300)  # Durée de l'animation (ms)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # État du menu (fermé par défaut)
        self.menu_open = False

        # ✅ Initialisation de la gestion de la webcam via WebcamHandler
        self.webcam_handler = WebcamHandler(
            self.webcam, self.toggle_webcam, self.text_output
        )
        self.btn_fps_graph.clicked.connect(self.webcam_handler.open_fps_window)

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
        self.menu.setFixedWidth(new_width)  # Mise à jour immédiate

        self.menu_open = not self.menu_open  # Inverser l'état du menu
