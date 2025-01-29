import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMainWindow,
    QHBoxLayout,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer


class MainWindowUi(QMainWindow):
    def __init__(self):
        super().__init__()

        # Propriété de la fenetre
        self.setWindowTitle("SILLDA")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget()
        self.layout = QVBoxLayout()

        self.webcam = QLabel(self)
        self.webcam.setFixedSize(640, 480)
        self.layout.addWidget(self.webcam)

        self.toggle_webcam = QPushButton("Activer la webcam", self)
        self.toggle_webcam.clicked.connect(self.handle_toggle_webcam)
        self.layout.addWidget(self.toggle_webcam)

        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def handle_toggle_webcam(self):
        """
        Active ou désactive la webcam
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # 0 c'est la webcam
            if not self.cap.isOpened():
                print("ERREUR")
                self.cap = None
                return
            self.timer.start(30)  # 30ms de couldown
            self.toggle_webcam.setText("Désactiver la webcam")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.webcam.clear()
            self.toggle_webcam.setText("Activer la webcam")

    def update_frame(self):
        """
        Capture l'image de la webcam et l'affiche dans le label
        """
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.webcam.setPixmap(QPixmap.fromImage(qt_image))
