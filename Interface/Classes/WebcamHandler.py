import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "View")))
from FPSWindow import FPSWindow


class WebcamHandler:
    def __init__(self, label_display, toggle_button, text_output):
        """Gestionnaire de webcam pour PyQt6."""
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.label_display = label_display
        self.toggle_button = toggle_button
        self.text_output = text_output

        self.toggle_button.clicked.connect(self.handle_toggle_webcam)

        self.filter_mode = 0
        self.last_time = time.time()
        self.fps_window = None
        self.fps_interval = 30  # Par défaut : 30 FPS

    # ========================
    # 🎥 Gestion de la webcam
    # ========================

    def handle_toggle_webcam(self):
        """Active ou désactive la webcam."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            print(f"FPS Max Webcam : {self.cap.get(cv2.CAP_PROP_FPS)}")
            if not self.cap.isOpened():
                self.text_output.appendPlainText(
                    "❌ ERREUR : Impossible d'accéder à la webcam"
                )
                self.cap = None
                return
            print(f"✅ Webcam activée à {1000 // self.fps_interval} FPS")
            self.timer.start(self.fps_interval)
            self.toggle_button.setText("Désactiver la webcam")
        else:
            self.stop_webcam()

    def stop_webcam(self):
        """Arrête la webcam proprement."""
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.label_display.clear()
            self.toggle_button.setText("Activer la webcam")
            print("🛑 Webcam désactivée")

    # ========================
    # 🖼️ Capture et affichage
    # ========================

    def update_frame(self):
        """Capture l’image de la webcam, applique un filtre et l’affiche."""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = self.apply_filter(frame)
                self.display_frame(frame)
                self.get_fps()

    def display_frame(self, frame):
        """Affiche l’image dans le QLabel après conversion."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888,
        )
        self.label_display.setPixmap(QPixmap.fromImage(qt_image))

    # ========================
    # 🎨 Gestion des filtres
    # ========================

    def apply_filter(self, frame):
        """Applique un filtre spécifique à l’image."""
        if self.filter_mode == 1:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Noir & Blanc
        elif self.filter_mode == 2:
            return cv2.bitwise_not(frame)  # Négatif
        elif self.filter_mode == 3:
            return self.apply_sepia(frame)  # Sépia
        else:
            return frame  # Aucun filtre

    def apply_sepia(self, frame):
        """Applique un effet sépia sur l’image."""
        sepia_matrix = cv2.transform(
            frame,
            cv2.mat(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            ),
        )
        return cv2.addWeighted(sepia_matrix, 1.5, frame, -0.5, 0)

    def change_filter(self, mode):
        """Change le mode du filtre (0 = normal, 1 = N&B, 2 = négatif, 3 = sépia)."""
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"❌ Erreur : Le mode {mode} n'est pas valide.")

        self.filter_mode = mode
        self.text_output.appendPlainText(f"🎨 Filtre appliqué : {self.filter_mode}")

    # ========================
    # 📊 Gestion des FPS
    # ========================

    def get_fps(self):
        """Calcule et met à jour le FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.text_output.appendPlainText(f"⏳ FPS actuel : {fps:.2f}")

        if self.fps_window:
            self.fps_window.add_fps_value(fps)

    def open_fps_window(self):
        """Ouvre la fenêtre FPS et connecte le slider."""
        if self.fps_window is None:
            self.fps_window = FPSWindow()
            self.fps_window.fps_changed.connect(self.set_fps)  # Connexion au slider FPS
            self.fps_window.show()
        else:
            self.fps_window.showNormal()

    def set_fps(self, fps):
        """Modifie la fréquence de rafraîchissement de la webcam."""
        fps = max(1, min(fps, 100))
        self.fps_interval = int(1000 / fps)
        self.timer.setInterval(self.fps_interval)
        if self.cap is not None:
            self.timer.setInterval(self.fps_interval)
        self.text_output.appendPlainText(f"🎛️ FPS réglé à : {fps}")

    def close(self):
        """Libère la webcam proprement."""
        self.stop_webcam()
