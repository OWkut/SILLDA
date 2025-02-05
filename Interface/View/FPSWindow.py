import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt, Signal


class FPSWindow(QMainWindow):
    fps_changed = Signal(int)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Courbe des FPS")
        self.setGeometry(300, 200, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.choose_fps = QSlider(self)
        self.choose_fps.setOrientation(Qt.Orientation.Horizontal)
        self.choose_fps.setRange(1, 30)
        self.choose_fps.setValue(30)

        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.choose_fps)

        self.fps_data = [0] * 100
        self.times = list(range(-100, 0))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(500)

        self.choose_fps.valueChanged.connect(self.emit_fps_change)

    def update_plot(self):
        """Met à jour le graphique des FPS"""
        self.ax.clear()
        self.ax.plot(self.times, self.fps_data, label="FPS", color="blue")
        self.ax.set_ylim(0, 60)  # Fixe l’échelle entre 0 et 60 FPS
        self.ax.set_xlabel("Temps (itérations)")
        self.ax.set_ylabel("FPS")
        self.ax.set_title("Évolution des FPS en temps réel")
        self.ax.legend()
        self.canvas.draw()

    def add_fps_value(self, fps_value):
        """Ajoute une nouvelle valeur FPS et met à jour le graphique"""
        self.fps_data.append(fps_value)
        self.fps_data.pop(0)  # Supprime l’ancienne valeur
        self.update_plot()

    def emit_fps_change(self):
        """Émet la nouvelle valeur FPS"""
        self.fps_changed.emit(self.choose_fps.value())
