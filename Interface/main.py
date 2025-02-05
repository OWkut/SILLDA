import sys
from PySide6.QtWidgets import QApplication
from View.MainWindoUi import MainWindowUi

app = QApplication(sys.argv)
window = MainWindowUi()
window.show()
app.exec()
