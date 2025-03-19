import sys
from PySide6.QtWidgets import QApplication
from Interface.View.MainWindoUi_MultiThread import MainWindowUi

app = QApplication(sys.argv)
window = MainWindowUi()
window.show()
app.exec()
