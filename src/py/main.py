from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QIcon
from menu_bar import create_menu_bar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Fenêtre
        self.setWindowTitle("PassWordGenius")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowIcon(QIcon("icone.png"))
        self.show_training_page()

         # Menu
        self.setMenuBar(create_menu_bar(self))

    def show_training_page(self):
        """Affiche la page d'entrainement."""
        training_widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel("Bienvenue sur la page d'Entrainement")
        layout.addWidget(label)
        training_widget.setLayout(layout)

        # Définir le widget central
        self.setCentralWidget(training_widget)

    def show_usage_page(self):
        """Affiche la page d'utilisation."""
        usage_widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel("Bienvenue sur la page d'Utilisation")
        layout.addWidget(label)
        usage_widget.setLayout(layout)

        # Définir le widget central
        self.setCentralWidget(usage_widget)

    def center_button(self):
        """Centre le bouton dans la fenêtre."""
        button_width = self.button.sizeHint().width()
        button_height = self.button.sizeHint().height()
        window_width = self.width()
        window_height = self.height()

        x = (window_width - button_width) // 2
        y = (window_height - button_height) // 2

        self.button.setGeometry(x, y, button_width, button_height)

    def on_button_click(self):
        self.button.setText("Merci d'avoir cliqué !")
        self.button.setEnabled(False)  # Désactiver le bouton après le clic
        self.button.adjustSize()
        self.center_button()


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
