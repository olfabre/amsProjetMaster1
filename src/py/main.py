from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QIcon
from menu_bar import create_menu_bar
from pages.training import TrainingPage
from pages.usage import UsagePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Fenêtre
        self.setWindowTitle("PassWordGenius")
        self.setGeometry(100, 100, 400, 300)
        self.setWindowIcon(QIcon("icone.png"))
        self.show_welcome_message()


         # Menu
        self.setMenuBar(create_menu_bar(self))

    def show_welcome_message(self):
        """
        Affiche un message d'accueil sur la page d'ouverture.
        """
        # Créer un widget central avec un texte
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout()
        welcome_widget.setLayout(welcome_layout)

        # Ajouter un label avec le texte d'accueil
        welcome_label = QLabel("Bienvenue sur PassWordGenius !\n\n"
                               "Choisissez une action dans le menu pour commencer.")
        welcome_label.setStyleSheet("font-size: 16px; text-align: center;")

        # Ajouter le label au layout
        welcome_layout.addWidget(welcome_label)

        # Définir ce widget comme le contenu central
        self.setCentralWidget(welcome_widget)

    def show_training_page(self):
        training_widget = TrainingPage()
        self.setCentralWidget(training_widget)

    def show_usage_page(self):
        usage_widget = UsagePage()
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
