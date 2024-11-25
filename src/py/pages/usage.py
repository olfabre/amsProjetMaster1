from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class UsagePage(QWidget):
    def __init__(self):
        super().__init__()

        # Mise en page de la page d'utilisation
        layout = QVBoxLayout()

        label = QLabel("Bienvenue sur la page d'Utilisation")
        use_button = QPushButton("Commencer à utiliser")

        # Ajouter les widgets au layout
        layout.addWidget(label)
        layout.addWidget(use_button)

        # Configurer le layout pour le widget
        self.setLayout(layout)
