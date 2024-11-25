from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()

        # Mise en page de la page d'entrainement
        layout = QVBoxLayout()

        label = QLabel("Bienvenue sur la page d'Entrainement")
        start_button = QPushButton("Démarrer l'entraînement")

        # Ajouter les widgets au layout
        layout.addWidget(label)
        layout.addWidget(start_button)

        # Configurer le layout pour le widget
        self.setLayout(layout)
