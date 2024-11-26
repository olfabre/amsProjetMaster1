from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()

        # Mise en page principale
        layout = QVBoxLayout()

        # Ajouter un titre
        title = QLabel("Formulaire d'Entrainement")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # Créer le formulaire
        form_layout = QFormLayout()

        # Champ de texte : Nom du modèle
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Entrez le nombre d'epoch souhaité")
        form_layout.addRow("Nombre d'epoch :", self.model_name_input)

        # Champ de texte : Nom du modèle
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Entrez le nombre de neurone par couche voulu")
        form_layout.addRow("Nombre de neurone par couche :", self.model_name_input)

        # Champ de texte : Nom du modèle
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Entrez le nombre de couche souhaité")
        form_layout.addRow("Nombre de couche :", self.model_name_input)

        # Champ de texte : Nom du modèle
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Entrez le taux d'apprentissage désiré")
        form_layout.addRow("Taux d'apprentissage :", self.model_name_input)

        # Ajouter le formulaire au layout principal
        layout.addLayout(form_layout)

        # Bouton pour valider le formulaire
        submit_button = QPushButton("Lancer l'entraînement")
        submit_button.clicked.connect(self.handle_submit)
        layout.addWidget(submit_button)

        # Configurer le layout principal
        self.setLayout(layout)

    def handle_submit(self):
        """
        Méthode appelée lorsque l'utilisateur clique sur le bouton de soumission.
        Elle récupère les données du formulaire et affiche un message.
        """
        model_name = self.model_name_input.text()
        iterations = self.iterations_input.text()

        if not model_name or not iterations:
            QMessageBox.warning(self, "Erreur", "Veuillez remplir tous les champs.")
            return

        try:
            iterations = int(iterations)
            QMessageBox.information(
                self,
                "Entrainement lancé",
                f"Le modèle '{model_name}' est en cours d'entrainement pour {iterations} itérations.",
            )
        except ValueError:
            QMessageBox.critical(self, "Erreur", "Le nombre d'itérations doit être un entier valide.")
