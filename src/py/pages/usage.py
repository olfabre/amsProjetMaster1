from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox


class UsagePage(QWidget):
    def __init__(self):
        super().__init__()

        # Mise en page de la page d'utilisation
        layout = QVBoxLayout()

        # Ajouter un titre
        title = QLabel("Formulaire d'Entrainement")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        # Créer le formulaire
        form_layout = QFormLayout()

        # Liste déroulante : Type de modèle
        self.model_type_dropdown1 = QComboBox()
        self.model_type_dropdown1.addItems(["aucun","mot de passe", "nom", "texte"])  # Ajouter des options
        form_layout.addRow("Type de modèle :", self.model_type_dropdown1)

        # Liste déroulante : modèle
        self.model_type_dropdown2 = QComboBox()
        self.model_type_dropdown2.addItems(["mdl1", "mdl2", "mdl3", "mdl4"])  # Ajouter des options
        form_layout.addRow("Modèle :", self.model_type_dropdown2)

        # Ajouter le formulaire au layout principal
        layout.addLayout(form_layout)

        # Bouton pour valider le formulaire
        submit_button = QPushButton("Générer")
        submit_button.clicked.connect(self.handle_submit)
        layout.addWidget(submit_button)

        # Affichage du contenue généré
        self.result_label = QLabel("")  # Le texte sera vide au début
        self.result_label.setStyleSheet("font-size: 14px; color: green;")  # Style du texte
        layout.addWidget(self.result_label)  # Ajouter le label en dessous du bouton

        # Configurer le layout pour le widget
        self.setLayout(layout)

    def handle_submit(self):
        """
        Méthode appelée lorsque l'utilisateur clique sur le bouton de soumission.
        Elle récupère les données du formulaire et affiche un message sous le bouton.
        """
        # Récupérer les valeurs sélectionnées
        selected_model_type = self.model_type_dropdown1.currentText()
        selected_model = self.model_type_dropdown2.currentText()

        # Vérifier que l'utilisateur a sélectionné un type de modèle
        if selected_model_type == "aucun":
            QMessageBox.warning(self, "Erreur", "Veuillez sélectionner un type de modèle.")
            return

        # Mettre à jour le texte du label avec les informations choisies
        self.result_label.setText(
            f"Vous avez sélectionné le type '{selected_model_type}' avec le modèle '{selected_model}'."
        )