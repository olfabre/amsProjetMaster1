from PyQt6.QtWidgets import QMenuBar
from PyQt6.QtGui import QAction


def create_menu_bar(main_window):
    """
    Crée et configure une barre de menu pour l'application.

    Args:
        main_window (QMainWindow): L'objet QMainWindow qui contiendra la barre de menu.

    Returns:
        QMenuBar: La barre de menu configurée.
    """
    menu_bar = QMenuBar(main_window)

    # Menu "Fichier"
    file_menu = menu_bar.addMenu("Fonction")

    # Actions du menu "Fichier"
    train_model = QAction("Entrainement", main_window)
    use_model = QAction("Utiliser", main_window)
    exit_action = QAction("Quitter", main_window)

    # Connecter les action au méthode
    train_model.triggered.connect(main_window.show_training_page)
    use_model.triggered.connect(main_window.show_usage_page)
    exit_action.triggered.connect(main_window.close)

    # Ajouter les actions au menu
    file_menu.addAction(train_model)
    file_menu.addAction(use_model)
    file_menu.addSeparator()
    file_menu.addAction(exit_action)

    # Menu "Édition"
    edit_menu = menu_bar.addMenu("Édition")
    edit_action = QAction("Copier", main_window)
    edit_menu.addAction(edit_action)

    return menu_bar
