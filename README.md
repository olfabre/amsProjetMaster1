## L'application est créée en python.  
## Il y a deux modes d'utilisations :   
- utilisation des modèles pour génerer des caractères ASCII  
- création et sauvegarde de modèles  
  
- utilisation des modèles pour génerer des caractères ASCII  
Une interface qui permet de sélectionner le modèle que l'on souhaite utiliser  
Une interface qui permet d'exécuter le modèle avec une affiche des résultats et la possibilité de les obtenir par fichier texte.  
  
  
- création et sauvegarde de modèles  
Une interface qui affiche les hyperparamètres par défaut avec la possibilités de les modifier et donner un nom au modèle et le bouton "créér"  
Une interface qui permet de charger le corpus d'entrainement et de visualiser le processus d'entrainement avec les KPIs.  
Une interface pour la sauvegarde du modèle qui a été entrainé directement sur un dépôt distant github dédié ou arrêter ou modifier la configuration du modèle.  
  
   
  
## Objectifs de l'application  
L'application vise à générer des mots de passe respectant les critères suivants :  
  

### Fonctionnalités principales  
  
#### 1. Utilisation des modèles pour générer des mots de passe ASCII  
Sélection du modèle :  
Interface pour choisir parmi les modèles existants.  
Affiche des informations sur les modèles disponibles (nom, date de création, performance, etc.).  
  
Exécution du modèle :  
Génération de mots de passe selon les paramètres sélectionnés (longueur, diversité).  
Affichage des résultats dans l’interface avec la possibilité de télécharger les mots de passe sous forme de fichier texte.  
  
#### 2. Création et sauvegarde de modèles  
  
Configuration des hyperparamètres :  
- Interface pour configurer ou modifier les hyperparamètres (nombre de couches, taille des couches, taux d’apprentissage, etc.).  
- Option pour donner un nom au modèle et sauvegarder la configuration.  
  
Chargement du corpus d'entraînement :  
- Interface permettant de sélectionner un fichier de données pour l’entraînement.  
- Prévisualisation des données pour validation avant lancement.  
  
Visualisation de l'entraînement :
- Affichage des métriques d'entraînement (ex. : courbe de perte, précision).  
- Mise à jour en temps réel pendant l'entraînement.  
  
Sauvegarde du modèle :  
- Option pour enregistrer le modèle localement ou directement sur un dépôt GitHub.
- Possibilité d'arrêter ou de reprendre l’entraînement.


### Interfaces Utiles  
#### Page d'accueil :  
  
Menu principal avec deux options :  
- Générer des mots de passe.  
- Créer/Sauvegarder un modèle.  
  
#### Interface pour la génération de mots de passe :  
  
- Liste déroulante pour sélectionner le modèle.  
- Paramètres de génération (ex. : longueur, diversité).  
- Zone d’affichage des résultats.  
- Bouton pour télécharger les résultats sous forme de fichier texte.  
  
#### Interface pour la configuration des modèles :  
  
Champs pour les hyperparamètres :  
- Taille des couches, taux d’apprentissage, etc.  
- Zone pour donner un nom au modèle.  
- Bouton "Créer" pour valider la configuration.  
  
#### Interface pour le chargement du corpus d'entraînement :   
- Champ pour sélectionner un fichier d'entraînement.  
- Aperçu du contenu du fichier (premières lignes).  
- Bouton "Lancer l'entraînement".  
  
##### Interface pour l'entraînement du modèle :  
  
- Graphiques affichant les métriques d'entraînement (ex. : perte, précision).  
- Logs de progression (ex. : nombre d'époques, temps écoulé).  
  
#### Boutons pour :  
- Mettre en pause.  
- Sauvegarder temporairement.  
- Arrêter et sauvegarder le modèle final.  
  
#### Interface pour la sauvegarde :  
  
- Option pour sauvegarder le modèle localement ou sur un dépôt GitHub.
- Champs pour les informations de dépôt GitHub (URL, identifiants, etc.).
- Outils techniques recommandés
  
  
### Framework GUI :

Utilisez Tkinter (inclus par défaut dans Python) pour créer les interfaces.
Alternativement, des bibliothèques modernes comme PyQt ou Kivy peuvent offrir des interfaces plus riches.
Bibliothèque pour les modèles :

PyTorch pour créer et entraîner les modèles.
Sauvegarde avec torch.save pour stocker les modèles.
Gestion de données :

Pandas pour manipuler le corpus d’entraînement.
Fichiers de configuration en JSON ou YAML pour sauvegarder les hyperparamètres.
Dépôt GitHub :

Utilisez GitPython pour automatiser la sauvegarde des modèles sur un dépôt GitHub.
