# Rapport de progression 

## Notre projet Application: PassWordGenius



### 1. Introduction

#### 1.1. Notre équipe:

Notre équipe se compose de:

- Olivier Fabre, uapv2014042

- François Demogue, uapv2101708



#### 1.2. Environement

##### 1.2.1. Notre dépôt gitHub

Branches: 

- main (master): https://github.com/olfabre/amsProjetMaster1
- olivier: https://github.com/olfabre/amsProjetMaster1/tree/olivier
- françois: https://github.com/olfabre/amsProjetMaster1/tree/francois



##### 1.2.2. Serveur Cuda (traitement GPU)

Serveur attaché au CERI avec un accès ssh:  ssh -p 22 uapvxxxxx@joyeux.univ-avignon.fr

Activation avec la commande: ```bash > conda activate shake```

**Note:** l'accès au serveur Cuda est très difficile et souvent inaccessible avec l'apparition de problèmes récurrents. Nous avons été obligés de changer de serveur en utilisant finalement Colab Google. Nous avons perdu tous nos fichier et modèles car ils sont inaccessibles depuis deux semaines. On a été obligé de tout reprendre à zéro.



##### 1.2.3. Serveur Colab Google (traitement T4 GPU)

Service gratuit pour traiter nos données sur une architecture GPU relativement puissante.

Les codes ont été enregistrés sur notre dépôt: https://github.com/olfabre/amsProjetMaster1/tree/olivier

![1](1.jpg)



##### 1.2.4. Nos données

nous avons rassembler toutes nos données et corpus d'entrée dans un même lieu à l'adresse suivante: https://olivier-fabre.com/passwordgenius/

Egalement en plus des corpus qui nous ont été remis, nous avons trouvé des corpus de mot de passe très intéressants classés par force à l'adresse suivante: https://github.com/Infinitode/PWLDS

Ce sera un set de data qui va contribuer à augmenter l'efficacité de notre application.



##### 1.2.5. Le choix de PyTorch et de Python (version 3)

**PyTorch (syntaxe et simplicité)** : utilise une syntaxe qui ressemble beaucoup à Python standard, rendant le code plus lisible et plus proche de la programmation impérative. Cela rend la prise en main plus facile et le processus de développement plus fluide, surtout pour les débutants comme nous.

**PyTorch (déboguage et flexibilité )** : grâce à sa nature impérative, il est facile de déboguer en utilisant des outils classiques comme `pdb` ou simplement en imprimant les valeurs des variables. C'est aussi plus flexible pour les expérimentations rapides ou les architectures de réseaux complexes.

**PyTorch (Gestion de l'Autograd )** : L'API **autograd** de PyTorch est intégrée et très intuitive pour les opérations de rétropropagation. Elle suit les opérations en direct, ce qui permet d’appliquer des gradients facilement, rendant la manipulation des réseaux plus naturelle.

PyTorch est souvent préféré par la communauté de recherche en intelligence artificielle et en apprentissage automatique, en particulier pour les projets de recherche académiques et expérimentaux. Les papiers de recherche et tutoriels sur les nouvelles architectures de modèles de réseaux de neurones sont souvent publiés avec du code PyTorch.






#### 3. Travaux Pratiques (suite)

##### 2.1. Atelier 2 - Corpus de prénoms (Russes)

Ce programme (python) permet d’apprendre depuis un ensemble de prénoms pour la génération de prénoms. L'algorithme d'apprentissage est un réseau de neurones récurrent (RNN) qui prend en entrée un caractère à la fois et essai de prédire le suivant.

Lors de la première phase, il faut lui fournir donc des prénoms de textes dans un seul fichier pour le volet apprentissage. Un fichier contenant un ensemble de prénoms russes (contenant le plus grand nombre de prénoms) est donné dans la section E-UAPV du défi. D'autres langues sont disponibles ici : https://download.pytorch.org/tutorial/data.zip.

Nous avons adapté le code pour le faire tourner dans Colab Google



==**Version initiale N°1 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V1.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

```python
try:
    import unidecode
except ModuleNotFoundError:
    !pip install unidecode
    import unidecode

import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé: {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Chargement des données encours...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in unidecode.unidecode(s)
        if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de noms: {len(lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 128
n_layers = 2
lr = 0.005
bidirectional = True
max_length = 20

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size, device=device)

# Entraînement
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor.unsqueeze_(-1)
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i].to(device), hidden.to(device))
        l = criterion(output.to(device), target_line_tensor[i].to(device))
        loss += l
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / input_line_tensor.size(0)

def training(n_epochs, lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  Entrainement  |\n-----------\n")
    start = time.time()
    total_loss = 0
    for epoch in range(1, n_epochs + 1):
        input_line_tensor, target_line_tensor = random_training_example(lines)
        loss = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)
        total_loss += loss
        if epoch % 500 == 0:
            print(f"{time_since(start)} ({epoch}/{n_epochs}) Perte: {total_loss / epoch:.4f}")

# Génération de noms
def sample(decoder, start_letter='A'):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topi = output.topk(1)[1][0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)
        return output_name

def time_since(since):
    """Retourne le temps écoulé au format mm:ss"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 5000

    print("Demarrage entrainement...")
    training(n_epochs, lines, decoder, decoder_optimizer, criterion)

    print("\nGénération de noms:")
    for letter in "ABC":
        print(sample(decoder, letter))

```





==**Version améliorée N°2 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V2.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt



 **Résumé des améliorations**

1. **Sauvegarde automatique** : Le modèle avec la meilleure perte de validation est sauvegardé dans `best_model_generation_prenom.pth`.

2. **Division des données** : Les données sont divisées en 80% (entraînement), 10% (validation), et 10% (test).

3. Progrès affichés:

    - Affichage de la perte d'entraînement et de validation.
- Sauvegarde du modèle lorsqu'une meilleure perte de validation est atteinte.

```python
try:
    import unidecode
except ModuleNotFoundError:
    !pip install unidecode
    import unidecode

import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in unidecode.unidecode(s)
        if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.8 * len(lines))
valid_split = int(0.1 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 128
n_layers = 2
lr = 0.005
bidirectional = True
max_length = 20

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size, device=device)

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)  # Déplacement vers le bon dispositif
    hidden = decoder.init_hidden().to(device)  # Initialisation sur le bon dispositif
    decoder.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)  # Déplacement explicite
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)  # Déplacement explicite
        output, hidden = decoder(input_tensor, hidden.detach())  # Utilisation de detach
        l = criterion(output, target_tensor)
        loss += l
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / input_line_tensor.size(0)

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():  # Pas de calcul de gradients pendant la validation
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l
        return loss.item() / input_line_tensor.size(0)

def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        # Entraînement
        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        # Validation
        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        # Sauvegarde du meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"Époch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")

        if epoch % 500 == 0:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Perte validation : {val_loss:.4f}")

# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 5000

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nGénération de prénoms :")
    for letter in "ABC":
        print(sample(decoder, letter))

```



==**Version améliorée N°3 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V3.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

Voici une version mise à jour du code, nous incluons :

- Une augmentation du nombre d'époques : nous pouvons maintenant ajuster facilement le nombre d'époques avec un affichage clair des progrès.

- Nous souhaitons un affichage d'une série de prénoms générés : à chaque amélioration du modèle (meilleure perte de validation), une série de prénoms est générée et affichée.



**Augmentation des époques :**

- Le nombre d'époques a été augmenté :

  ```python
  n_epochs = 10000  # Augmentez le nombre d'époques ici
  ```

**Génération de prénoms après chaque amélioration du modèle :**

- La fonction `generate_series` génère des prénoms à partir d'une série de lettres de départ.

- Elle est appelée chaque fois que le modèle améliore sa perte de validation :

  ```python
  if val_loss < best_loss:
      generate_series(decoder)  # Affiche une série de prénoms
  ```

**Affichage régulier des progrès :**

- Le code affiche les pertes d'entraînement et de validation toutes les 500 époques :

  ```python
  if epoch % 500 == 0:
      print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Perte validation : {val_loss:.4f}")
  ```

**Génération finale des prénoms :**

- Après l'entraînement, une série finale de prénoms est générée :

  ```python
  print("\nGénération finale de prénoms :")
  generate_series(decoder, start_letters="JKLMNOP")
  ```



```python
try:
    import unidecode
except ModuleNotFoundError:
    !pip install unidecode
    import unidecode

import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in unidecode.unidecode(s)
        if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.8 * len(lines))
valid_split = int(0.1 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 128
n_layers = 2
lr = 0.005
bidirectional = True
max_length = 20

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size, device=device)

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)  # Déplacement vers le bon dispositif
    hidden = decoder.init_hidden().to(device)  # Initialisation sur le bon dispositif
    decoder.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)  # Déplacement explicite
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)  # Déplacement explicite
        output, hidden = decoder(input_tensor, hidden.detach())  # Utilisation de detach
        l = criterion(output, target_tensor)
        loss += l
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / input_line_tensor.size(0)

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():  # Pas de calcul de gradients pendant la validation
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l
        return loss.item() / input_line_tensor.size(0)

def sample(decoder, start_letter='A'):
    """Génère un prénom à partir d'une lettre de départ."""
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topi = output.topk(1)[1][0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)
        return output_name

def generate_series(decoder, start_letters="ABCDE"):
    """Génère une série de prénoms à partir de lettres de départ."""
    print("Prénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        # Entraînement
        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        # Validation
        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        # Sauvegarde du meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            generate_series(decoder)  # Affiche une série de prénoms

        if epoch % 500 == 0:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Perte validation : {val_loss:.4f}")

# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 10000  # Augmentez le nombre d'époques ici

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nGénération finale de prénoms :")
    generate_series(decoder, start_letters="JKLMNOP")

```



==**Version améliorée N°4 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V4.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt



### **Optimisations apportées**

1. **Précision ajoutée :**

   - La précision est calculée en divisant le nombre de prédictions correctes par le nombre total de caractères dans une séquence.
   - Elle est affichée pendant l'entraînement et la validation.

2. **Augmentation du nombre d'époques :**

   - Le nombre d'époques est défini à 200,000 :

     ```python
     n_epochs = 200000
     ```

3. **Ajustements des hyperparamètres :**

   - **Taille cachée (`hidden_size`)** : Augmentée à 256 pour une meilleure capacité d'apprentissage.
   - **Nombre de couches cachées (`n_layers`)** : Augmenté à 3 pour une meilleure représentation des données.
   - **Taux d'apprentissage (`lr`)** : Réduit à 0.003 pour une convergence plus stable.

4. **Affichage des progrès :**

   - Tous les 500 époques, les pertes et précisions d'entraînement et de validation sont affichées.
   - Lorsque le modèle est sauvegardé, la précision de validation est également affichée.

5. **Génération de prénoms :**

   - Une série de prénoms est générée et affichée chaque fois que le modèle s'améliore.

```python
try:
    import unidecode
except ModuleNotFoundError:
    !pip install unidecode
    import unidecode

import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in unidecode.unidecode(s)
        if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.8 * len(lines))
valid_split = int(0.1 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
bidirectional = True
max_length = 20


# Optimisation des paramètres et hyperparamètres
hidden_size = 256  # Augmentation de la taille cachée pour un meilleur apprentissage
n_layers = 3  # Augmentation du nombre de couches cachées
lr = 0.003  # Ajustement du taux d'apprentissage
n_epochs = 200000  # Nombre d'époques augmenté


# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions, 1, self.hidden_size, device=device)

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Pour calculer la précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0  # Pour calculer la précision
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        # Entraînement
        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        # Validation
        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        # Sauvegarde du meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision de validation : {val_acc:.4f}")
            generate_series(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")



# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nGénération finale de prénoms :")
    generate_series(decoder, start_letters="JKLMNOP")
```



==**Version améliorée N°5 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V5.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 200000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A"):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topi = output.topk(1)[1][0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)
        return output_name

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

```





==**Version améliorée N°6 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V7.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

Le modèle ajuste dynamiquement son taux d’apprentissage.

Génération de prénoms après chaque amélioration du modèle.

Visualisation en direct des pertes et précisions.

Évaluation finale avec la perte moyenne et la précision.



```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 1000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A"):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topi = output.topk(1)[1][0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)
        return output_name

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)
```



==**Version améliorée N°7 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V8.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt



On souhaite seulement qu'à l'évaluation finale, ll génerere 20 prénoms

```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip().lower()) for line in f]

lines = read_lines(data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 1000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A"):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            topi = output.topk(1)[1][0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = input_tensor(letter)
        return output_name

# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

    # Génération de 20 prénoms avec le meilleur modèle
    print("\nPrénoms générés avec le meilleur modèle :")
    for _ in range(20):
        start_letter = random.choice(all_letters)  # Démarrer avec une lettre aléatoire
        print(f"- {sample(decoder, start_letter)}")


# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)
```



==**Version améliorée N°8 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V10.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

Nous avons améliorée de nouveau le code de façon à mieux générer des prénoms russes.
Cette version est la meilleur au terme de génération

```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt
import subprocess

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")
shuffled_data_path = os.path.join(data_dir, "russian_shuffled.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)





def shuffle_file(input_path, output_path):
    """
    Désordonne les lignes d'un fichier en utilisant la commande Bash `shuf`.
    """
    try:
        subprocess.run(['shuf', input_path, '-o', output_path], check=True)
        print(f"Fichier mélangé avec succès : {output_path}")
    except FileNotFoundError:
        print("Erreur : La commande `shuf` n'est pas disponible. Assurez-vous qu'elle est installée.")
        exit(1)








# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    # Filtrer et nettoyer les lignes
    clean_lines = []
    for line in lines:
        # Convertir en minuscules et supprimer les espaces autour
        line = line.strip().lower()
        # Vérifier que tous les caractères sont alphabétiques
        if all(c in string.ascii_letters for c in line) and len(line) >= 3:
            clean_lines.append(line)

    # Supprimer les doublons et trier les prénoms
    clean_lines = list(set(clean_lines))
    clean_lines.sort()

    return clean_lines


# Mélanger les lignes du fichier
shuffle_file(data_path, shuffled_data_path)

# Charger le fichier mélangé
lines = read_lines(shuffled_data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 512
n_layers = 4
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 3000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A", temperature=0.8):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter.lower()  # Commencer en minuscule
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            # Appliquer la température
            probabilities = torch.exp(output / temperature)
            probabilities /= probabilities.sum()  # Normaliser les probabilités
            topi = torch.multinomial(probabilities, 1)[0][0]  # Échantillonnage multinomial
            if topi == n_letters - 1:  # Fin de chaîne
                break
            else:
                letter = all_letters[topi]
                if letter.isalpha():  # Garder uniquement les lettres
                    output_name += letter.lower()
                else:
                    break  # Arrêter si un caractère non alphabétique est généré
            input = input_tensor(letter)
        return output_name.capitalize()



# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

    # Génération de 20 prénoms uniques avec le meilleur modèle
    print("\nPrénoms générés avec le meilleur modèle :")
    generated_names = set()
    attempts = 0  # Limiter les tentatives pour éviter les boucles infinies
    while len(generated_names) < 20 and attempts < 50:
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Assurer une taille minimale de 3 lettres
            generated_names.add(name)
        attempts += 1

    # Afficher les prénoms générés
    for name in sorted(generated_names):  # Trier pour lisibilité
        print(f"- {name}")




# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    #decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)
```

hidden_size = 512
n_layers = 4
lr = 0.003
Dropout(0.3)

train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))



==**Version améliorée N°9 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V11.ipynb.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt

Nous avons ajouté un test de couverture

```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt
import subprocess

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")
shuffled_data_path = os.path.join(data_dir, "russian_shuffled.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)





def shuffle_file(input_path, output_path):
    """
    Désordonne les lignes d'un fichier en utilisant la commande Bash `shuf`.
    """
    try:
        subprocess.run(['shuf', input_path, '-o', output_path], check=True)
        print(f"Fichier mélangé avec succès : {output_path}")
    except FileNotFoundError:
        print("Erreur : La commande `shuf` n'est pas disponible. Assurez-vous qu'elle est installée.")
        exit(1)








# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    # Filtrer et nettoyer les lignes
    clean_lines = []
    for line in lines:
        # Convertir en minuscules et supprimer les espaces autour
        line = line.strip().lower()
        # Vérifier que tous les caractères sont alphabétiques
        if all(c in string.ascii_letters for c in line) and len(line) >= 3:
            clean_lines.append(line)

    # Supprimer les doublons et trier les prénoms
    clean_lines = list(set(clean_lines))
    clean_lines.sort()

    return clean_lines


# Mélanger les lignes du fichier
shuffle_file(data_path, shuffled_data_path)

# Charger le fichier mélangé
lines = read_lines(shuffled_data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 3000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A", temperature=0.8):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter.lower()  # Commencer en minuscule
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            # Appliquer la température
            probabilities = torch.exp(output / temperature)
            probabilities /= probabilities.sum()  # Normaliser les probabilités
            topi = torch.multinomial(probabilities, 1)[0][0]  # Échantillonnage multinomial
            if topi == n_letters - 1:  # Fin de chaîne
                break
            else:
                letter = all_letters[topi]
                if letter.isalpha():  # Garder uniquement les lettres
                    output_name += letter.lower()
                else:
                    break  # Arrêter si un caractère non alphabétique est généré
            input = input_tensor(letter)
        return output_name.capitalize()



# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

    # Génération de 20 prénoms uniques avec le meilleur modèle
    print("\nPrénoms générés avec le meilleur modèle :")
    generated_names = set()
    attempts = 0  # Limiter les tentatives pour éviter les boucles infinies
    while len(generated_names) < 20 and attempts < 50:
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Assurer une taille minimale de 3 lettres
            generated_names.add(name)
        attempts += 1

    # Afficher les prénoms générés
    for name in sorted(generated_names):  # Trier pour lisibilité
        print(f"- {name}")


# Test de couverture : Générer 10 000 prénoms et calculer le pourcentage dans le corpus
def test_coverage(decoder, lines, num_samples=10000):
    """
    Génère `num_samples` prénoms et calcule le pourcentage de prénoms présents dans le corpus.
    """
    print("\n-----------\n|  TEST DE COUVERTURE |\n-----------\n")
    generated_names = set()
    corpus_set = set(lines)  # Transformer les prénoms du corpus en un ensemble pour une recherche rapide
    matches = 0

    for _ in range(num_samples):
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Vérifier que le prénom généré a au moins 3 lettres
            generated_names.add(name)
            if name.lower() in corpus_set:  # Vérifier si le prénom est dans le corpus (insensible à la casse)
                matches += 1

    coverage = (matches / num_samples) * 100
    print(f"Prénoms générés : {len(generated_names)} uniques sur {num_samples} générés.")
    print(f"Couverture : {coverage:.2f}% des prénoms générés sont présents dans le corpus.")
    return coverage



# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    #decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)

    # Appel du test de couverture
    coverage = test_coverage(decoder, train_lines)
```



----------- |  TEST DE COUVERTURE | ----------- 

Prénoms générés : 9295 uniques sur 10000 générés. 

Couverture : 0.30% des prénoms générés sont présents dans le corpus.



```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt
import subprocess

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")
shuffled_data_path = os.path.join(data_dir, "russian_shuffled.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)





def shuffle_file(input_path, output_path):
    """
    Désordonne les lignes d'un fichier en utilisant la commande Bash `shuf`.
    """
    try:
        subprocess.run(['shuf', input_path, '-o', output_path], check=True)
        print(f"Fichier mélangé avec succès : {output_path}")
    except FileNotFoundError:
        print("Erreur : La commande `shuf` n'est pas disponible. Assurez-vous qu'elle est installée.")
        exit(1)








# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    # Filtrer et nettoyer les lignes
    clean_lines = []
    for line in lines:
        # Convertir en minuscules et supprimer les espaces autour
        line = line.strip().lower()
        # Vérifier que tous les caractères sont alphabétiques
        if all(c in string.ascii_letters for c in line) and len(line) >= 3:
            clean_lines.append(line)

    # Supprimer les doublons et trier les prénoms
    clean_lines = list(set(clean_lines))
    clean_lines.sort()

    return clean_lines


# Mélanger les lignes du fichier
shuffle_file(data_path, shuffled_data_path)

# Charger le fichier mélangé
lines = read_lines(shuffled_data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.8 * len(lines))
valid_split = int(0.1 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 3000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A", temperature=0.8):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter.lower()  # Commencer en minuscule
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            # Appliquer la température
            probabilities = torch.exp(output / temperature)
            probabilities /= probabilities.sum()  # Normaliser les probabilités
            topi = torch.multinomial(probabilities, 1)[0][0]  # Échantillonnage multinomial
            if topi == n_letters - 1:  # Fin de chaîne
                break
            else:
                letter = all_letters[topi]
                if letter.isalpha():  # Garder uniquement les lettres
                    output_name += letter.lower()
                else:
                    break  # Arrêter si un caractère non alphabétique est généré
            input = input_tensor(letter)
        return output_name.capitalize()



# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

    # Génération de 20 prénoms uniques avec le meilleur modèle
    print("\nPrénoms générés avec le meilleur modèle :")
    generated_names = set()
    attempts = 0  # Limiter les tentatives pour éviter les boucles infinies
    while len(generated_names) < 20 and attempts < 50:
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Assurer une taille minimale de 3 lettres
            generated_names.add(name)
        attempts += 1

    # Afficher les prénoms générés
    for name in sorted(generated_names):  # Trier pour lisibilité
        print(f"- {name}")


# Test de couverture : Générer 10 000 prénoms et calculer le pourcentage dans le corpus
def test_coverage(decoder, lines, num_samples=10000):
    """
    Génère `num_samples` prénoms et calcule le pourcentage de prénoms présents dans le corpus.
    """
    print("\n-----------\n|  TEST DE COUVERTURE |\n-----------\n")
    generated_names = set()
    corpus_set = set(lines)  # Transformer les prénoms du corpus en un ensemble pour une recherche rapide
    matches = 0

    for _ in range(num_samples):
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Vérifier que le prénom généré a au moins 3 lettres
            generated_names.add(name)
            if name.lower() in corpus_set:  # Vérifier si le prénom est dans le corpus (insensible à la casse)
                matches += 1

    coverage = (matches / num_samples) * 100
    print(f"Prénoms générés : {len(generated_names)} uniques sur {num_samples} générés.")
    print(f"Couverture : {coverage:.2f}% des prénoms générés sont présents dans le corpus.")
    return coverage



# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    #decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)

    # Appel du test de couverture
    coverage = test_coverage(decoder, train_lines)
```



==**Version améliorée N°10 pour Colab Google**==

Dépôt: https://github.com/olfabre/amsProjetMaster1/blob/olivier/Generation_prenoms_V11.ipynb.ipynb

Data set: https://olivier-fabre.com/passwordgenius/russian.txt



nous l'avons amélioré avec un réseau de neurones LSTM

```python
import requests
import torch
import torch.nn as nn
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt
import subprocess
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")
shuffled_data_path = os.path.join(data_dir, "russian_shuffled.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)

def shuffle_file(input_path, output_path):
    """
    Désordonne les lignes d'un fichier en utilisant la commande Bash `shuf`.
    """
    try:
        subprocess.run(['shuf', input_path, '-o', output_path], check=True)
        print(f"Fichier mélangé avec succès : {output_path}")
    except FileNotFoundError:
        print("Erreur : La commande `shuf` n'est pas disponible. Assurez-vous qu'elle est installée.")
        exit(1)

# Chargement des données
def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        line = line.strip().lower()
        if all(c in string.ascii_letters for c in line) and len(line) >= 3:
            clean_lines.append(line)

    clean_lines = list(set(clean_lines))
    clean_lines.sort()
    return clean_lines

# Mélanger les lignes du fichier
shuffle_file(data_path, shuffled_data_path)

# Charger le fichier mélangé
lines = read_lines(shuffled_data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.8 * len(lines))
valid_split = int(0.1 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 15000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class LSTMLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3, bidirectional=True):
        super(LSTMLight, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input.unsqueeze(0), hidden)
        output = self.out(output.squeeze(0))
        return self.softmax(output), hidden

    def init_hidden(self):
        return (
            torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device),
            torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device),
        )

# Entraînement
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    hidden = decoder.init_hidden()
    decoder_optimizer.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = decoder(input_line_tensor[i].to(device), hidden)
        l = criterion(output, target_line_tensor[i].to(device).unsqueeze(0))
        loss += l

    loss.backward()
    decoder_optimizer.step()
    return loss.item() / input_line_tensor.size(0)

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        loss = 0

        for i in range(input_line_tensor.size(0)):
            output, hidden = decoder(input_line_tensor[i].to(device), hidden)
            l = criterion(output, target_line_tensor[i].to(device).unsqueeze(0))
            loss += l

        return loss.item() / input_line_tensor.size(0)

# Fonction d'entraînement principale
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"
    scheduler = ReduceLROnPlateau(decoder_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):
        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Perte validation : {val_loss:.4f}")

            # Affichage des graphiques
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Perte d'entraînement")
            plt.plot(val_losses, label="Perte de validation")
            plt.legend()
            plt.show()

# Génération de prénoms
def sample(decoder, start_letter="A", temperature=0.8):
    with torch.no_grad():
        hidden = decoder.init_hidden()  # hidden est maintenant un tuple (hidden_state, cell_state)
        input = input_tensor(start_letter)
        output_name = start_letter.lower()  # Commencer en minuscule

        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), (hidden[0].to(device), hidden[1].to(device)))

            # Appliquer la température
            probabilities = torch.exp(output / temperature)
            probabilities /= probabilities.sum()  # Normaliser les probabilités
            topi = torch.multinomial(probabilities, 1)[0][0]  # Échantillonnage multinomial

            if topi == n_letters - 1:  # Fin de chaîne
                break
            else:
                letter = all_letters[topi]
                if letter.isalpha():  # Garder uniquement les lettres
                    output_name += letter.lower()
                else:
                    break  # Arrêter si un caractère non alphabétique est généré

            input = input_tensor(letter)
        return output_name.capitalize()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += 1 if loss < 0.5 else 0
            total_samples += 1

    avg_loss = total_loss / len(test_lines)
    accuracy = total_correct / total_samples
    print(f"Perte moyenne : {avg_loss:.4f}, Précision moyenne : {accuracy:.4f}")

# Test de couverture
def test_coverage(decoder, lines, num_samples=10000):
    print("\n-----------\n|  TEST DE COUVERTURE |\n-----------\n")
    generated_names = set()
    corpus_set = set(lines)
    matches = 0

    for _ in range(num_samples):
        start_letter = random.choice(string.ascii_uppercase)
        name = sample(decoder, start_letter)
        if len(name) >= 3:
            generated_names.add(name)
            if name.lower() in corpus_set:
                matches += 1

    coverage = (matches / num_samples) * 100
    print(f"Prénoms générés : {len(generated_names)} uniques sur {num_samples} générés.")
    print(f"Couverture : {coverage:.2f}%")
    return coverage

# Exécution principale
if __name__ == "__main__":
    decoder = LSTMLight(n_letters, hidden_size, n_letters).to(device)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device)
    decoder.load_state_dict(state_dict)

    evaluate_model(test_lines, decoder, criterion)
    test_coverage(decoder, train_lines)

```

**Détail ligne par ligne du code**

1-8 : **Importations et vérification du GPU**

- Importation des bibliothèques nécessaires : `requests` pour télécharger des données, `torch` pour PyTorch, `time` pour la gestion du temps, `math` pour les fonctions mathématiques, `string` pour manipuler des caractères, et `random` pour les opérations aléatoires.
- `torch.device` détermine si un GPU (CUDA) est disponible. Sinon, la CPU est utilisée. Cela optimise les calculs si un GPU est disponible.

9-22 : **Téléchargement et configuration des fichiers**

- Définit un chemin vers un fichier texte contenant des données.
- Vérifie si le fichier `russian.txt` existe déjà. Si non, télécharge son contenu depuis l’URL spécifiée et le sauvegarde localement.

23-32 : **Fonction de mélange**

- Implémente une fonction pour mélanger les lignes d’un fichier texte en utilisant la commande Bash `shuf`.
- Si `shuf` n’est pas installé, affiche un message d’erreur.

33-49 : **Chargement et nettoyage des données**

- `read_lines` charge un fichier ligne par ligne, nettoie chaque ligne (supprime les espaces inutiles, convertit en minuscules).
- Garde uniquement les lignes avec des caractères alphabétiques d'une certaine longueur (≥ 3).
- Élimine les doublons et trie les lignes.

50-54 : **Mélange et chargement des lignes**

- Mélange les lignes du fichier source et charge les données mélangées en mémoire.

55-62 : **Division des données**

- Mélange les lignes aléatoirement.
- Divise les données en trois ensembles : entraînement (80%), validation (10%), et test (10%).

63-70 : **Paramètres globaux**

- Définit des variables pour les caractères acceptés, la taille des données d’entrée et de sortie, la structure du réseau (taille des couches cachées, directionnalité, etc.), et les paramètres d’entraînement (taux d’apprentissage, nombre d’époques).

71-93 : **Fonctions utilitaires pour les tensors**

- Transforme les chaînes de caractères en tenseurs utilisables par PyTorch pour les entrées et cibles du modèle.
- `random_training_example` sélectionne un exemple aléatoire pour l’entraînement.

94-99 : **Gestion du temps**

- `time_since` mesure et formate la durée écoulée depuis un temps donné.

100-120 : **Définition du modèle LSTM**

- ```
  LSTMLight
  ```

   est une classe pour un réseau LSTM :

  - Comporte une couche LSTM bidirectionnelle ou unidirectionnelle.
  - Une couche linéaire transforme la sortie LSTM en une distribution sur les caractères possibles.
  - Utilise une activation `LogSoftmax`.

121-129 : **Fonction d’entraînement**

- Entraîne le modèle sur un exemple unique en calculant une perte, effectue une rétropropagation, et met à jour les paramètres.

130-139 : **Validation**

- Évalue le modèle sur des données de validation sans rétropropagation pour mesurer la qualité du modèle.

140-176 : **Entraînement principal**

- Gère plusieurs époques :
  - Entraîne sur les données d’entraînement.
  - Valide sur les données de validation.
  - Sauvegarde le modèle si la perte de validation s’améliore.
  - Réduit dynamiquement le taux d’apprentissage avec `ReduceLROnPlateau`.
  - Affiche les courbes de perte d’entraînement et de validation.

177-200 : **Génération de prénoms**

- `sample` génère un prénom à partir d’une lettre initiale donnée en utilisant le modèle LSTM.
- Applique une température pour ajuster la créativité de la génération.

201-217 : **Évaluation finale**

- Évalue la performance globale du modèle sur l’ensemble de test.
- Calcule la perte moyenne et une précision basée sur un seuil.

218-232 : **Test de couverture**

- Génère des prénoms aléatoires, calcule combien apparaissent dans les données d’origine, et estime la "couverture".

233-243 : **Exécution principale**

- Crée une instance du modèle LSTM.
- Entraîne le modèle, charge le meilleur modèle sauvegardé, et l’évalue sur les ensembles de test.
- Effectue un test de couverture.

Ce code constitue une chaîne complète pour la génération de prénoms avec un LSTM, depuis la préparation des données jusqu'à l'évaluation finale.



Voici le code et les paramètres choisis après exploration et reflexion

```python
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import random
import os
import matplotlib.pyplot as plt
import subprocess

# Vérification GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# Téléchargement des données
url = "https://olivier-fabre.com/passwordgenius/russian.txt"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
data_path = os.path.join(data_dir, "russian.txt")
shuffled_data_path = os.path.join(data_dir, "russian_shuffled.txt")

if not os.path.exists(data_path):
    print("Téléchargement des données...")
    response = requests.get(url)
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(response.text)





def shuffle_file(input_path, output_path):
    """
    Désordonne les lignes d'un fichier en utilisant la commande Bash `shuf`.
    """
    try:
        subprocess.run(['shuf', input_path, '-o', output_path], check=True)
        print(f"Fichier mélangé avec succès : {output_path}")
    except FileNotFoundError:
        print("Erreur : La commande `shuf` n'est pas disponible. Assurez-vous qu'elle est installée.")
        exit(1)








# Chargement des données
def unicode_to_ascii(s):
    return ''.join(
        c for c in s if c in (string.ascii_letters + " .,;'-")
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()

    # Filtrer et nettoyer les lignes
    clean_lines = []
    for line in lines:
        # Convertir en minuscules et supprimer les espaces autour
        line = line.strip().lower()
        # Vérifier que tous les caractères sont alphabétiques
        if all(c in string.ascii_letters for c in line) and len(line) >= 3:
            clean_lines.append(line)

    # Supprimer les doublons et trier les prénoms
    clean_lines = list(set(clean_lines))
    clean_lines.sort()

    return clean_lines


# Mélanger les lignes du fichier
shuffle_file(data_path, shuffled_data_path)

# Charger le fichier mélangé
lines = read_lines(shuffled_data_path)
print(f"Nombre de prénoms : {len(lines)}")

# Division des données
random.shuffle(lines)
train_split = int(0.7 * len(lines))
valid_split = int(0.2 * len(lines))
train_lines = lines[:train_split]
valid_lines = lines[train_split:train_split + valid_split]
test_lines = lines[train_split + valid_split:]
print(f"Ensemble d'entraînement : {len(train_lines)}, Validation : {len(valid_lines)}, Test : {len(test_lines)}")

# Paramètres globaux
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # EOS marker
hidden_size = 256
n_layers = 3
lr = 0.003
bidirectional = True
max_length = 20
n_epochs = 3000

# Fonctions utilitaires
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_letters.index(string[c])
    return tensor

def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

def random_training_example(lines):
    line = random.choice(lines)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return input_line_tensor, target_line_tensor

# Fonction pour afficher le temps écoulé
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s:.2f}s"

# Définition du modèle
class RNNLight(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLight, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, bidirectional=self.bidirectional, batch_first=True
        )
        self.out = nn.Linear(self.num_directions * hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        _, hidden = self.rnn(input.unsqueeze(0), hidden)
        hidden_concat = hidden if not self.bidirectional else torch.cat((hidden[0], hidden[1]), 1)
        output = self.out(hidden_concat)
        output = self.dropout(output)
        return self.softmax(output), hidden

    def init_hidden(self):
        return torch.zeros(self.num_directions * n_layers, 1, self.hidden_size, device=device)

# Fonction pour générer des prénoms
def generate_prenoms(decoder, start_letters="ABCDE"):
    print("\nPrénoms générés :")
    for letter in start_letters:
        print(f"- {sample(decoder, letter)}")

def sample(decoder, start_letter="A", temperature=0.8):
    with torch.no_grad():
        hidden = decoder.init_hidden()
        input = input_tensor(start_letter)
        output_name = start_letter.lower()  # Commencer en minuscule
        for _ in range(max_length):
            output, hidden = decoder(input[0].to(device), hidden.to(device))
            # Appliquer la température
            probabilities = torch.exp(output / temperature)
            probabilities /= probabilities.sum()  # Normaliser les probabilités
            topi = torch.multinomial(probabilities, 1)[0][0]  # Échantillonnage multinomial
            if topi == n_letters - 1:  # Fin de chaîne
                break
            else:
                letter = all_letters[topi]
                if letter.isalpha():  # Garder uniquement les lettres
                    output_name += letter.lower()
                else:
                    break  # Arrêter si un caractère non alphabétique est généré
            input = input_tensor(letter)
        return output_name.capitalize()



# Entraînement avec sauvegarde
def train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion):
    target_line_tensor = target_line_tensor.to(device)
    hidden = decoder.init_hidden().to(device)
    decoder.zero_grad()
    loss = 0
    correct = 0  # Précision
    total = target_line_tensor.size(0)

    for i in range(input_line_tensor.size(0)):
        input_tensor = input_line_tensor[i].to(device)
        target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
        output, hidden = decoder(input_tensor, hidden.detach())
        l = criterion(output, target_tensor)
        loss += l

        # Calcul de la précision
        predicted = output.topk(1)[1][0][0]
        correct += (predicted == target_tensor[0]).item()

    loss.backward()
    decoder_optimizer.step()

    accuracy = correct / total
    return loss.item() / input_line_tensor.size(0), accuracy

def validation(input_line_tensor, target_line_tensor, decoder, criterion):
    with torch.no_grad():
        target_line_tensor = target_line_tensor.to(device)
        hidden = decoder.init_hidden().to(device)
        loss = 0
        correct = 0
        total = target_line_tensor.size(0)

        for i in range(input_line_tensor.size(0)):
            input_tensor = input_line_tensor[i].to(device)
            target_tensor = target_line_tensor[i].unsqueeze(0).to(device)
            output, hidden = decoder(input_tensor, hidden.detach())
            l = criterion(output, target_tensor)
            loss += l

            # Calcul de la précision
            predicted = output.topk(1)[1][0][0]
            correct += (predicted == target_tensor[0]).item()

        accuracy = correct / total
        return loss.item() / input_line_tensor.size(0), accuracy

# Ajustement dynamique du taux d'apprentissage
def adjust_learning_rate(optimizer, epoch, decay_rate=0.5, step=20000):
    if epoch % step == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Taux d'apprentissage ajusté à : {param_group['lr']}")

# Suivi des pertes et précisions
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Fonction principale d'entraînement
def training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion):
    print("\n-----------\n|  ENTRAÎNEMENT  |\n-----------\n")
    start = time.time()
    best_loss = float("inf")
    model_path = "best_model_generation_prenom.pth"

    for epoch in range(1, n_epochs + 1):
        adjust_learning_rate(decoder_optimizer, epoch)

        input_line_tensor, target_line_tensor = random_training_example(train_lines)
        train_loss, train_acc = train(input_line_tensor, target_line_tensor, decoder, decoder_optimizer, criterion)

        input_line_tensor, target_line_tensor = random_training_example(valid_lines)
        val_loss, val_acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(decoder.state_dict(), model_path)
            print(f"\nÉpoch {epoch} : La perte de validation a diminué à {best_loss:.4f}. Modèle sauvegardé.")
            print(f"Précision validation : {val_acc:.4f}")
            generate_prenoms(decoder)

        if epoch % 500 == 0 or epoch == 1:
            print(f"{time_since(start)} Époch {epoch}/{n_epochs}, Perte entraînement : {train_loss:.4f}, Précision entraînement : {train_acc:.4f}")
            print(f"Perte validation : {val_loss:.4f}, Précision validation : {val_acc:.4f}")

            # Afficher les graphiques interactifs
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Perte Entraînement')
            plt.plot(val_losses, label='Perte Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Perte')
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(train_accuracies, label='Précision Entraînement')
            plt.plot(val_accuracies, label='Précision Validation')
            plt.legend()
            plt.xlabel('Époques')
            plt.ylabel('Précision')
            plt.show()

# Évaluation finale
def evaluate_model(test_lines, decoder, criterion):
    print("\n-----------\n|  ÉVALUATION FINALE |\n-----------\n")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    decoder.eval()

    with torch.no_grad():
        for line in test_lines:
            input_line_tensor = input_tensor(line)
            target_line_tensor = target_tensor(line)
            loss, acc = validation(input_line_tensor, target_line_tensor, decoder, criterion)
            total_loss += loss
            total_correct += acc * len(line)
            total_samples += len(line)

    avg_loss = total_loss / len(test_lines)
    avg_accuracy = total_correct / total_samples
    print(f"Perte moyenne sur l'ensemble de test : {avg_loss:.4f}")
    print(f"Précision moyenne sur l'ensemble de test : {avg_accuracy:.4f}")

    # Génération de 20 prénoms uniques avec le meilleur modèle
    print("\nPrénoms générés avec le meilleur modèle :")
    generated_names = set()
    attempts = 0  # Limiter les tentatives pour éviter les boucles infinies
    while len(generated_names) < 20 and attempts < 50:
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Assurer une taille minimale de 3 lettres
            generated_names.add(name)
        attempts += 1

    # Afficher les prénoms générés
    for name in sorted(generated_names):  # Trier pour lisibilité
        print(f"- {name}")


# Test de couverture : Générer 10 000 prénoms et calculer le pourcentage dans le corpus
def test_coverage(decoder, lines, num_samples=10000):
    """
    Génère `num_samples` prénoms et calcule le pourcentage de prénoms présents dans le corpus.
    """
    print("\n-----------\n|  TEST DE COUVERTURE |\n-----------\n")
    generated_names = set()
    corpus_set = set(lines)  # Transformer les prénoms du corpus en un ensemble pour une recherche rapide
    matches = 0

    for _ in range(num_samples):
        start_letter = random.choice(string.ascii_uppercase)  # Démarrer avec une lettre majuscule
        name = sample(decoder, start_letter)
        if len(name) >= 3:  # Vérifier que le prénom généré a au moins 3 lettres
            generated_names.add(name)
            if name.lower() in corpus_set:  # Vérifier si le prénom est dans le corpus (insensible à la casse)
                matches += 1

    coverage = (matches / num_samples) * 100
    print(f"Prénoms générés : {len(generated_names)} uniques sur {num_samples} générés.")
    print(f"Couverture : {coverage:.2f}% des prénoms générés sont présents dans le corpus.")
    return coverage



# Exécution principale
if __name__ == "__main__":
    decoder = RNNLight(n_letters, hidden_size, n_letters).to(device)
    #decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-5)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    print("Démarrage de l'entraînement...")
    training(n_epochs, train_lines, valid_lines, decoder, decoder_optimizer, criterion)

    print("\nChargement du meilleur modèle...")
    # Chargement sécurisé pour éviter tout code malveillant
    state_dict = torch.load("best_model_generation_prenom.pth", map_location=device, weights_only=True)
    decoder.load_state_dict(state_dict)
    evaluate_model(test_lines, decoder, criterion)

    # Appel du test de couverture
    coverage = test_coverage(decoder, train_lines)
```

```
poch 2813 : La perte de validation a diminué à 1.0691. Modèle sauvegardé.
Précision validation : 0.8333

Prénoms générés :
- Aldgkkov
- Btezteko
- Cerevekyud
- Darimkelodody
- Eiumarekoky
1m 8.93s Époch 3000/3000, Perte entraînement : 3.8394, Précision entraînement : 0.1667
Perte validation : 3.5413, Précision validation : 0.1818
```



```
Chargement du meilleur modèle...

-----------
|  ÉVALUATION FINALE |
-----------

Perte moyenne sur l'ensemble de test : 2.7502
Précision moyenne sur l'ensemble de test : 0.2659

Prénoms générés avec le meilleur modèle :
- Aemelen
- Bunoreno
- Cerevel
- Cskreov
- Earnoleky
- Fumroso
- Gregerov
- Idensaeko
- Kukokitolyhor
- Lalrrovslo
- Lkatyulovlo
- Naberkog
- Naerdnov
- Ncheolov
- Oozovach
- Qzanyrov
- Wanrrtkn
- Xolmdolov
- Xvozielhollikho
- Zanargov

-----------
|  TEST DE COUVERTURE |
-----------

Prénoms générés : 9767 uniques sur 10000 générés.
Couverture : 0.08% des prénoms générés sont présents dans le corpus.
```
