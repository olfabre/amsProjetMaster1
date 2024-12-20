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

![img](data:image/png;base64,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oVClQWRdmJ6A397cS4PEhB6WBBERERGp1eptAAuCSBt8yxQoDIKIiEgV3qKJ2rLwxCAKKgyCiIjIZ6waR+HuwVFdkZMcgwUji41OChGpwDZBRESkisTIh8hh5tBOuPvaQp4XpANWstQTS4KIiMhnfOwj4osB0hCPpYBhEERERKrwFk1ERMGOQRARERERkRlExlz52xJlXDrCgKFB0KZNm3DjjTciJycHkiRh9erVRiaHiIgUYG0NIiKdxNqAMU8DN/wesCYYnZqQZmgQdPbsWZSWluKFF14wMhlEROQjtoUgItJY/2lAvzuNTkXIM7R3uNGjR2P06NFGJoGIiFRi4ENERMEuqLrIbmhoQENDg+P/U6dOGZgaIiJiOERERMEoqDpGWLJkCZKTkx0/ubm5RieJiIiIiIiCTFAFQYsWLUJ9fb3j5/Dhw0YniYiIiIiIgkxQVYezWq2wWq1GJ4OIiIiIiIJYUJUEERGRubCPBCIiCkaGlgSdOXMGBw4ccPxfV1eHnTt3IjU1FXl5eQamjIiIiIiIQpWhQdD27dsxbNgwx//z5s0DAFRXV2P58uUGpYqIiIiIiEKZoUHQ0KFDIYQwMglERERERBRm2CaIiIh8lp8Wb3QSiIiIVAuq3uGIiMgcdv7iOjQ0NSMpJsropBAREanGIIiIiFSzxUUbnQQiIiKfsTocERERERGFFQZBREREREQUVhgEERERERFRWGEQREREREREYYVBEBERERERhRUGQUREREREFFYYBBERERERUVhhEERERERERGGFQRAREREREYUVBkFERERERBRWGAQREREREVFYYRBERERERERhhUEQERERERGFFQZBREREREQUViKNToA/hBAAgFOnThmcEiIiIiIiMlJLTNASI3gS1EHQ6dOnAQC5ubkGp4SIiIiIiMzg9OnTSE5O9jiPJJSESibV3NyMf//730hMTIQkSYam5dSpU8jNzcXhw4eRlJRkaFpIO8zX0MM8DU3M19DDPA1NzNfQY6Y8FULg9OnTyMnJgcXiudVPUJcEWSwWdOjQwehkyCQlJRl+AJD2mK+hh3kampivoYd5GpqYr6HHLHnqrQSoBTtGICIiIiKisMIgiIiIiIiIwgqDII1YrVYsXrwYVqvV6KSQhpivoYd5GpqYr6GHeRqamK+hJ1jzNKg7RiAiIiIiIlKLJUFERERERBRWGAQREREREVFYYRBERERERERhhUEQERERERGFFQZBGnnhhReQn5+PmJgYlJWV4bPPPjM6SeTGY489BkmSZD9du3Z1TL9w4QJmz56Ndu3aISEhARMnTsSxY8dk6zh06BDGjBmDuLg4ZGRkYMGCBWhqagr0VwlbmzZtwo033oicnBxIkoTVq1fLpgsh8Itf/ALZ2dmIjY1FZWUlvvnmG9k8J06cQFVVFZKSkmCz2XDXXXfhzJkzsnl2796NwYMHIyYmBrm5uXjyySf1/mphzVu+Tpkypc25O2rUKNk8zFdzWbJkCfr374/ExERkZGRg3Lhx2L9/v2wera65NTU16NOnD6xWK4qKirB8+XK9v15YUpKnQ4cObXOu3n333bJ5mKfmsnTpUvTq1csx4Gl5eTnWrFnjmB6S56kgv61YsUJER0eLl19+WXz55Zdi+vTpwmaziWPHjhmdNHJh8eLFokePHuLIkSOOn++//94x/e677xa5ubliw4YNYvv27eLqq68WAwcOdExvamoSJSUlorKyUnzxxRfi/fffF2lpaWLRokVGfJ2w9P7774uHH35YvPXWWwKAWLVqlWz6E088IZKTk8Xq1avFrl27xE033SQKCgrE+fPnHfOMGjVKlJaWiq1bt4qPP/5YFBUViVtvvdUxvb6+XmRmZoqqqipRW1srXn/9dREbGyv++Mc/Buprhh1v+VpdXS1GjRolO3dPnDghm4f5ai4jR44Uy5YtE7W1tWLnzp3i+uuvF3l5eeLMmTOOebS45v7jH/8QcXFxYt68eWLv3r3i+eefFxEREWLt2rUB/b7hQEmeXnvttWL69Omyc7W+vt4xnXlqPu+884547733xNdffy32798vHnroIREVFSVqa2uFEKF5njII0sCAAQPE7NmzHf9funRJ5OTkiCVLlhiYKnJn8eLForS01OW0kydPiqioKLFy5UrHZ/v27RMAxJYtW4QQ9gc1i8Uijh496phn6dKlIikpSTQ0NOiadmqr9cNyc3OzyMrKEr/73e8cn508eVJYrVbx+uuvCyGE2Lt3rwAgPv/8c8c8a9asEZIkiX/9619CCCFefPFFkZKSIsvTBx98UBQXF+v8jUiItvkqhD0IGjt2rNtlmK/md/z4cQFAbNy4UQih3TX3gQceED169JBta/LkyWLkyJF6f6Ww1zpPhbAHQXPnznW7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jx6V/fzwww+O6StXrsTLL7+Mr7/+GosXL8Znn33m6PigqqoKMTExqK6uRm1tLT766CPcc889uP3225GZmQkAmDt3Lp544gmsXr0aX331FWbNmoWTJ0/K0jB8+HC8+uqr+Pjjj7Fnzx5UV1cjIiLCMb2yshLl5eUYN24c1q1bh2+//RaffvopHn74YWzfvh3nz5/HnDlzUFNTg++++w6ffPIJPv/8c0ewRkREwYnV4YiISBdr165Fdna27LPi4mJ89dVXAIDHH38cK1aswKxZs5CdnY3XX38d3bt3BwDExcXh73//O+bOnYv+/fsjLi4OEydOxDPPPONY1/z583HkyBFUV1fDYrHgzjvvxPjx41FfX++YZ9GiRairq8MNN9yA5ORk/OpXv5KVBEmShPfffx8PP/wwpk6diu+//x5ZWVkYMmQIMjMzERERgR9//BF33HEHjh07hrS0NEyYMAGPP/64nruOiIh0Jgk1LUiJiIg0IEkSVq1aJRvTRwtTpkzByZMnsXr1ak3XS0REoYXV4YiIiIiIKKwwCCIiIiIiorDC6nBERERERBRWWBJERERERERhhUEQERERERGFFQZBREREREQUVhgEERERERFRWGEQREREREREYYVBEBERERERhRUGQUREREREFFYYBBERERERUVj5f1AZMDaUhmuPAAAAAElFTkSuQmCC)

![img](data:image/png;base64,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REGwcCIiIiIiIgqChRMREREREVEQLJyIiIiIiIiCYOFEREREREQUBAsnIiIiIiKiIFg4ERHRkLN8+XIsXrwYPp8v1KEQEREBYOFERERDTH19PTIzM/H2229DqeTbFBERDQ0KEZFQB0FERERERDSU8as8IiIaEp566ikoFIprbjNnzgx1aERERFCFOgAiIqKrZs6cia1btwYs02g0IYqGiIjoazzjREREQ4ZGo4HNZgu4mc1mAIBCocDmzZsxa9YsaLVapKam4v333w9Y/+TJk5g2bRq0Wi1iYmKwePFidHZ2+tu9Xi9WrFgBk8mEmJgYvPDCC8jPz8fs2bP9fUaOHImNGzcGbHfixIlYu3at/7HL5cLChQsRGxsLg8GAadOmobS01N9eWlqKBx54AFFRUTAYDJg8eTKOHTs2eC8UERHdciyciIjotvHqq69i7ty5KC0txYIFC/Doo4+ivLwcAODxeDBjxgyYzWYcPXoU27dvx6effoply5b513/jjTewbds2bNmyBQcPHkRbWxt27NjxneOYN28empqasGvXLhQXF2PSpEmYPn062traAAALFizAiBEjcPToURQXF+Oll15CeHj44LwIREQUEiyciIhoyPjoo4+g1+sDbr/5zW/87fPmzcPChQsxatQovP7668jJycGmTZsAAO+++y56enrwt7/9DVlZWZg2bRr++Mc/4p133kFjYyMAYOPGjVi9ejXmzJmDMWPG4M9//jOMRuN3ivHgwYP48ssvsX37duTk5CAjIwO///3vYTKZ/GfA6urqkJeXh9GjRyMjIwPz5s1Ddnb2IL1KREQUCvyNExERDRkPPPAANm/eHLAsOjraf9/hcAS0ORwOlJSUAADKy8uRnZ2NyMhIf/s999wDn8+HiooKREREoKGhAbm5uf52lUqFnJwcfJcJZktLS9HZ2YmYmJiA5d3d3aiurgYArFixAgsXLsQ777yDvLw8zJs3D2lpad/6OYiIaOhh4URERENGZGQk0tPTQxqDUqm8ppDq7+/33+/s7ITdbkdhYeE165pMJgDA2rVr8fjjj+Pjjz/Grl27sGbNGhQUFOCRRx65maETEdFNxEv1iIjotnH48OFrHo8ZMwYAMGbMGJSWlsLj8fjbDx06BKVSiczMTBiNRtjtdhw5csTfPjAwgOLi4oBtxsbGoqGhwf/Y7XajpqbG/3jSpElwOp1QqVRIT08PuFksFn+/UaNG4Re/+AX27NmDOXPmXDNbIBER3V5YOBER0ZDR29sLp9MZcGtpafG3b9++HVu2bMHZs2exZs0afPnll/7JHxYsWICIiAjk5+ejrKwMBw4cwM9//nM88cQTsFqtAIDly5djw4YN2LlzJ86cOYMlS5bA5XIFxDBt2jS88847+OKLL3Dy5Enk5+cjLCzM356XlweHw4HZs2djz549qK2txb///W+88sorOHbsGLq7u7Fs2TIUFhbi/PnzOHToEI4ePeov8IiI6PbES/WIiGjI2L17N+x2e8CyzMxMnDlzBgCwbt06FBQUYMmSJbDb7XjvvfcwduxYAIBOp8O//vUvLF++HHfddRd0Oh3mzp2LN99807+tlStXoqGhAfn5+VAqlXjmmWfwyCOPoL293d9n9erVqKmpwY9//GMYjUa8/vrrAWecFAoFPvnkE7zyyit4+umn0dzcDJvNhqlTp8JqtSIsLAytra148skn0djYCIvFgjlz5mDdunU386UjIqKbTCHf5RexREREIaJQKLBjx46A/7k0GJ566im4XC7s3LlzULdLRETDCy/VIyIiIiIiCoKFExERERERURC8VI+IiIiIiCgInnEiIiIiIiIKgoUTERERERFRECyciIiIiIiIgmDhREREREREFAQLJyIiIiIioiBYOBEREREREQXBwomIiIiIiCgIFk5ERERERERBsHAiIiIiIiIK4v8Ad/B2DpkFp10AAAAASUVORK5CYII=)

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
