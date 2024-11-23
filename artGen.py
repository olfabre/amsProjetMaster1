import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd  # Pour le téléchargement et la manipulation de données
import time
import math
from os import path, makedirs

# Détection du device (CPU ou GPU)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')

# Variables globales
all_characters = string.printable
n_characters = len(all_characters)
chunk_len = 13

# Hyperparamètres
n_epochs = 200000
hidden_size = 512
n_layers = 3
lr = 0.005
model_dir = "models"
file_url = "https://olivier-fabre.com/passwordgenius/shakespeare2.txt"

# Téléchargement du fichier distant avec Pandas
def download_file_with_pandas(url):
    print(f"Téléchargement du fichier depuis {url} avec Pandas...")
    try:
        data = pd.read_csv(url, header=None)  # Chargement du fichier en tant que DataFrame
        text = "\n".join(data[0].tolist())   # Fusion des lignes en une seule chaîne
        print("Fichier téléchargé et chargé avec succès.")
        return unidecode.unidecode(text)     # Retrait des accents pour uniformiser
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        exit(1)

# Préparation des données
def random_chunk(file):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set(file):
    chunk = random_chunk(file)
    inp = char_tensor(chunk[:-1]).to(device)
    target = char_tensor(chunk[1:]).to(device)
    return inp, target

# Définition du modèle
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size, device=device))

# Fonctions d'entraînement
def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    for c in range(inp.size(0)):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / chunk_len

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def training(n_epochs, file, chunk_count=10):
    print("\n-----------\n|  TRAIN  |\n-----------\n")
    start = time.time()
    all_losses = []
    loss_avg = 0
    best_loss = float('inf')

    for epoch in range(1, n_epochs + 1):
        losses = []
        for _ in range(chunk_count):
            loss = train(*random_training_set(file))
            losses.append(loss)

        # Moyenne des pertes
        loss_avg += sum(losses) / chunk_count

        # Affichage périodique
        if epoch % (n_epochs // 100) == 0:
            print(f'[{time_since(start)} ({epoch} {epoch / n_epochs * 100:.1f}%) {loss_avg / epoch:.4f}]')

        # Sauvegarde du meilleur modèle
        if best_loss > (loss_avg / epoch):
            best_loss = loss_avg / epoch
            torch.save(decoder, join(model_dir, model_file))

# Évaluation du modèle
def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str).to(device)
    predicted = prime_str

    # Utilisation de la chaîne de départ
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char).to(device)

    return predicted

def evaluating(decoder, length):
    print("\n------------\n|   EVAL   |\n------------\n")
    try:
        while True:
            prime_str = input("Entrez une chaîne de départ : ")
            if len(prime_str) > 0:
                print(f"Résultat généré ({length} caractères) :")
                print(evaluate(decoder, prime_str=prime_str, predict_len=length))
            else:
                print("Veuillez entrer au moins un caractère.")
    except KeyboardInterrupt:
        print("\nFin de l'évaluation.")

# Exécution principale
if __name__ == '__main__':
    # Téléchargement du fichier
    file = download_file_with_pandas(file_url)
    file_len = len(file)

    # Initialisation du modèle
    decoder = RNN(n_characters, hidden_size, n_characters, n_layers).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Nom du fichier modèle
    model_file = f"rnn_{n_layers}_{hidden_size}.pt"
    if not path.exists(model_dir):
        makedirs(model_dir)

    # Mode entraînement
    print("Entraînement du modèle...")
    training(n_epochs, file)

    # Mode évaluation
    print("\nÉvaluation du modèle...")
    evaluating(decoder, length=100)
