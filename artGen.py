import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import time
import math
import os

# Configurer l'appareil (CPU ou GPU)
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

# Paramètres de configuration
n_epochs = 200000
print_every = 10
hidden_size = 512
n_layers = 3
lr = 0.005

# Téléchargement avec Pandas
def download_file_with_pandas(url):
    print(f"Téléchargement du fichier depuis {url} avec Pandas...")
    try:
        # Lire tout le fichier texte en DataFrame
        data = pd.read_table(url, header=None, quoting=3)  # Lecture ligne par ligne
        text = "\n".join(data[0].tolist())  # Combiner toutes les lignes en une seule chaîne
        print("Fichier téléchargé et chargé avec succès.")
        return unidecode.unidecode(text)  # Nettoyage des caractères accentués
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        exit(1)

# Fonctions d'entraînement et d'évaluation
def random_chunk(file):
    start_index = random.randint(0, len(file) - chunk_len - 1)
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

def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str).to(device)
    predicted = prime_str

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

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

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

def training(n_epochs, file, chunk_count=10):
    start = time.time()
    loss_avg = 0
    best_loss = float("inf")
    print_every = n_epochs // 100

    for epoch in range(1, n_epochs + 1):
        losses = []
        for _ in range(chunk_count):
            loss = train(*random_training_set(file))
            losses.append(loss)

        loss_avg += sum(losses) / chunk_count

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch))

            if best_loss > (loss_avg / epoch):
                best_loss = loss_avg / epoch
                print(f"Meilleure perte améliorée : {best_loss:.4f}")

def evaluating(decoder, length):
    print("\nMode Évaluation :")
    try:
        while True:
            input1 = input("Entrez un début de texte : ")
            if len(input1) > 0:
                print(evaluate(decoder, prime_str=input1, predict_len=length, temperature=0.8))
            else:
                print("Entrée invalide.")
    except KeyboardInterrupt:
        print("\nÉvaluation terminée.")

# Exécution principale
if __name__ == '__main__':
    file_url = "https://olivier-fabre.com/passwordgenius/shakespeare2.txt"
    file = download_file_with_pandas(file_url)
    file_len = len(file)

    print("Longueur du fichier :", file_len)

    decoder = RNN(n_characters, hidden_size, n_characters, n_layers).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Entraînement
    decoder.train()
    training(n_epochs, file)

    # Évaluation
    decoder.eval()
    evaluating(decoder, length=100)
import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import time
import math
import os

# Configurer l'appareil (CPU ou GPU)
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

# Paramètres de configuration
n_epochs = 200000
print_every = 10
hidden_size = 512
n_layers = 3
lr = 0.005

# Téléchargement avec Pandas
def download_file_with_pandas(url):
    print(f"Téléchargement du fichier depuis {url} avec Pandas...")
    try:
        # Lire tout le fichier texte en DataFrame
        data = pd.read_table(url, header=None, quoting=3)  # Lecture ligne par ligne
        text = "\n".join(data[0].tolist())  # Combiner toutes les lignes en une seule chaîne
        print("Fichier téléchargé et chargé avec succès.")
        return unidecode.unidecode(text)  # Nettoyage des caractères accentués
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        exit(1)

# Fonctions d'entraînement et d'évaluation
def random_chunk(file):
    start_index = random.randint(0, len(file) - chunk_len - 1)
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

def evaluate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str).to(device)
    predicted = prime_str

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

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

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

def training(n_epochs, file, chunk_count=10):
    start = time.time()
    loss_avg = 0
    best_loss = float("inf")
    print_every = n_epochs // 100

    for epoch in range(1, n_epochs + 1):
        losses = []
        for _ in range(chunk_count):
            loss = train(*random_training_set(file))
            losses.append(loss)

        loss_avg += sum(losses) / chunk_count

        if epoch % print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss_avg / epoch))

            if best_loss > (loss_avg / epoch):
                best_loss = loss_avg / epoch
                print(f"Meilleure perte améliorée : {best_loss:.4f}")

def evaluating(decoder, length):
    print("\nMode Évaluation :")
    try:
        while True:
            input1 = input("Entrez un début de texte : ")
            if len(input1) > 0:
                print(evaluate(decoder, prime_str=input1, predict_len=length, temperature=0.8))
            else:
                print("Entrée invalide.")
    except KeyboardInterrupt:
        print("\nÉvaluation terminée.")

# Exécution principale
if __name__ == '__main__':
    file_url = "https://olivier-fabre.com/passwordgenius/shakespeare2.txt"
    file = download_file_with_pandas(file_url)
    file_len = len(file)

    print("Longueur du fichier :", file_len)

    decoder = RNN(n_characters, hidden_size, n_characters, n_layers).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Entraînement
    decoder.train()
    training(n_epochs, file)

    # Évaluation
    decoder.eval()
    evaluating(decoder, length=100)