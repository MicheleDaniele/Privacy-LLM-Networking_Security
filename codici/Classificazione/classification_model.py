# === SEZIONE 1: Importazioni e Configurazione Iniziale ===
import torch
import torch.nn as nn         # Moduli per le reti neurali
import torch.optim as optim   # Ottimizzatori (Adam)
import torchvision            # Dataset e trasformazioni
import torchvision.transforms as transforms  # Preprocessing immagini
import matplotlib.pyplot as plt  # Visualizzazione risultati

# Configurazione iperparametri
batch_size = 32  # Dimensione batch ridotta per limitare uso memoria GPU
num_epochs = 100  # Numero di passate complete sul dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponibile

# Spiegazione:
# batch_size=32 riduce il carico di memoria rispetto a 128, cruciale per evitare l'errore "OutOfMemory"
# device determina automaticamente se usare GPU (CUDA) o CPU, ottimizzando le performance

# === SEZIONE 2: Preparazione Dati ===
# Trasformazioni per i dati di training (Nessuna augmentation per favorire overfitting)
transform_train_overfit = transforms.Compose([
    transforms.ToTensor(),  # Converte immagini in tensori e normalizza in [0,1]
])

# Trasformazioni per i dati di test (uguali al training)
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# Caricamento dataset CIFAR100
trainset_overfit = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform_train_overfit
)

testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# Creazione DataLoader per il training
trainloader_overfit = torch.utils.data.DataLoader(
    trainset_overfit,
    batch_size=batch_size,
    shuffle=True,   # Mescola i dati ad ogni epoca
    num_workers=2   # Thread paralleli per il caricamento
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,  # Non necessario per il test
    num_workers=2
)

# Spiegazione:
# num_workers=2 accelera il caricamento dati usando processi paralleli
# shuffle=True previene l'apprendimento di pattern legati all'ordinamento dei dati

# === SEZIONE 3: Definizione del Modello OverfitNet ===

# Architettura chiave per l'overfitting:
# Nessuna regolarizzazione (manca dropout/L2 regularization)
# Capacity elevata: 3 layer convoluzionali con canali crescenti
# Dimensionalità hidden layer: 512 unità nello strato fully-connected
# Assenza di batch normalization che aiuterebbe la generalizzazione

class OverfitNet(nn.Module):
    def __init__(self):
        super(OverfitNet, self).__init__()
        # Blocco feature extraction
        self.features = nn.Sequential(
            # Layer 1: 3 canali input (RGB), 64 filtri 3x3, padding per mantenere dimensione
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),  # Funzione di attivazione

            # Max pooling riduce dimensione spaziale a 16x16
            nn.MaxPool2d(2),

            # Layer 2: 64 -> 128 canali
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            # Layer 3: 128 -> 256 canali
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4

            nn.Flatten()  # Appiattisce l'output per i layer fully-connected
        )

        # Blocco classificazione
        self.classifier = nn.Sequential(
            # 256 canali * 4x4 = 4096 features
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            # Layer finale: 512 features -> 100 classi (CIFAR100)
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.features(x)  # Passaggio attraverso i layer convoluzionali
        x = self.classifier(x)  # Passaggio attraverso i layer fully-connected
        return x

# === SEZIONE 4: Inizializzazione Modello e Ottimizzatore ===

# Scelta dell'ottimizzatore:
# Adam combina i vantaggi di RMSProp e Momentum
# Learning rate 0.001 è un valore standard per dataset di medie dimensioni

model = OverfitNet().to(device)  # Sposta il modello sulla GPU
criterion = nn.CrossEntropyLoss()  # Loss function per classificazione
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Ottimizzatore con learning rate

# Liste per tracciare le accuratezze
train_acc_list = []
test_acc_list = []

# === SEZIONE 5: Ciclo di Addestramento ===
for epoch in range(num_epochs):
    # Fase di training

    # Punti cruciali del training loop:
    # optimizer.zero_grad(): Resetta i gradienti per evitare accumuli
    # loss.backward(): Calcola gradienti tramite autograd
    # optimizer.step(): Aggiorna i pesi usando gli ottimizzatori

    model.train() # Imposta il modello in modalità training
    correct_train, total_train = 0, 0
    for inputs, labels in trainloader_overfit:
        # Sposta dati su GPU
        inputs, labels = inputs.to(device), labels.to(device)
        # Azzera i gradienti accumulati
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calcola loss
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward() # Calcola gradienti
        # Aggiorna pesi
        optimizer.step()
        # Calcolo accuratezza
        _, predicted = outputs.max(1) # Indici delle classi predette
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
    train_acc = correct_train / total_train
    train_acc_list.append(train_acc)

    # Fase di valutazione

    # Modalità evaluation:
    # model.eval(): Disabilita layer come dropout
    # torch.no_grad(): Riduce l'uso di memoria non tenendo traccia dei gradienti

    model.eval() # Disabilita dropout/batchnorm
    correct_test, total_test = 0, 0
    with torch.no_grad(): #  Disabilita calcolo gradienti per risparmiare memoria
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    test_acc = correct_test / total_test
    test_acc_list.append(test_acc)

    # Stampa risultati epoca
    print(f"Epoch {epoch+1}/{num_epochs} - Train acc: {train_acc:.4f} - Test acc: {test_acc:.4f}")

    # Pulizia memoria CUDA
    torch.cuda.empty_cache()

# === SEZIONE 6: Visualizzazione Risultati ===
# Plot accuracy
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_acc_list, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('OverfitNet Accuracy on CIFAR100')
plt.legend()
plt.grid(True)
plt.show()

# Interpretazione del plot:
#   Overfitting evidente quando:
#     Accuratezza training ↗ rapidamente
#     Accuratezza test ↗ lentamente o plateau
#     Divario crescente tra le due curve
#   Esempio tipico: Training accuracy >80% con test accuracy <40%
# Analisi prestazioni attese:
#   Prime 5 epoche: Rapido aumento accuratezza training
#   Epoche 5-15: Training accuracy continua a salire, test accuracy stabilizza
#   Ultime epoche: Divario training-test >30%, chiaro segnale di overfitting
# Elementi che favoriscono l'overfitting:
#   Architettura profonda senza regolarizzazione
#   Assenza di data augmentation
#   Learning rate costante
#   Ottimizzatore Adam che converge rapidamente
#   Elevata capacità del modello (256 canali nell'ultimo conv)

# Trasformazioni per il training che aiutano la generalizzazione (data augmentation)
transform_train_no_overfit = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

# Dataset di training con data augmentation per il modello no overfit
trainset_no_overfit = torchvision.datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform_train_no_overfit
)

trainloader_no_overfit = torch.utils.data.DataLoader(
    trainset_no_overfit,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

class NoOverfitNet(nn.Module):
    def __init__(self):
        super(NoOverfitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Regolarizzazione
            nn.Linear(128*4*4, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

no_overfit_model = NoOverfitNet().to(device)
criterion_no = nn.CrossEntropyLoss()
optimizer_no = optim.Adam(no_overfit_model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regolarizzazione

train_acc_list_no = []
test_acc_list_no = []

for epoch in range(num_epochs):
    # Training
    no_overfit_model.train()
    correct_train, total_train = 0, 0
    for inputs, labels in trainloader_no_overfit:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_no.zero_grad()
        outputs = no_overfit_model(inputs)
        loss = criterion_no(outputs, labels)
        loss.backward()
        optimizer_no.step()
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
    train_acc = correct_train / total_train
    train_acc_list_no.append(train_acc)

    # Valutazione
    no_overfit_model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = no_overfit_model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    test_acc = correct_test / total_test
    test_acc_list_no.append(test_acc)

    print(f"[NO OVERFIT] Epoch {epoch+1}/{num_epochs} - Train acc: {train_acc:.4f} - Test acc: {test_acc:.4f}")

    torch.cuda.empty_cache()

plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs+1), train_acc_list, label='OverfitNet - Train')
plt.plot(range(1, num_epochs+1), test_acc_list, label='OverfitNet - Test')
plt.plot(range(1, num_epochs+1), train_acc_list_no, label='NoOverfitNet - Train')
plt.plot(range(1, num_epochs+1), test_acc_list_no, label='NoOverfitNet - Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Confronto Overfitting vs No Overfitting su CIFAR100')
plt.legend()
plt.grid(True)
plt.show()