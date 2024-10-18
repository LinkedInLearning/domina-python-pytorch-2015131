import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModelo(nn.Module):
    def __init__(self):
        super(CNNModelo, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_subset = Subset(train_dataset, range(0, 10000))
test_subset = Subset(test_dataset, range(0, 2000))

train_loader = DataLoader(dataset=train_subset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_subset, batch_size=500, shuffle=False, pin_memory=True)

modelo = CNNModelo().to(device)
criterio = nn.CrossEntropyLoss()
optimizador = optim.AdamW(modelo.parameters(), lr=0.001)

torch.backends.cudnn.benchmark = True

for epoch in range(7):
    modelo.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1} comenzando...")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizador.zero_grad()
        output = modelo(data)
        loss = criterio(output, target)
        loss.backward()
        optimizador.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} procesado, Pérdida acumulada: {running_loss / (batch_idx + 1)}")

    print(f'Epoch {epoch+1} completada, Pérdida promedio: {running_loss/len(train_loader)}')

modelo.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = modelo(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        if batch_idx % 1 == 0:
            print(f"Batch {batch_idx}/{len(test_loader)} evaluado")

print(f'Precisión en el conjunto de prueba: {100 * correct / total:.2f}%')

imagen_prueba = Image.open('numero_5.png').convert('L')
imagen_prueba = imagen_prueba.resize((28, 28))
transform_prueba = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
imagen_prueba = transform_prueba(imagen_prueba).unsqueeze(0).to(device)

modelo.eval()
with torch.no_grad():
    prediccion = modelo(imagen_prueba)
    _, etiqueta_predicha = torch.max(prediccion, 1)

print(f'El modelo predice que el dígito es: {etiqueta_predicha.item()}')
