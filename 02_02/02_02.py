import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

transformaciones = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset_mnist = datasets.MNIST(root='./datos', train=True, transform=transformaciones, download=True)

cargar_datos = DataLoader(dataset_mnist, batch_size=32, shuffle=True)

for lote, (dato, etiqueta) in enumerate(cargar_datos):
    print(f"Lote {lote + 1} -> Datos: {dato.shape}, Objetivo: {etiqueta.shape}")
    if lote == 5:
        break
