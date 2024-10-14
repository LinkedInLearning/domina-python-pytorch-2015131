import torch
from torch.utils.data import DataLoader, TensorDataset

datos = torch.randn(100, 3) 
print(datos)

etiquetas = torch.randint(0, 2, (100,))
print(etiquetas)