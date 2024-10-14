import torch
from torch.utils.data import DataLoader, TensorDataset

datos = torch.randn(100, 3) 
print(datos)

etiquetas = torch.randint(0, 2, (100,))
print(etiquetas)

dataset = TensorDataset(datos, etiquetas)

data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

for datos_lote, etiquetas_lote in data_loader:
    print(f"Datos: \n {datos_lote}")
    print(f"Etiqueta: \n {etiquetas_lote}")
    print("\n")