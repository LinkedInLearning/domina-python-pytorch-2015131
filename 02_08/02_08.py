import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ModeloCompra(nn.Module):
    def __init__(self):
        super(ModeloCompra, self).__init__()
        self.lineal1 = nn.Linear(2, 4)
        self.lineal2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.lineal1(x))
        return torch.sigmoid(self.lineal2(x))

modelo = ModeloCompra()

datos_entrenamiento = np.array([
    [5, 2.5],
    [2, 1.0],
    [10, 4.0],
    [3, 1.5]
], dtype=np.float32)

etiquetas_entrenamiento = np.array([[1], [0], [1], [0]], dtype=np.float32)

entradas_entrenamiento = torch.tensor(datos_entrenamiento, dtype=torch.float32)
etiquetas_entrenamiento = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)

criterio = nn.BCELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

epochs = 1500
for epoch in range(epochs):
    salidas = modelo(entradas_entrenamiento)
    perdida = criterio(salidas, etiquetas_entrenamiento)
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Pérdida: {perdida.item():.4f}')

with torch.no_grad():
    salidas_entrenamiento = modelo(entradas_entrenamiento)
    perdida_entrenamiento = criterio(salidas_entrenamiento, etiquetas_entrenamiento)
    print(f'\nPérdida final en conjunto de entrenamiento: {perdida_entrenamiento.item()}')

    for i, salida in enumerate(salidas_entrenamiento):
        print(f'Cliente {i + 1}: Probabilidad de compra: {salida.item():.4f}, Etiqueta real: {etiquetas_entrenamiento[i].item()}')

nuevo_cliente = torch.tensor([[8, 3.5]], dtype=torch.float32)
prediccion = modelo(nuevo_cliente)

print(f'\nPredicción para el nuevo cliente: Probabilidad de compra: {prediccion.item():.4f}')
