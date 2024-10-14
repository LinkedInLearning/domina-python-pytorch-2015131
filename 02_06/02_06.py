import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.lineal = nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.lineal(x))
    
datos = np.array([
    [50, 1, 0], [300, 0, 0], [150, 1, 1], [20, 0, 0], [400, 1, 0],
    [250, 0, 1], [60, 1, 0], [500, 0, 0], [75, 0, 0], [30, 0, 1],
    [200, 1, 1], [125, 0, 0], [90, 0, 1], [45, 1, 0], [180, 1, 1]
], dtype=np.float32)

etiquetas = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1]], dtype=np.float32)

modelo = Perceptron()

criterio = nn.BCELoss()
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

datos[:, 0] = datos[:, 0] / np.max(datos[:, 0])
datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2)

entradas_entrenamiento = torch.tensor(datos_entrenamiento, dtype=torch.float32)
etiquetas_entrenamiento = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)
entradas_prueba = torch.tensor(datos_prueba, dtype=torch.float32)
etiquetas_prueba = torch.tensor(etiquetas_prueba, dtype=torch.float32)

epocas = 1000
for epoca in range(epocas):
    optimizador.zero_grad()
    salidas = modelo(entradas_entrenamiento)
    perdida = criterio(salidas, etiquetas_entrenamiento)
    perdida.backward()
    optimizador.step()

    if epoca % 100 == 0:
        print(f'Epoca {epoca}, Pérdida: {perdida.item()}')

with torch.no_grad():
    salidas_prueba = modelo(entradas_prueba)
    perdida_prueba = criterio(salidas_prueba, etiquetas_prueba)
    print(f'\nPérdida en conjunto de prueba: {perdida_prueba.item()}')

    for i, salida in enumerate(salidas_prueba):
        print(f'Correo {i + 1}: Probabilidad de ser spam: {salida.item():.4f}, Etiqueta real: {etiquetas_prueba[i].item()}')


nuevo_correo = torch.tensor([[0.2, 1, 1]], dtype=torch.float32)
prediccion = modelo(nuevo_correo)
print(f'Predicción: Probabilidad de spam del nuevo correo: {prediccion.item()}')
