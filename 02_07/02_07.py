import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

class ModeloEstudiante(nn.Module):
    def __init__(self):
        super(ModeloEstudiante, self).__init__()
        self.lineal = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.lineal(x))

modelo = ModeloEstudiante()

datos = np.array([
    [4.0, 0.9], [2.5, 0.5], [1.0, 0.3], [6.0, 0.8], [7.0, 1.0],
    [5.5, 0.6], [3.0, 0.4], [2.0, 0.2], [8.0, 0.9], [6.5, 1.0],
    [3.5, 0.7], [4.5, 0.6], [1.5, 0.3], [7.0, 0.9], [2.5, 0.4]
], dtype=np.float32)

etiquetas = np.array([[1], [0], [0], [1], [1], [1], [0], [0], [1], [1], [0], [1], [0], [1], [0]], dtype=np.float32)

datos_entrenamiento, datos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(datos, etiquetas, test_size=0.2)

entradas_entrenamiento = torch.tensor(datos_entrenamiento, dtype=torch.float32)
etiquetas_entrenamiento = torch.tensor(etiquetas_entrenamiento, dtype=torch.float32)
entradas_prueba = torch.tensor(datos_prueba, dtype=torch.float32)
etiquetas_prueba = torch.tensor(etiquetas_prueba, dtype=torch.float32)

criterio = nn.BCELoss()
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

epocas = 1000
for epoca in range(epocas):
    optimizador.zero_grad()
    salidas = modelo(entradas_entrenamiento)
    
    perdida = criterio(salidas, etiquetas_entrenamiento)
    
    if epoca % 100 == 0:
        print(f'Epoca {epoca}: Predicciones: {salidas.detach().numpy()}, Pérdida: {perdida.item()}')
    
    perdida.backward()
    optimizador.step()

with torch.no_grad():
    salidas_prueba = modelo(entradas_prueba)
    perdida_prueba = criterio(salidas_prueba, etiquetas_prueba)
    print(f'Pérdida en conjunto de prueba: {perdida_prueba.item()}')

    for i, salida in enumerate(salidas_prueba):
        print(f'Estudiante {i + 1}: Probabilidad de aprobar: {salida.item():.4f}, Etiqueta real: {etiquetas_prueba[i].item()}')

nuevo_estudiante = torch.tensor([[5.0, 0.8]], dtype=torch.float32)
prediccion = modelo(nuevo_estudiante)

print(f'Probabilidad de aprobar del nuevo estudiante): {prediccion.item()}')
