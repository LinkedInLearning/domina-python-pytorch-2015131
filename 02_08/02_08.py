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
