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

