import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

datos = np.array([
    [50, 1, 0], [300, 0, 0], [150, 1, 1], [20, 0, 0], [400, 1, 0],
    [250, 0, 1], [60, 1, 0], [500, 0, 0], [75, 0, 0], [30, 0, 1],
    [200, 1, 1], [125, 0, 0], [90, 0, 1], [45, 1, 0], [180, 1, 1]
], dtype=np.float32)

etiquetas = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1]], dtype=np.float32)
