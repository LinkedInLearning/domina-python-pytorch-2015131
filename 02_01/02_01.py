import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[10], [20], [30], [40], [50]], dtype=torch.float32) / 50.0
y = torch.tensor([[15], [25], [35], [45], [55]], dtype=torch.float32) / 55.0

modelo = nn.Linear(1, 1)