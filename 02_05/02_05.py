import torch
import torch.nn as nn
import torch.optim as optim

modelo = nn.Sequential(
    nn.Linear(10, 5),  
    nn.ReLU(),        
    nn.Linear(5, 2)  
)

valor_entrada = torch.randn(4, 10) 
etiquetas = torch.tensor([0, 1, 0, 1]) 

print("Datos de entrada:")
print(valor_entrada)

print("Etiquetas:")
print(etiquetas)

