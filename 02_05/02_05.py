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

funcion_perdida = nn.CrossEntropyLoss()
optimizado = optim.Adam(modelo.parameters(), lr=0.001)

predicciones = modelo(valor_entrada)
print("Salidas del modelo (predicciones):")
print(predicciones)

perdida = funcion_perdida(predicciones, etiquetas)
print("Pérdida:")
print(perdida.item())  

optimizado.zero_grad()  
perdida.backward()        
optimizado.step()

