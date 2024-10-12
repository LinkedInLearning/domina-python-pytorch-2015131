import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[10], [20], [30], [40], [50]], dtype=torch.float32) / 50.0
y = torch.tensor([[15], [25], [35], [45], [55]], dtype=torch.float32) / 55.0

modelo = nn.Linear(1, 1)

nn.init.xavier_uniform_(modelo.weight)
nn.init.constant_(modelo.bias, 0.0)

perdida_mseloss = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.01)

for epoch in range(5000):
    y_pred = modelo(x)

    perdida = perdida_mseloss(y_pred, y)
    
    print(f"Epoch {epoch}, Pérdida: {perdida.item()}")
    
    optimizador.zero_grad()
    perdida.backward()
    optimizador.step()

with torch.no_grad():
    nueva_inversion = torch.tensor([[60]], dtype=torch.float32) / 50.0 
    prediccion = modelo(nueva_inversion)
    prediccion_final = prediccion.item() * 55.0
    print(f"Predicción para una inversión de 60 mil: {prediccion_final}")