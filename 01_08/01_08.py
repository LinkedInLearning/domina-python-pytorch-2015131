import torch

temperaturas = torch.tensor([[15.5, 20.3, 18.7, 22.1, 19.6, 21.0, 23.4],  
                             [13.4, 17.6, 16.8, 20.0, 18.5, 19.7, 22.0],  
                             [16.2, 18.5, 17.1, 19.8, 21.2, 22.5, 24.1], 
                             [14.8, 19.9, 20.5, 21.0, 20.3, 22.1, 25.0]])

print("Tensor de temperaturas por ciudad durante 7 días:")
print(temperaturas)

max_temp = torch.max(temperaturas)
print("Temperatura máxima:", max_temp)

min_temp = torch.min(temperaturas)
print("Temperatura mínima:", min_temp)

max_temp_ciudad = torch.max(temperaturas, dim=1)[0] 
print("Temperaturas máximas por ciudad:", max_temp_ciudad)

min_temp_ciudad = torch.min(temperaturas, dim=1)[0]
print("Temperaturas mínimas por ciudad:", min_temp_ciudad)

print("Temperatura máxima entre todas las ciudades y días:", max_temp.item())
print("Temperatura mínima entre todas las ciudades y días:", min_temp.item())
