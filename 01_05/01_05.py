import torch

imagen_tensor = torch.tensor([[0, 128, 255], 
                              [64, 128, 192], 
                              [32, 64, 96]], dtype=torch.float32)

print("Tensor bidimensional (imagen simulada):")
print(imagen_tensor)

imagen_transformada = imagen_tensor * 2
print("Tensor transformado (imagen después de multiplicar por 2):")
print(imagen_transformada)

pixel_valor = imagen_tensor[1, 1]
print(f"Valor del píxel en la posición (1,1): {pixel_valor}")

suma_total = imagen_tensor.sum()
print(f"\nSuma total de los valores de los píxeles: {suma_total}")
