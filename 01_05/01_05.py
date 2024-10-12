import torch

imagen_tensor = torch.tensor([[0, 128, 255], 
                              [64, 128, 192], 
                              [32, 64, 96]], dtype=torch.float32)

print("Tensor bidimensional (imagen simulada):")
print(imagen_tensor)
