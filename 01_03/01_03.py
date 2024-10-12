
import torch

tensor = torch.tensor([150, 200, 300], dtype=torch.float32)
print("Tensor:", tensor)

print("Atributos clave del tensor")

print("Tipo de dato del tensor (dtype):", tensor.dtype)
print("Forma del tensor (shape):", tensor.shape)

print("Dispositivo del tensor (device):", tensor.device)

if torch.cuda.is_available():
    produccion_juguetes = tensor.to('cuda')
    print("Tensor movido a la GPU:", produccion_juguetes.device)
else:
    print("GPU no disponible. El tensor permanece en la CPU.")
