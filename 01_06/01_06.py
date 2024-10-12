import torch

tensor_a = torch.tensor([[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9]], dtype=torch.float32)

tensor_b = torch.tensor([[9, 8, 7], 
                         [6, 5, 4], 
                         [3, 2, 1]], dtype=torch.float32)


suma_tensores = tensor_a + tensor_b
print("Suma de Tensor A y Tensor B:")
print(suma_tensores)

multiplicacion_tensores = tensor_a * tensor_b
print("Multiplicación de Tensor A y Tensor B:")
print(multiplicacion_tensores)

resta_tensores = tensor_a - tensor_b
print("Resta de Tensor A y Tensor B:")
print(resta_tensores)
