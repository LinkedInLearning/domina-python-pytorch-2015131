import torch

tensor_1 = torch.tensor([[1, 2], [3, 4]])
tensor_2 = torch.tensor([[5, 6], [7, 8]])

multiplicacion_elemento = tensor_1 * tensor_2
print("Multiplicación elemento por elemento:") 
print(multiplicacion_elemento)

multiplicacion_matrices = torch.matmul(tensor_1, tensor_2)
print("Multiplicación de matrices:")
print(multiplicacion_matrices)
