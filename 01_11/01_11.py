import torch

tensor_datos = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("Tensor original:")
print(tensor_datos)

slice_tensor = tensor_datos[0:2, 1:3]
print("Slice del tensor:") 
print(slice_tensor)
