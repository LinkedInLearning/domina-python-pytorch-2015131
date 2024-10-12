import torch
import numpy as np

tensor_pytorch = torch.tensor([[3, 5], [7, 9]])
print("Tensor de PyTorch:")
print(tensor_pytorch)

array_numpy = tensor_pytorch.numpy()
print("Array convertido a NumPy:")
print(array_numpy)

array_numpy_2 = np.array([[1, 2], [3, 4]])
tensor_pytorch_2 = torch.from_numpy(array_numpy_2)
print("Array de NumPy convertido a tensor:")
print(tensor_pytorch_2)