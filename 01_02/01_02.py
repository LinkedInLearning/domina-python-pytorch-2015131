import torch

tensor_1d = torch.tensor([5000, 7000, 10000])
print("Tensor en una dimisión:", tensor_1d)

tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

print("Tensor 2D:")
print(tensor_2d)

tensor_3d = torch.randn(2, 3, 3)
print("Tensor 3D Aleatorio:")
print(tensor_3d)

