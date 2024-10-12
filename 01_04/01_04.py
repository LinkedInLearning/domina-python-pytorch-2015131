import torch

tensor1 = torch.tensor([100, 150, 120]) 
tensor2 = torch.tensor([80, 60, 90])    
tensor3 = torch.tensor([50, 70, 60])    

print("Tensor 1:", tensor1)
print("Tensor 2:", tensor2)
print("Tensor 3:", tensor3)

primeros_elementos = tensor1[:2]
print("Primeros elementos:", primeros_elementos)

elemento1 = tensor2[1]
print("Elemente 1:", elemento1)

inventario_total = torch.cat((tensor1, tensor2, tensor2), dim=0)
print("Tensor concatenando:", inventario_total)