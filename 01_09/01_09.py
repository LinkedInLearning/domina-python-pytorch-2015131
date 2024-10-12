import torch 
from scipy import stats

salarios = torch.tensor([[50, 52, 55, 48, 52],  
                         [70, 68, 71, 69, 70],  
                         [85, 80, 82, 90, 85],  
                         [45, 42, 40, 43, 41],  
                         [60, 58, 59, 61, 60]]) 

print("Tensor de salarios por empresa:")
print(salarios)

media_salarios = torch.mean(salarios.float(), dim=1)
print("Media de salarios por empresa:")
print(media_salarios)

mediana_salarios = torch.median(salarios, dim=1)[0]
print("Mediana de salarios por empresa:")
print(mediana_salarios)

moda_salarios = torch.tensor([stats.mode(empresa.numpy(), keepdims=True)[0][0] for empresa in salarios])
print("Moda de salarios por empresa:")
print(moda_salarios)