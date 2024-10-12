import torch 
from scipy import stats

salarios = torch.tensor([[50, 52, 55, 48, 52],  
                         [70, 68, 71, 69, 70],  
                         [85, 80, 82, 90, 85],  
                         [45, 42, 40, 43, 41],  
                         [60, 58, 59, 61, 60]]) 

print("Tensor de salarios por empresa:")
print(salarios)