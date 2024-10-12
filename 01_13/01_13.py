import torch

x = torch.tensor(3.0, requires_grad=True)

#Función simple f(x) = x^2
y = x ** 2

y.backward()

print("Gradiente de y con respecto a x:", x.grad)
