import torch 

imagen_3d = torch.tensor([[[0, 1, 2], 
                           [3, 4, 5], 
                           [6, 7, 8]],

                          [[9, 10, 11], 
                           [12, 13, 14], 
                           [15, 16, 17]],

                          [[18, 19, 20], 
                           [21, 22, 23], 
                           [24, 25, 26]]], dtype=torch.float32)

print("Tensor 3D:")
print(imagen_3d)

seccion_2d = imagen_3d[1, :, :]
print("Sección 2D seleccionada (capa 2 del tensor):")
print(seccion_2d)

valor_especifico = imagen_3d[2, 2, 2]
print("Valor específico:")
print(valor_especifico)

rango_subtensor = imagen_3d[0, :, 0:2] 
print("Primera capa, primeras dos columnas):")
print(rango_subtensor)
