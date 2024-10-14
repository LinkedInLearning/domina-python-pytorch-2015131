from torchvision import transforms
from PIL import Image

imagen = Image.open('planta.png')

transformacion = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

imagen_transformada = transformacion(imagen)
print(imagen_transformada)
