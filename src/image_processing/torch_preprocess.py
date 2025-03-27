import torch
import cv2 as cv
from torchvision.transforms import v2
import torchvision.transforms
from PIL import Image

def load_img_to_tensor(img_path, std, mean, resize=None, device='cpu') -> torch.Tensor:
    img_PIL = Image.open(img_path).convert('RGB')

    if resize is None:
        w,h = img_PIL.size
    else:
        w,h = resize, resize

    transformation = v2.Compose([
        v2.Resize((w,h)),
        v2.ToDtype(torch.float32),
    ])

    image = torchvision.transforms.PILToTensor()(img_PIL)

    image = transformation(image).to(device)
    image = image[None] # Adding 'batch' dimension
    image = (image - mean) / std # Normalise image

    return image
