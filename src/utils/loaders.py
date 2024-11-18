import os
import json
import torch
import segmentation_models_pytorch as smp
import cv2 as cv
from torchvision.transforms import v2
import torchvision.transforms
from PIL import Image

def load_smp_model(model_path:str, device='cpu') -> tuple((torch.nn.Module, torch.Tensor, torch.Tensor)):
    '''
    Given a path to an smp model, load it into memory and return a torch model ready to be used. Extracts encoder name and pre-processing parameters used by the encoder. Sets the model into eval mode.
    Args:
        path: path to the model weights
        device: which device the model, mean and std should be loaded to, Defaults to 'cpu'
    Returns:
        A triple of (model, mean, std), the mean and std should be used for normalising the image.
    '''
    config_file = os.path.join(model_path, 'config.json')
    with open(config_file) as f:
        d = json.load(f)
        encoder_name = d['encoder_name']
        arch_name    = d['_model_class'] 

    params = smp.encoders.get_preprocessing_params(encoder_name)
    mean   = torch.tensor(params['mean']).view(1, 3, 1, 1).to(device)
    std    = torch.tensor(params["std"]).view(1, 3, 1, 1).to(device)

    model = smp.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f'Loaded model architecture {arch_name} with {encoder_name} encoder on {device}')

    return model, mean, std

def load_img_to_tensor(img_path, std, mean, resize=None, device='cpu') -> torch.Tensor:
    '''
    Given an image path, model's std and mean, load the image to a torch tensor. std and mean should be specific to the architecture. In the case of an `smp_model` they should be easily attainable.
    Args:
        img_path: relative path to the location of the image
        std: standard deviation of the model
        mean: mean of the model
    Optional args:
        resize: Dimensions of the returned image. Defaults to the same dimensions as the image.
        device: Where to store the image tensor. Defaults to `cpu`.
    Returns:
        Torch tensor with the image, its shape is [1, C, H, W] that is already normalised.
        
    '''
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
