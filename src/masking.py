import os
import sys
import cv2 as cv
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from models import smp_model
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms
import matplotlib.pyplot as plt


def sirotenko_model():
    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CURRENT_DIR = os.path.dirname(os.path.abspath('__file__'))
    parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
    sys.path.append(parent_dir)

    checkpoint_path = "../weights/circle_segmentation/circle_segmentation_30e_sched"
    circle_seg_model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).to(device)
    return circle_seg_model

def my_model():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_name = './models/logs/unet_cylinder_segmentation.py-'
    model_name = 'unet_resnet34_bs2_ep3_r1024_lr0.0002-2024-10-03_165301'
    weights_path = f'{base_name}{model_name}/stored_model'

    encoder_name = weights_path.split('/')[-2].split('_')[3]
    params = smp.encoders.get_preprocessing_params(encoder_name)
    global mean, std
    mean = torch.tensor(params['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1).to(device)


    model = smp.from_pretrained(weights_path).to(device)
    model.eval()

    return model


def imshow(title= None, **images):
    """Displays images in one row"""
    n = len(images)
    plt.figure(figsize=(n*4,5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name=='image':
            plt.imshow(image.permute(1, 2, 0))
        else:
            plt.imshow(image)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('result' if title is None else title)

    plt.show()
    plt.pause(0.1)

# Works mostly OK, might need some adjustments with parameters.
def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8)
    small_size = (1000, 1000)

    shrunk_mask = cv.resize(mask, small_size, interpolation=cv.INTER_AREA)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20,20))
    morphed_mask = cv.morphologyEx(shrunk_mask, cv.MORPH_CLOSE, kernel)

    og_size = (mask.shape[1], mask.shape[0])
    filled_mask = cv.resize(morphed_mask, og_size, interpolation=cv.INTER_LINEAR)

    return filled_mask

def get_circle(model, img_path, output_dir, new_size=None, use_mine=True):
    if use_mine:
        my_segmentation(model, img_path, output_dir, new_size)
    else:
        sirotenko_segmentation(model, img_path, output_dir, new_size)

def my_segmentation(model, img_path, output_dir, new_size=None):
    if model is None:
        model = my_model()

    transformation = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize((1024,1024))
        ])

    img_PIL = Image.open(img_path).convert('RGB')
    og_dims = img_PIL.size #get original dimensions of the image

    img_tensor = torchvision.transforms.PILToTensor()(img_PIL)
    img_tensor = transformation(img_tensor).to(device)
    img_tensor = img_tensor[None]
    img_tensor = (img_tensor - mean) / std

    with torch.no_grad():
        logits = model(img_tensor)

    prob_mask = logits.sigmoid()
    temp_mask = prob_mask.squeeze().cpu().numpy()
    temp_mask = (temp_mask * 255).astype(np.uint8)
    temp_mask = cv.applyColorMap(temp_mask, cv.COLORMAP_JET)
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    output_dir = f'{output_dir}-cropped-rawmask.jpg'
    cv.imwrite(output_dir,temp_mask)

    prob_mask = (prob_mask > 0.5).type(torch.uint8)
    prob_mask = prob_mask.squeeze().cpu().numpy()

    img_np = np.array(img_PIL)

    prob_mask = cv.resize(prob_mask, og_dims, interpolation=cv.INTER_CUBIC)
    img_masked = cv.bitwise_and(img_np, img_np, mask=prob_mask)

    # Extract nonzero coordinates
    x,y = np.nonzero(prob_mask)
    if x.any() and y.any():
        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        # crop the image
        cropped = img_masked[xl : xr+1, yl : yr+1]
    else:
        cropped = prob_mask # Empty mask in case of failure

    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    output_dir = f'{output_dir}-cropped.jpg'
    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2RGB)
    cv.imwrite(output_dir, cropped)


# Segmentation of the cylinder via the use of Sirotenko's Segformer model
def sirotenko_segmentation(model, img_path, output_dir, new_size=None):
    if model is None:
        model = sirotenko_model()

    img = cv.imread(img_path)
    img_processor = SegformerImageProcessor(reduce_labels=True)
    pixel_values = img_processor(img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    predicted_seg_map = img_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.shape[0:2]]
    )[0]

    predicted_seg_map = predicted_seg_map.cpu().numpy()

    mask = predicted_seg_map.astype(np.uint8)

    #mask = fill_holes(mask)

    img_masked = cv.bitwise_and(img, img, mask=mask)

    # Extract nonzero coordinates
    x,y = np.nonzero(mask)
    if x.any() and y.any():
        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        # crop the image
        cropped = img_masked[xl : xr+1, yl : yr+1]
    else:
        cropped = mask # Empty mask in case of failure

    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    output_dir = f'{output_dir}-cropped.jpg'
    cv.imwrite(output_dir, cropped)

def load_images(directory_path):
    output_dir = './imgs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(directory_path):
        if file.endswith('.jpg'):
            print(f'\rprocessing file {file}', end='', flush=False)
            img_path = os.path.join(directory_path, file)

            if os.path.isfile(img_path):
                get_circle(circle_seg_model, img_path, output_dir)
    print('')
