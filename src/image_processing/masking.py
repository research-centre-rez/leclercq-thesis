import os
import sys
import cv2 as cv
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from PIL import Image
import torchvision.transforms

def sirotenko_model():
    """
    Initialises Sirotenko's cylinder segmentation model
    """

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CURRENT_DIR = os.path.dirname(os.path.abspath('__file__'))
    parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
    sys.path.append(parent_dir)

    checkpoint_path = "../weights/circle_segmentation/circle_segmentation_30e_sched"
    circle_seg_model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).to(device)
    return circle_seg_model


def my_smp_model(weights_path, encoder_name):
    """
    Initialises Leclercq's smp cylinder segmentation model.
    :weights_path: path to the desired model
    :encoder_name: name of the encoder, this is used to extract mean, std that is specific to the model
    """

    global device, mean, std
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = smp.encoders.get_preprocessing_params(encoder_name)

    mean = torch.tensor(params['mean']).view(1, 3, 1, 1).to(device)
    std = torch.tensor(params["std"]).view(1, 3, 1, 1).to(device)

    model = smp.from_pretrained(weights_path).to(device)
    model.eval()

    return model

def pad_to_size(img, new_size):
    """
    Pads image so that it is of the exactly needed size
    """
    h,w = img.shape[:2]
    pad_h = max(0, (new_size - h) // 2)
    pad_w = max(0, (new_size - w) // 2)


    inc_h = new_size - (h + pad_h * 2)
    inc_w = new_size - (w + pad_w * 2)

    padded_img = cv.copyMakeBorder(img,pad_h + inc_h, pad_h, pad_w + inc_w, pad_w,
                                   borderType=cv.BORDER_CONSTANT,
                                   value=[0,0,0])
    return padded_img

def resize_or_pad(img, new_size):
    """
    Decides whether an image should be resized or padded
    :img: image to be padded
    :new_size: size to be padded to
    """

    h,w = img.shape[:2]

    if max(h,w) > new_size:
        scale = new_size / max(h,w)
        img = cv.resize(img, (int(w * scale), int(h*scale)), interpolation=cv.INTER_AREA)

    if h < new_size or w < new_size:
        img = pad_to_size(img, new_size)

    return img

def fill_holes(mask, shrunk_img_size=(1000,1000), kernel_size=(20,20)):
    """
    Fills holes that get created inside a mask. Currently this is not being used.
    """
    mask = (mask > 0).astype(np.uint8)
    small_size = shrunk_img_size

    shrunk_mask = cv.resize(mask, small_size, interpolation=cv.INTER_AREA)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    morphed_mask = cv.morphologyEx(shrunk_mask, cv.MORPH_CLOSE, kernel)

    og_size = (mask.shape[1], mask.shape[0])
    filled_mask = cv.resize(morphed_mask, og_size, interpolation=cv.INTER_LINEAR)

    return filled_mask

def crop_img(masked_image, mask):
    """
    Crops an image based on non-zero coordinates in the mask
    """
    # Extract nonzero coordinates
    x,y = np.nonzero(mask)
    if x.any() and y.any():
        xl, xr = x.min(), x.max()
        yl, yr = y.min(), y.max()

        # crop the image via index slicing
        return masked_image[xl : xr+1, yl : yr+1]

    return mask


def my_segmentation(model, img_path, output_dir, new_size=2200) -> None:
    """
    Given a model, perform image cropping on it

    :model: PyTorch model that is used for inference
    :img_path: Path to the image
    :output_dir: Target destination
    :new_size: Specifies the size of the cropped image
    """

    def convert_img_to_tensor(img:Image) -> torch.Tensor:
        transformation = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32),
                v2.Resize((1024,1024))
            ])
        img_tensor = torchvision.transforms.PILToTensor()(img)
        img_tensor = transformation(img_tensor).to(device)
        img_tensor = img_tensor[None] # Add 'batch' dimension
        img_tensor = (img_tensor - mean) / std # Normalise image
        return img_tensor

    # Stores predicted mask as a heatmap
    def save_heatmap(prob_mask) -> None:
        temp_mask = prob_mask.squeeze().cpu().numpy()
        temp_mask = (temp_mask * 255).astype(np.uint8)
        temp_mask = cv.applyColorMap(temp_mask, cv.COLORMAP_JET)
        temp_mask = cv.resize(temp_mask, og_dims, interpolation=cv.INTER_CUBIC)
        tmp_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
        tmp_output_dir = f'{tmp_output_dir}-cropped-rawmask.jpg'
        cv.imwrite(tmp_output_dir,temp_mask)

    img_PIL    = Image.open(img_path).convert('RGB')
    img_np     = np.array(img_PIL)
    img_tensor = convert_img_to_tensor(img_PIL)

    og_dims = img_PIL.size # save the original dimensions of the image

    with torch.no_grad():
        logits = model(img_tensor)

    prob_mask = logits.sigmoid()
    save_heatmap(prob_mask)

    # Thresholding + sending to CPU
    prob_mask = (prob_mask > 0.7).type(torch.uint8)
    prob_mask = prob_mask.squeeze().cpu().numpy()

    # Resize to original size and fill out remaining holes
    prob_mask = cv.resize(prob_mask, og_dims, interpolation=cv.INTER_CUBIC)

    # Mask image
    img_masked = cv.bitwise_and(img_np, img_np, mask=prob_mask)
    cropped    = crop_img(img_masked, prob_mask)
    cropped    = resize_or_pad(cropped, new_size)
    cropped    = cv.cvtColor(cropped, cv.COLOR_BGR2RGB) #Set to correct colour channels

    # Saving the newly cropped image
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    output_dir = f'{output_dir}-cropped.jpg'
    cv.imwrite(output_dir, cropped)

def sirotenko_segmentation(model, img_path, output_dir, new_size=None) -> None:
    """
    Segmentation done via the use of Sirotenko's SegFormer model
    """
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

    img_masked = cv.bitwise_and(img, img, mask=mask)
    cropped    = crop_img(img_masked, mask)

    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    output_dir = f'{output_dir}-cropped.jpg'
    cv.imwrite(output_dir, cropped)
