import os
import sys
import cv2 as cv
import numpy as np

import torch

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# CURRENT_DIR = os.path.dirname(os.path.abspath('__file__'))
# parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
# sys.path.append(parent_dir)
#
# checkpoint_path = "../weights/circle_segmentation/circle_segmentation_30e_sched"
# circle_seg_model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).to(device)

def init_model():
    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CURRENT_DIR = os.path.dirname(os.path.abspath('__file__'))
    parent_dir = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
    sys.path.append(parent_dir)

    checkpoint_path = "../weights/circle_segmentation/circle_segmentation_30e_sched"
    circle_seg_model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path).to(device)
    return circle_seg_model


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

# Segmentation of the cylinder via the use of Sirotenko's Segformer model
def get_circle(model, img_path, output_dir, new_size=None):
    if model is None:
        model = init_model()

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

    mask = fill_holes(mask)

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
    #print(f"\rSaving to:{output_dir}", end='', flush=True)
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

#load_images('../data/dev/temp-test')
