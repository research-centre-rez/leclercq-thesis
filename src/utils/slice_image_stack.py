import sys
import os
import numpy as np
import argparse

import logging

from image_registration import matrix_processing
from utils import visualisers
from utils import pprint
from utils import filename_builder

def parse_args():
    parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Required arguments
    required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')
    optional.add_argument('-fs', '--frame_step', type=int, help='What frame step do you want to use?')
    return parser.parse_args()

def process_image_stack(img_stack:np.ndarray, frame_step:int):
    '''
    Generates image slices of `image_stack`, showing slices of size `frame_step`.
    Args:
        image_stack (np.ndarray): Image stack to be sliced
        frame_step (int): Number of frames per stack
    Returns:
        dict: Gallery of slices converted to grayscale
    '''

    gallery={}

    for idx in range(frame_step, img_stack.shape[0], frame_step):
        start_id = idx - frame_step
        gallery[f'{start_id}_to_{idx}_frames'] = matrix_processing.max_image(img_stack[start_id:idx])

    # Add all frames as the last image
    gallery['all_frames'] = matrix_processing.max_image(img_stack)

    # Convert all images to grayscale (for saving as an image)
    for key, value in gallery.items():
        gallery[key] = matrix_processing.gray_to_rgb(value)

    return gallery

def main(args):

    img_stack = np.load(args.input)
    FRAME_STEP = 200

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    if args.frame_step is not None:
        FRAME_STEP = args.frame_step

    gallery = process_image_stack(img_stack, FRAME_STEP)

    _, name = os.path.split(args.input)
    base, _ = os.path.splitext(name)
    new_filename = filename_builder.create_out_filename(f'./images/{base}', prefixes=[], suffixes=['slices'])
    print(new_filename)
    visualisers.imshow(new_filename, **gallery)

if __name__ == "__main__":
    args = parse_args()

    main(args)
