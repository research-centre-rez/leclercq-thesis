import os
import sys
import argparse

import numpy as np
import cv2 as cv
from tqdm import tqdm

from image_processing.masking import crop_img, resize_or_pad
from utils.create_video_from_stack import create_video_from_stack
from image_registration.matrix_processing import create_min_mask, min_image
from image_registration.video_matrix import create_video_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Creates a video from a given image stack. There are options to display the displacement as well and to show the estimated center of rotation.')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', required=True, help='Path to the video that should be masked')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-ns', '--new_size', default=1090, type=int, help='What should be the size of the new video, only square videos supported right now')
    return parser.parse_args()

def mask_video(vid_name:str, new_size=1090):
    _, name = os.path.split(vid_name)
    base, _ = os.path.splitext(name)

    img_stack = create_video_matrix(vid_path, downscale_factor=1)
    min_image = min_image(img_stack)
    min_mask = create_min_mask(min_image)

    cropped_frames = []
    for i, frame in tqdm(enumerate(img_stack), total=len(img_stack), desc='Cropping video'):
        cropped = crop_img(frame, min_mask)
        padded  = resize_or_pad(cropped, new_size)
        cropped_frames.append(padded)

    np_stack = np.array(cropped_frames)
    create_video_from_stack(np_stack, disp=None, mesh=None, sample_name=f'{base}_cropped')

def main(args):
    mask_video(args.input, args.new_size)

if __name__ == "__main__":
    args = parse_args()
    main(args)
