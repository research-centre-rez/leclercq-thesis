import os
import sys

import numpy as np
import cv2 as cv
from tqdm import tqdm

from image_processing.masking import crop_img, resize_or_pad
from image_registration.create_video_from_stack import create_video_from_stack
from image_registration.matrix_processing import create_min_mask, mask_img_with_min, min_image
from image_registration.video_matrix import create_video_matrix

# TODO: Make this into a function so that it can be called in other parts as well
if __name__ == "__main__":
    vid_path = './test_video.mp4'
    img_stack = create_video_matrix(vid_path, downscale_factor=1)
    min_image = min_image(img_stack)
    print(img_stack.shape)

    min_mask = create_min_mask(min_image)
    print(min_mask.shape)

    new_size = 1090
    cropped_frames = []
    for i,frame in tqdm(enumerate(img_stack), total=img_stack.shape[0], desc='Cropping frames'):
        #masked  = cv.bitwise_and(img_stack[i], img_stack[i], mask=min_mask)
        cropped = crop_img(frame, min_mask)
        padded  = resize_or_pad(cropped, new_size)
        cropped_frames.append(padded)

    np_frames = np.array(cropped_frames)
    create_video_from_stack(np_frames, disp=None, mesh=None, sample_name='test_video_cropped')
