#!/bin/python3
import os
import sys
import numpy as np
import video_matrix
import cv2 as cv
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

argparser = argparse.ArgumentParser(description='Data processing of matrices that are output by video_matrix.py')
req = argparser.add_argument_group('required arguments')
req.add_argument('-i', '--input', type=str, required=True, help='Input file, can be either `.npy` or `.mp4` and it will be parsed correctly')

optional = argparser.add_argument_group('optional arguments')

argparser._action_groups.append(optional)


def min_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.min(axis=0)

def max_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.max(axis=0)

def variance_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.var(axis=0)

def create_histogram(image, save_as):
    '''
    Creates a histogram of an image, throws away 0s because there is so many of them that it makes it difficult to read the histogram.
    '''
    hist, bins = np.histogram(image[image > 0], bins=256, range=(0,256))

    plt.figure()
    plt.title(f'Histogram of masked Max-Min image for {save_as}')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.bar(bins[:-1], hist, width=1, align='center')
    plt.savefig(f'{save_as}_hist')
    plt.close()

def mask_img_with_min(to_mask, min_img):

    mask = (min_img > 0).astype(dtype=np.uint8)
    kernel_size = (20,20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    morphed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return cv.bitwise_and(to_mask, to_mask, mask=morphed_mask)

def gray_to_rgb(in_img):
    return cv.cvtColor(in_img, cv.COLOR_GRAY2RGB)

def main(args):

    base_name      = args.input.split('/')[-1].split('.')[0]
    file_extension = args.input.split('/')[-1].split('.')[-1]

    if file_extension == 'mp4':
        out = video_matrix.create_video_matrix(args.input)
        vid_mat = video_matrix.rotate_frames(out)
    else:
        try:
            print('Loading .npy file')
            vid_mat = np.load(args.input)
        except OSError as e:
            print('Could not load the .npy file, please try again')
            print(e)
            sys.exit(-1)

    print('Creating different representations..')
    min_img = min_image(vid_mat)
    max_img = max_image(vid_mat)
    max_min = max_img - min_img
    var_img = variance_image(vid_mat)

    masked_maxmin = mask_img_with_min(to_mask=max_min, min_img=min_img)
    create_histogram(masked_maxmin, base_name)

    visualisers.imshow(title=f'./images/{base_name}_gallery', min=gray_to_rgb(min_img), max=gray_to_rgb(max_img), max_minus_min=gray_to_rgb(max_min), masked_maxmin=gray_to_rgb(masked_maxmin))

    visualisers.imshow(title=f'./images/{base_name}_variance', variance=var_img)

if __name__ == "__main__":

    args = argparser.parse_args()
    main(args)
