import os
import sys
import argparse
import logging
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from image_registration import video_matrix

from utils import visualisers
from utils import pprint
from utils import filename_builder

def parse_args():
    argparser = argparse.ArgumentParser(description='Data processing of matrices that are output by video_matrix.py')
    req = argparser.add_argument_group('required arguments')
    req.add_argument('-i', '--input', type=str, required=True, help='Input file, can be either `.npy` or `.mp4` and it will be parsed correctly')

    optional = argparser.add_argument_group('optional arguments')
    optional.add_argument('-r', '--representations', nargs='+', choices=['min', 'max', 'max_minus_min', 'masked_maxmin', 'variance'], default=['min', 'max', 'max_minus_min', 'masked_maxmin'], help='Choose which representations to display')

    return argparser.parse_args()

def min_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.min(axis=0)

def max_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.max(axis=0)

def variance_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.var(axis=0, dtype=np.float32)

def create_histogram(image, save_as):
    '''
    Creates a histogram of an image, throws away 0s. 
    Args:
        image: Image to create histogram of
        save_as: name of the output file
    '''
    hist, bins = np.histogram(image[image > 0], bins=256, range=(0,256))

    name = os.path.basename(save_as).split('_hist')[0]

    plt.figure()
    plt.title(f'Histogram of masked Max-Min image for {name}')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.bar(bins[:-1], hist, width=1, align='center')
    plt.savefig(save_as)
    plt.close()

def mask_img_with_min(to_mask, min_img):
    '''
    Masks image with a min image. This can be done because the min image creates a nice circular mask as the video is rotated.
    Args:
        to_mask: Image to be masked
        min_img: Mask to be applied
    Returns:
        Masked image
    '''

    mask = (min_img > 0).astype(dtype=np.uint8)
    kernel_size = (20,20)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
    morphed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return cv.bitwise_and(to_mask, to_mask, mask=morphed_mask)

def gray_to_rgb(in_img):
    return cv.cvtColor(in_img, cv.COLOR_GRAY2RGB)

def main(args):

    base_name, file_extension = os.path.basename(args.input).split('.')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    if file_extension == 'mp4':
        out = video_matrix.create_video_matrix(args.input)
        vid_mat = video_matrix.rotate_frames(out, save_as=None)
    else:
        try:
            logger.info('Loading .npy file')
            vid_mat = np.load(args.input)
        except OSError as e:
            logger.error('Could not load the .npy file, please try again')
            logger.error(e)
            sys.exit(-1)

    logger.info('Creating different representations..')
    reps = {}

    if 'min' in args.representations:
        reps['min'] = min_image(vid_mat)

    if 'max' in args.representations:
        reps['max'] = max_image(vid_mat)

    if 'max_minus_min' in args.representations:
        max_min = reps.get('max', max_image(vid_mat)) - reps.get('min', min_image(vid_mat))
        reps['max_minus_min'] = max_min

    if 'masked_maxmin' in args.representations:
        max_min = reps.get('max_minus_min', max_image(vid_mat) - min_image(vid_mat))
        masked_maxmin = mask_img_with_min(to_mask=max_min, min_img=reps.get('min', min_image(vid_mat)))
        reps['masked_maxmin'] = masked_maxmin
        save_as = filename_builder.create_out_filename(base_name, [], ['hist'])
        save_to = os.path.join('./images', save_as)
        create_histogram(masked_maxmin, save_to)

    if 'variance' in args.representations:
        reps['variance'] = variance_image(vid_mat)

    for key, value in reps.items():
        # Variance image can have values higher than 255, therefore we don't convert it to RGB
        if key == 'variance':
            continue
        reps[key] = gray_to_rgb(value)

    save_as = filename_builder.create_out_filename(base_name, [], ['gallery'])
    save_to = os.path.join('./images', save_as)

    visualisers.imshow(title=save_to, **reps)

if __name__ == "__main__":

    args = parse_args()
    main(args)
