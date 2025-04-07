import csv
import os
import sys
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from image_registration.video_matrix import create_video_matrix
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from utils.filename_builder import create_out_filename
from utils.pprint import pprint_argparse

def parse_args():
    # Argparse configuration
    argparser = argparse.ArgumentParser(description='Creating a video matrix and rotating it')

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input', type=str, help='Path to the input video', required=True)

    # Optional arguments
    optional = argparser.add_argument_group('optional arguments')
    optional.add_argument('--save', action=argparse.BooleanOptionalAction, default=False, help='Save the registered image stack')
    optional.add_argument('-sr', '--sampling_rate', default=1, type=int, help='Sampling rate for the rotation')

    return argparser.parse_args()


def glue_image_stack(img_stack:np.ndarray, config=None):

    extractor = DISK(max_num_keypoints=2500).eval().cuda()  # load the extractor
    matcher   = LightGlue(features='disk').eval().cuda()  # load the matcher

    fixed_np    = img_stack[0]
    fixed_image = numpy_image_to_torch(cv.cvtColor(img_stack[0], cv.COLOR_GRAY2RGB)).cuda()
    fixed_feats = extractor.extract(fixed_image)

    for i, moving in tqdm(enumerate(img_stack[1:], start=1), total=len(img_stack)-1, desc='Glueing'):
        moving_image = numpy_image_to_torch(cv.cvtColor(moving, cv.COLOR_GRAY2RGB)).cuda()
        moving_feats = extractor.extract(moving_image)
        matches      = matcher({'image0': fixed_feats, 'image1': moving_feats})

        feats0, feats1, matches01 = [rbd(x) for x in [fixed_feats, moving_feats, matches]]  # remove batch dimension

        matches = matches01['matches']  # indices with shape (K,2)
        points_fixed = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points_moved = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        #points_fixed = points_fixed.cpu().numpy()
        #points_moved = points_moved.cpu().numpy()

        H, _ = cv.findHomography(points_moved.cpu().numpy(), points_fixed.cpu().numpy(),
                                 method=cv.RANSAC,
                                 ransacReprojThreshold=1.0)
        moving_warp = cv.warpPerspective(moving, H, (fixed_np.shape[1], fixed_np.shape[0]))
        img_stack[i] = moving_warp

    return img_stack


def main(args):
    _, name = os.path.split(args.input)
    base, _ = os.path.splitext(name)
    pprint_argparse(args)
    img_stack = create_video_matrix(args.input, downscale_factor=1, sampling_rate=args.sampling_rate)[5:]
    print(img_stack.shape)
    out_stack = glue_image_stack(img_stack)

    if args.save:
        save_as = create_out_filename(f'./_npy_files/{base}', prefixes=[], suffixes=['registered'])
        np.save(save_as, out_stack)
    np.save('tmp_reg_stack', out_stack)

if __name__ == "__main__":
    args = parse_args()
    main(args)
