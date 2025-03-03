#!/bin/python3
from fileinput import filename
import cv2 as cv
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
import logging

from utils import visualisers
from utils import pprint
from utils import filename_builder

def parse_args():
    # Argparse configuration
    argparser = argparse.ArgumentParser(description='Creating a video matrix and rotating it')

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input', type=str, help='Path to the input video', required=True)
    req.add_argument('--rotate', action=argparse.BooleanOptionalAction, help='Whether to register each frame onto the starting frame', required=True)

    # Optional arguments
    optional = argparser.add_argument_group('optional arguments')
    optional.add_argument('--save', action=argparse.BooleanOptionalAction, help='Save the rotated video matrix into a .npy file, if --no-rotate is parsed then the non-rotated video matrix is saved into a .npy file. File name will be the same as the video filename.')
    optional.add_argument('-gs', '--grayscale', default=True, action=argparse.BooleanOptionalAction, help='Use grayscale video or not. It is recommended to use grayscale as it uses 1/3 of the storage.')
    optional.add_argument('-ds', '--downscale_factor',default=2, type=int, help='How much the video should be downscaled. Has to be an integer')
    optional.add_argument('-co', '--center_offset', default=(14.08033127, 19.36611469), type=float, nargs=2, help='Custom center_offset')
    optional.add_argument('-sr', '--sampling_rate', default=1, type=int, help='Sampling rate for the rotation')

    return argparser.parse_args()


def tqdm_generator():
    '''
    Simple tqdm generator that allows the use of tqdm with a while loop.
    '''
    while True:
        yield

def rotate_frames(frames, center_offset=(14.08033127, 19.36611469), save_as=None, sampling_rate=1) -> np.ndarray:
    if type(frames) == str:
        try:
            frames = np.load(frames)
        except OSError as e:
            print('Could not load the .npy file, please try again')
            print(e)
            sys.exit(-1)

    # Calculate rotation centre
    h,w = frames[0].shape

    # Center offset
    c_offset = np.array(center_offset) / 2 # this is because i am halving the videos in size

    # Rotation center
    img_center = np.array(frames[0].shape[:2][::-1]) / 2
    rot_center = img_center + c_offset

    print(f'Image center: {img_center}')
    print(f'Rotation center: {rot_center}')

    # Rotation between each frame
    #rot_per_frame = 0.1851 # from jupyter notebook
    rot_per_frame = 0.1491 # from training data
    rots = np.array([np.round(rot_per_frame * i, 5) for i in range(frames.shape[0])])

    for i,(frame, rotation) in enumerate(tqdm(zip(frames, rots), total=frames.shape[0])):
        if i % sampling_rate != 0:
            frames[i] = 0
        else:
            # Get the rotation around the rotation center
            M = cv.getRotationMatrix2D(center=rot_center, angle=rotation, scale=1)

            # Rotate the frame
            rotated_frame = cv.warpAffine(frame, M, (w,h))

            # Replace the frame in-memory, saving space
            frames[i] = rotated_frame


    if sampling_rate != 1:
        zeroes = np.where(np.all(frames == 0, axis=(1,2)))[0]
        frames = np.delete(frames, zeroes, axis=0)

    if save_as:
        np.save(save_as, frames)

    return frames

def create_video_matrix(vid_path:str, grayscale=True, save_as=None, downscale_factor = 2) -> np.ndarray:
    '''
    This function loads a video into a matrix where each row is a frame in the video. There is an option to save the matrix as a `.npy` file. If the video is too large to fit into memory, it can be further downscaled with the `downscale_factor` parameter.

    Parameters:
        `vid_path`:str : Path to the video file, if the path points to a non-existing file, returns an empty np array.
        `grayscale`:bool : Whether the store the video as grayscale, defaults to True.
        `save_as`:str : If the user wants to store the matrix as a `.npy` file, they can pass in a name of the file and it will be saved. If None is passed in, no saving is done.
        `downscale_factor`:int : How much should the video be downscaled. Both `h` and `w` will be divided by this number, preserving aspect ratio.

    Returns:
        `frames`:np.ndarray : Video loaded into a matrix.
    '''

    # First, check whether the video has been loaded
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Could not load video, returning')
        return np.zeros(0)

    # Append the frames to a Python list first
    frames = []

    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    new_w = int(w // downscale_factor)
    new_h = int(h // downscale_factor)

    for _ in tqdm(tqdm_generator(), desc='Buffering video'):
        ret, frame = cap.read()
        if ret:
            if grayscale:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_CUBIC)
            frames.append(frame)
        else:
            cap.release()
            break

    frames_np = np.array(frames)
    try:
        cap.release()
    finally:
        # Cache the matrix
        if save_as:
            np.save(save_as, frames_np)
    return frames_np

def main(args):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    # Removing .mp4 extension
    base_name = os.path.basename(args.input).split('.')[0]

    if not args.rotate:
        save_as = filename_builder.create_out_filename(base_name, [], ['not', 'rotated'])
        save_to = os.path.join('./npy_files', save_as)
        create_video_matrix(vid_path=args.input,
                            grayscale=args.grayscale,
                            save_as=save_to,
                            downscale_factor=args.downscale_factor)

    if args.rotate:
        save_as = filename_builder.create_out_filename(base_name, ['temp'], ['rotated'])
        save_to = os.path.join('./npy_files', save_as)
        out = create_video_matrix(vid_path=args.input,
                                  grayscale=args.grayscale,
                                  downscale_factor=args.downscale_factor)
        rotate_frames(frames=out,
                      center_offset=args.center_offset,
                      save_as=save_to if args.save else None,
                      sampling_rate=args.sampling_rate)

if __name__ == "__main__":
    args = parse_args()
    main(args)
