import os
import sys
import argparse
import logging
import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils import pprint
from utils import filename_builder
from utils.tqdm_utils import tqdm_generator

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
    optional.add_argument('-p', '--prefix', type=str, nargs='*', help='Prefixes for the saved file')
    optional.add_argument('-s', '--suffix', type=str, nargs='*', help='Suffixes for the saved file')

    return argparser.parse_args()


def rotate_frames_optical_flow(video_path, angles):
    logger = logging.getLogger(__name__)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f'Error: Could not open file {video_path}')
        return None
    
    ret, frame = cap.read()
    if not ret:
        logger.error('Error with reading the first frame')
        return None

    rot_x      = 1870.1321
    rot_y      = 1074.0583
    rot_center = (rot_x, rot_y)

    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    out_w = 1920
    out_h = 1080
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv.VideoWriter('./test_video.mp4', fourcc, 35, (out_w, out_h))

    angle_idx = 0
    angle     = np.float64(0.0)

    print(len(angles))
    print(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_counter = tqdm(desc='Creating new video')
    for _ in tqdm_generator():
        ret, frame = cap.read()

        if not ret or angle_idx >= len(angles):
            break

        angle         += angles[angle_idx]
        M             = cv.getRotationMatrix2D(center=rot_center, angle=angle, scale=1)
        rotated_frame = cv.warpAffine(frame, M, (w,h))

        frame_counter.set_postfix(angle=f'{angle:.4f}')

        scaled_frame = cv.resize(rotated_frame, (out_w, out_h))
        angle_idx    += 1

        frame_counter.update(1)
        out.write(scaled_frame)

    out.release()
    cap.release()
    cv.destroyAllWindows()
    print('Video successfully created')

def rotate_frames(frames, center_offset=(14.08033127, 19.36611469), save_as=None, sampling_rate=1) -> np.ndarray:
    logger = logging.getLogger(__name__)
    if isinstance(frames, str):
        try:
            frames = np.load(frames)
        except OSError as e:
            logger.error('Could not load the .npy file, please try again')
            logger.error(e)
            sys.exit(-1)

    # Calculate rotation centre
    h,w = frames[0].shape

    # Center offset
    c_offset = np.array(center_offset) / 2 # this is because i am halving the videos in size

    # Rotation center
    img_center = np.array(frames[0].shape[:2][::-1]) / 2
    rot_center = img_center + c_offset

    logger.info('Image center: %s', img_center)
    logger.info('Rotation center: %s', rot_center)

    # Rotation between each frame
    #rot_per_frame = 0.1851 # from jupyter notebook
    #rot_per_frame = np.float32(0.14815) # from training data
    rot_per_frame = np.float32(0.14798) # from training data
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

def create_video_matrix(vid_path:str, grayscale=True, save_as=None, downscale_factor=2, sampling_rate=1) -> np.ndarray:
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
    if sampling_rate != 1:
        indices   = np.array(range(0, len(frames_np), sampling_rate))
        frames_np = frames_np[tuple(indices.T), :]


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
        save_as = filename_builder.create_out_filename(f'./npy_files/{base_name}',
                                                       args.prefix,
                                                       ['not', 'rotated'] + (args.suffix))
        create_video_matrix(vid_path=args.input,
                            grayscale=args.grayscale,
                            save_as=save_as,
                            downscale_factor=args.downscale_factor,
                            sampling_rate=args.sampling_rate)

    if args.rotate:
        save_as = filename_builder.create_out_filename(f'./npy_files/{base_name}',
                                                       args.prefix,
                                                       ['rotated'] + (args.suffix))
        out = create_video_matrix(vid_path=args.input,
                                  grayscale=args.grayscale,
                                  downscale_factor=args.downscale_factor)
        rotate_frames(frames=out,
                      center_offset=args.center_offset,
                      save_as=save_as if args.save else None,
                      sampling_rate=args.sampling_rate)

if __name__ == "__main__":
    args = parse_args()
    main(args)
