import cv2 as cv
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

def tqdm_generator():
    '''
    Simple tqdm generator that allows the use of tqdm with a while loop.
    '''
    while True:
        yield

def rotate_frames(frames, center_offset=(14.08033127, 19.36611469), save_as=None) -> np.ndarray:
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
    rot_center = np.array(frames[0].shape[:2][::-1]) / 2 + c_offset

    print(f'Rotation center: {rot_center}')

    # Rotation between each frame
    rot_per_frame = 0.1851 # from jupyter notebook
    rots = np.array([rot_per_frame * i for i in range(frames.shape[0])])

    for i,(frame, rotation) in enumerate(tqdm(zip(frames, rots), total=frames.shape[0])):
        # Get the rotation around the rotation center
        M = cv.getRotationMatrix2D(center=rot_center, angle=rotation, scale=1)

        # Rotate the frame
        rotated_frame = cv.warpAffine(frame, M, (w,h))

        # Replace the frame in-memory, saving space
        frames[i] = rotated_frame

    if save_as:
        np.save(save_as, frames)

    return frames

def buffer_video(vid_path:str, grayscale=True, save_as=None, downscale_factor = 2) -> np.ndarray:
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

    frames = np.array(frames)

    # Cache the matrix
    if save_as:
        np.save(save_as, frames)
    return np.array(frames)

if __name__ == "__main__":
    vid_path = '../video_processing/calibration_video/calibration_video-part0.mp4'

    rotate_frames('calib_vid.npy', save_as='calib_vid_rotated.npy')
