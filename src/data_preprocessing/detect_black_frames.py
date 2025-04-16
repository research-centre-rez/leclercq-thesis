import os
import logging
import sys
import cv2 as cv
import numpy as np
from tqdm import tqdm
def detect_black_frames(vid_path:str, threshold=5):
    """
    Given a video path, identify the frames where there the screen turns black
    Args:
        vid_path (str): Path to the video where you want to find black parts
        threshold (int): Maximum threshold for a frame to be considered not black
    Returns:
        black_frames_ids:list
        fps: float
    """
    logger = logging.getLogger(__name__)

    prev_black = True

    cap = cv.VideoCapture(vid_path, cv.CAP_FFMPEG)
    fps = cap.get(cv.CAP_PROP_FPS)
    if cap.isOpened():
        logger.info('Succesfully loaded video: %s', vid_path)

    black_frame_indices = []
    frame_count         = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    center_x = width // 2
    center_y = height // 2
    half_size = 500

    start_x = center_x - half_size
    start_y = center_y - half_size
    end_x   = center_x + half_size
    end_y   = center_y + half_size

    with tqdm(range(frame_count), leave=False) as progress:
        for i in progress:
            ret, frame = cap.read()

            if not ret:
                break

            mean = np.mean(frame[start_y:end_y, start_x:end_x])
            progress.set_postfix(mean=mean)

            # End of video and its not black
            if i == frame_count-1 and not prev_black:
                black_frame_indices.append(i)

            # Previous frame was black and the current one is not
            elif prev_black and mean > threshold:
                black_frame_indices.append(i)
                prev_black = False

            # Previous frame wasn't black and the current one is
            elif not prev_black and mean < threshold:
                black_frame_indices.append(i-1)
                prev_black = True

    cap.release()
    logger.info("Indices where there are black frames: %s", black_frame_indices)
    return black_frame_indices, fps
