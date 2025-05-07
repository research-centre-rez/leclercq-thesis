import logging
import cv2 as cv
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_video_matrix(vid_path: str, grayscale: bool) -> np.ndarray:
    """
    This function loads a video into a matrix where each row is a frame in the video.

    Parameters:
        `vid_path`:str : Path to the video file, if the path points to a non-existing file, returns an empty np array.

    Returns:
        `frames`:np.ndarray : Video loaded into a matrix.
    """

    # First, check whether the video has been loaded
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        logger.error("Could not load video, returning")
        return np.zeros(0)

    # Append the frames to a Python list first
    frames = []

    with tqdm(
        desc="Creating video matrix", total=cap.get(cv.CAP_PROP_FRAME_COUNT)
    ) as pbar:
        while True:
            ret, frame = cap.read()
            if ret:
                if grayscale:
                    frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
                else:
                    frames.append(frame)
                pbar.update(1)
            else:
                break

    frames_np = np.array(frames)

    return frames_np
