import os
import subprocess
import logging

import cv2 as cv

from utils.filename_builder import append_file_extension, create_out_filename

def seconds_to_hms(seconds) -> str:
    '''
    Conversion from seconds to hours/minutes/secods
    '''
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f'{hours:02}:{minutes:02}:{secs:05.3f}'

def split_video(video_path:str, frame_idx:list[int], output_dir:str, fps:float) -> None:
    """"
    Given the path of a video, split it into separate videos such that each sub-video contains only one angle of lighting.
    ffmpeg is used in order to preserve the quality of the video(s).

    Args:
        video_path (str): Path to the video you wish to split
        frame_idx (list[int]): Indices of how to split the video
        output_dir (str): Target location for the new videos, they will be stored in their own directory to prevent cluttering
        fps (float): Used for calculating start and end times for ffmpeg
    Returns: 
        None
    """

    logger = logging.getLogger(__name__)
    base_name, _ = os.path.splitext(os.path.basename(video_path))

    output_dir = os.path.join(output_dir, (base_name.split('-')[0]))

    # The videos will be saved in their own directory under `output_dir`
    logger.info('%s is going to be saved to %s', base_name, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    video_index = 0 #index of video part
    for i in range(0, len(frame_idx), 2):
        logger.info('indices of frames: %s', (frame_idx[i], frame_idx[i+1]))

        start, end = frame_idx[i], frame_idx[i+1]

        # calculate the start and end times
        start_t = round((start / fps), 2)
        end_t = round((end / fps), 2)

        # ffmpeg uses hms time format so we convert
        start_t = seconds_to_hms(start_t)
        end_t = seconds_to_hms(end_t)

        out_name = create_out_filename(base_name, [], [f'part{video_index}'])
        out_name = append_file_extension(out_name, 'mp4')
        output_file = os.path.join(output_dir, out_name)
        subprocess.call([
            'ffmpeg', '-i', video_path, '-ss', str(start_t), '-to', str(end_t), '-c', 'copy', '-n', output_file
        ])
        video_index += 1
