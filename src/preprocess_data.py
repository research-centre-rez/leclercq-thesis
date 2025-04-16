"""
Video preprocessing

OpenCV has a default setting that specifies how many frames it will read from a video, you should change this before running any of these functions with:

export OPENCV_FFMPEG_READ_ATTEMPTS=100000

While the os.environ written below should do it for you, it is possible that it might not work.
"""
import os
import argparse
import logging

from utils import pprint
from data_preprocessing import detect_black_frames
from data_preprocessing import split_video

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100000"

def parse_args():
    argparser = argparse.ArgumentParser(description='Program for identifying black sections of a video and splitting the video into 2 separate parts. Note: The parts are not labeled according to their light angle, they are arbitrarily numbered.')

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input', type=str, required=True, help='Path to the input video')
    req.add_argument('-o', '--output', type=str,required=True, help='Where do you want the video to be saved')

    optional.add_argument('--is_dir', default=False, action=argparse.BooleanOptionalAction, help='Is input a directory?')

    return argparser.parse_args()

def process_videos(directory_path:str, output_dir:str) -> None:
    '''
    Processes all videos found in `directory_path` and saves them in `output_dir`.
    '''
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    failed_files = []

    for file in sorted(os.listdir(directory_path)):
        if file.endswith('.MP4'):
            video_path = os.path.join(directory_path, file)

            black_frame_idx, fps = detect_black_frames(video_path)

            if len(black_frame_idx) % 2 == 0:
                split_video(video_path, black_frame_idx, output_dir, fps)
            else:
                logger.info('Was not able to split file %s.', file)
                failed_files.append(file)

    with open('failed_files.log', 'w') as log_file:
        for file in failed_files:
            log_file.write(file + '\n')

def process_video(vid_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    black_f_id, fps = detect_black_frames(vid_path)
    split_video(vid_path, black_f_id, out_path, fps)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    if args.is_dir and os.path.isdir(args.input):
        logger.info('Processing all video in directory %s', args.input)
        process_videos(directory_path=args.input, output_dir=args.output)

    else:
        logger.info('Processing video %s', args.input)
        process_video(vid_path=args.input, out_path=args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
