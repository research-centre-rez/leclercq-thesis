import os
import argparse
import logging
from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename
import video_preprocessing

def parse_args():
    argparser = argparse.ArgumentParser(description='Program for processing a single video, interacts with the video_processor class API')

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input',  type=str, required=True, help='Path to the input video')
    req.add_argument('-o', '--output', type=str,required=True, help='Where do you want the video to be saved')
    req.add_argument('--method', choices=['none', 'opt_flow', 'approx'], required=True, help='Which video processing method do you want to use?')

    optional.add_argument('-sr', '--sampling_rate',type=int, default=1, help='Sampling rate of the new video')
    optional.add_argument('-df', '--downscale_factor',type=int, default=2, help='How much you want to downscale the video, due to encoding issues keeping the video in original resolution does not work')
    optional.add_argument('-gs', '--grayscale', default=True, action=argparse.BooleanOptionalAction, help='Do you want the out video to be in grayscale?')
    optional.add_argument('--start_at', default=15,type=int, help='How many frames should be thrown away from the original video')

    return argparser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    proc = video_preprocessing.VideoProcessor(sampling_rate=args.sampling_rate,
                                              downscale_factor=args.downscale_factor,
                                              gray_scale=args.grayscale,
                                              method=args.method,
                                              start_at=args.start_at)

    if args.output == 'auto' or args.output == 'automatic':
        base, _ = os.path.splitext(args.input)
        save_as = create_out_filename(base, [], ['preprocessed'])
        args.output = append_file_extension(save_as, 'mp4')
    proc.process_video(args.input, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
