import os
import sys
import json5
import argparse
import logging

import jsonschema
from jsonschema.exceptions import ValidationError

from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename
from video_processing import VideoProcessor
from video_processing import ProcessorMethod

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "fps_out": {"type": "integer"}, # the fps of the video that is produced
        "sampling_rate": {"type": "integer"}, # taking every ith frame from the original video
        "downscale_factor": {"type": "integer"}, # how much should the resolution be downscaled
        "grayscale": {"type": "boolean"}, # whether the out video is grayscale or not
        "start_at": {"type": "integer"}, # frame at which the new video will start
        "opt_flow_params": {
                "type": "object",
                "properties": {
                    "f_params": {
                        "type": "object",
                        "properties": {
                            "maxcorners": {"type": "integer"},
                            "qualitylevel": {"type": "number"},
                            "mindistance": {"type": "integer"},
                            "blocksize": {"type": "integer"},
                        },
                        "required": ["maxCorners", "qualityLevel", "minDistance", "blockSize"]
                    },
                    "lk_params": {
                        "type": "object",
                        "properties": {
                            "winsize": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minitems": 2,
                                "maxitems": 2
                            },
                            "maxlevel": {"type": "integer"},
                            "criteria": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minitems": 3,
                                "maxitems": 3
                            },
                        },
                        "required": ["winSize", "maxLevel", "criteria"]
                }
            },
            "required": ["f_params", "lk_params"]
        }
    },
    "estimate_params": {
            "type": "object",
            "rotation_center": {"type": "object",
                              "x": {"type": "float"},
                              "y": {"type": "float"},
                              "required": [
                              "x",
                              "y"
                              ]
                              },
            "rotation_per_frame": {"type": "float"},
            "required": [
                "center_offset",
                "rotation_per_frame"
            ]
    },
    "required": [
        "fps_out",
        "sampling_rate",
        "downscale_factor",
        "grayscale",
        "start_at",
        "opt_flow_params",
        "estimate_params"
    ]
}


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program that is used for performing rotation correction on a given video. The new video will be saved separately.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    # Required arguments
    req.add_argument(
        "-i", "--input", type=str, required=True, help="Relative (or absolute) path to the video you want to process. Example: --i ../data/1/1-part0.mp4"
    )
    req.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want the video to be saved. If 'auto' is passed, the program will automatically construct the name of the output file.\n"
            " Example #1: --o auto -> will automatically append '-processed' to the input file and save it like that.\n"
            " Example #2: --o ../data/tmp.mp4 -> store the video as 'tmp.mp4' at the location '../data/'.")
    )
    req.add_argument(
        "--method",
        choices=["none", "opt_flow", "approx"],
        required=True,
        help=(
            "Which video processing method to use? \n"
            " none - No rotation correction, mainly used for other types of video processing. \n"
            " opt_flow - Optical flow with the parameters from the config file to undo rotation. \n"
            " approx - Pre-calculated rotation center and angle correction to undo rotation. \n"
            "For each of these methods, you can downscale, downsample and grayscale the resulting video."
        )
    )
    optional.add_argument(
        "--config",
        default="./video_processing/default_config.json5",
        type=str,
        help="Path to a JSON config file that follows the config schema for video processing. The default config schema can be found in './video_process/default_config.json5'",
    )

    return argparser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json5.load(f)


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    try:
        config = load_config(args.config)
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        logger.info("Successfully loaded a JSON schema")
        pprint.pprint_dict(config, desc='Config parameters')
    except ValidationError as e:
        logger.error("Invalid configuration: \n %s", e.message)
        sys.exit(1)


    if args.output in ["auto", "automatic"]:
        base, _ = os.path.splitext(args.input)
        save_as = create_out_filename(base, [], ["preprocessed"])
        args.output = append_file_extension(save_as, "mp4")

    proc = VideoProcessor(method=ProcessorMethod[args.method.upper()], config=config)
    
    analysis = proc.get_rotation_analysis(args.input)
    proc.write_out_video(args.input, analysis, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
