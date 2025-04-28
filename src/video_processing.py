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

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "fps_out": {"type": "integer"}, # the fps of the video that is produced
        "sampling_rate": {"type": "integer"}, # taking every ith frame from the original video
        "downscale_factor": {"type": "integer"}, # how much should the resolution be downscaled
        "grayscale": {"type": "boolean"}, # whether the out video is grayscale or not
        "start_at": {"type": "integer"}, # where should the video start
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
            "center_offset": {"type": "array",
                              "items": [
                                {"type": "float"},
                                {"type": "float"}
                              ]},
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
        description="Program for processing a single video, interacts with the video_processor class API",
        formatter_class=argparse.RawTextHelpFormatter
    )

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    # Required arguments
    req.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input video"
    )
    req.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Where do you want the video to be saved",
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
        help="Path to a JSON config file that follows the config schema for video processing.",
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

    proc = VideoProcessor(method=args.method, config=config)

    if args.output in ["auto", "automatic"]:
        base, _ = os.path.splitext(args.input)
        save_as = create_out_filename(base, [], ["preprocessed"])
        args.output = append_file_extension(save_as, "mp4")
    proc.process_video(args.input, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
