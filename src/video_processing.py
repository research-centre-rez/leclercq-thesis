import os
import json
import argparse
import logging
from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename
from utils.validate_json_config import pprint_errors, validate_config
import video_processing

CONFIG_SCHEMA = {
  "fps_out": int,
  "sampling_rate": int,
  "downscale_factor": int,
  "grayscale": bool,
  "start_at": int,
  "num_points": int,
  "f_params": {
    "maxCorners": int,
    "qualityLevel": float,
    "minDistance": int,
    "blockSize": int
  },
  "lk_params": {
    "winSize": list,
    "maxLevel": int,
    "criteria": list
  }
}
def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for processing a single video, interacts with the video_processor class API"
    )

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group("required arguments")

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
        help="Which video processing method do you want to use?",
    )

    optional.add_argument(
        "--config",
        default="./video_processing/default_config.json",
        type=str,
        help="Path to a JSON config file for optical flow",
    )

    return argparser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args)

    config = load_config(args.config)
    pprint.pprint_dict(config, desc='Config parameters')
    if config:
        validation_errors = validate_config(config, CONFIG_SCHEMA)
        if validation_errors:
            pprint_errors(validation_errors)

    proc = video_processing.VideoProcessor(method=args.method, config=config)

    if args.output == "auto" or args.output == "automatic":
        base, _ = os.path.splitext(args.input)
        save_as = create_out_filename(base, [], ["preprocessed"])
        args.output = append_file_extension(save_as, "mp4")
    proc.process_video(args.input, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
