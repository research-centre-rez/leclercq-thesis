import os
import json
import json5
import jsonschema
import argparse
import logging

from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename
from video_registration import RegMethod, VideoRegistrator


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for registering a single video. Stores the registered video as a numpy matrix where each row represents one frame of the video. Default config can be found in video_registration/default_config.json",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    argparser._action_groups.pop()
    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    # Required arguments
    required.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Relative (or absolute) path to the video you want to register. Example: --i ../data/1/1-part0_processed.mp4",
    )
    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want the registered matrix to be saved. If 'auto' is passed, the program will automatically construct the name of the output file.\n"
            "  Example #1: --o auto -> will automatically append _registered_stack to the output file \n"
            "  Example #2: --o ../data/tmp_registered will save the registered matrix as ../data/tmp_registered.npy"
        ),
    )
    required.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["orb", "lightglue", "mudic"],
        required=True,
        help=(
            "Which method do you want to use for performing video registration. Options are:\n"
            "  orb: Performs ORB finding and matching of keypoints.\n"
            "  mudic: Performs digital image correlation for video registration.\n"
            "  lightglue: Performs registration with the use of the LightGlue neural network."
        ),
    )

    optional.add_argument(
        "-c",
        "--config",
        default="./video_registration/default_config.json5",
        type=str,
        help="Path to a JSON5 config file that follows the config schema for video registration. Default config can be found in './video_registration/default_config.json5'",
    )

    return argparser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return json5.load(f)


def load_json_schema(path):
    with open(path, "r") as f:
        return json.load(f)


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    CONFIG_SCHEMA = load_json_schema("./video_registration/video_registration_schema.json")
    try:
        config = load_config(args.config)
        jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)
        logger.info("Successfully validated the submitted config")
        pprint.pprint_dict(config, "Config parameters", logger)
    except jsonschema.ValidationError as e:
        logger.error("Invalid configuration: \n %s", e.message)
        sys.exit(1)

    if args.output == "auto" or args.output == "automatic":
        base, _ = os.path.splitext(args.input)
        args.output = create_out_filename(base, [], ["registered", "stack"])

    args.method = RegMethod[args.method.upper()]

    reg = VideoRegistrator(method=args.method, config=config)

    reg_analysis = reg.get_registered_block(args.input)

    reg.save_registered_block(reg_analysis, args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
