import argparse
import logging
import os
import sys

from image_fusion import ImageFuser, FuseMethod
from utils import pprint
from utils.filename_builder import create_out_filename

def parse_args():
    argparser = argparse.ArgumentParser(description="Program for image fusion of a video stack",
                                        formatter_class=argparse.RawTextHelpFormatter)

    optional = argparser._action_groups.pop()
    required = argparser.add_argument_group("required arguments")
    optional = argparser.add_argument_group("optional arguments")

    required.add_argument(
        "-i",
        "--input", 
        type=str,
        nargs="+",
        required=True,
        help="TODO" # TODO: Add description for this arg
    )

    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="TODO" # TODO: Add description for this arg
    )

    # TODO: Add description for this arg
    required.add_argument(
        "-m",
        "--method",
        nargs="+",
        required=True,
        choices=['min', 'max', 'var'], 
        help='TODO' 
    )

    optional.add_argument(
        "-c",
        "--config", 
        default="./image_fusion/default_config.json5",
        type=str,
        help="TODO"
    )

    return argparser.parse_args()


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    # TODO: add json schema loading + validation

    fuser = ImageFuser(config=None)

    methods = [FuseMethod[m.upper()] for m in args.method]
    for input_path in args.input:
        if args.output in ["auto", "automatic"]:
            path, name = os.path.split(input_path)
            base, _ = os.path.splitext(name)
            output_path = create_out_filename(f'{path}/images/{base}', [], ["fused"])
        else:
            output_path = args.output

        logger.info("Processing %s -> %s", input_path, output_path)
        gallery = fuser.get_fused_gallery(input_path, methods)
        fuser.write_gallery_to_memory(gallery, output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
