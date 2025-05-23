import argparse
import logging
import os
import sys

from image_fusion import ImageFuser, FuseMethod
from utils import pprint
from utils.filename_builder import create_out_filename


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for image fusion of a video stack",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = argparser.add_argument_group("required arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Relative (or absolute) path to a registered video stack. You can pass in multiple files at the same time.",
    )

    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Where do you want to save the fused image(s). If 'auto' is passed, the program will automatically construct the name of the output file.\n"
                " Example: --o auto -> will automatically append all the used methods to the output file"
        ),
    )

    required.add_argument(
        "-m",
        "--method",
        nargs="+",
        required=True,
        choices=["min", "max", "var", "med", "mean"],
        help=(
            "Which image fusing methods do you want to use? The choices are:"
            "\n min: Take minimum from each frame across the whole stack"
            "\n max: Take maximum from each frame across the whole stack"
            "\n var: Variance of each pixel across the stack"
            "\n med: Median of each pixel across the stack"
            "\n mean: Mean of each pixel across the stack"
        ),
    )
    return argparser.parse_args()


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    fuser = ImageFuser()

    methods = [FuseMethod[m.upper()] for m in args.method]
    for input_path in args.input:
        if args.output in ["auto", "automatic"]:
            path, name = os.path.split(input_path)
            base, _ = os.path.splitext(name)
            output_path = create_out_filename(f"{path}/images/{base}", [], ["fused"])
        else:
            output_path = args.output

        logger.info("Processing %s -> %s", input_path, output_path)
        gallery = fuser.get_fused_gallery(input_path, methods)
        fuser.write_gallery_to_disc(gallery, output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
