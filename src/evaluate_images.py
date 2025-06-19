import argparse
import logging
import os
import sys

import image_evaluation
from tqdm import tqdm
from utils import pprint

def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for evaluating the quality of a fused image stack",
        formatter_class=argparse.RawTextHelpFormatter
    )

    required = argparser.add_argument_group("required arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="TODO"
    )

    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="TODO"
    )

    required.add_argument(
        "--m",
        "--method",
        type=str,
        required=True,
        help="Which method was used for registration?"
    )
    return argparser.parse_args()

def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)


    for file in tqdm(args.input, desc="Evaluating images"):
        print(file)



if __name__ == "__main__":
    args = parse_args()
    main(args)

