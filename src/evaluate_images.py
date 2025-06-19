import argparse
import logging
import os
import sys
import csv

import image_evaluation
from tqdm import tqdm
from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Program for evaluating the quality of a fused image stack",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = argparser.add_argument_group("required arguments")

    required.add_argument(
        "-i", "--input", type=str, nargs="+", required=True, help="TODO"
    )

    required.add_argument("-o", "--output", type=str, required=True, help="TODO")

    required.add_argument(
        "-m",
        "--method",
        type=str,
        required=True,
        help="Which method was used for registration?",
    )
    return argparser.parse_args()


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    min_images = []
    max_images = []
    for filename in tqdm(args.input, desc="Evaluating images"):
        nglv_score = image_evaluation.normalised_grey_level_variance(filename)
        nglv_score = round(nglv_score, 3)
        brenner_score = image_evaluation.brenner_method(filename)

        _, name = os.path.split(filename)
        base, _ = os.path.splitext(name)

        base_split = base.split("_")

        sample_name = base_split[0]
        fuse_type = base_split[-1]

        if fuse_type == "MAX":
            max_images.append([sample_name, nglv_score, brenner_score])
        elif fuse_type == "MIN":
            min_images.append([sample_name, nglv_score, brenner_score])

    out_filename_max = create_out_filename(args.output, [], [args.method, "max"])
    out_filename_max = append_file_extension(out_filename_max, "csv")
    out_filename_min = create_out_filename(args.output, [], [args.method, "min"])
    out_filename_min = append_file_extension(out_filename_min, "csv")

    logger.info("Storing evaluation for max images into the file %s", out_filename_max)

    with open(out_filename_max, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Sample name", "NGLV score", "Brenner score"])
        for line in max_images:
            csv_writer.writerow(line)

    logger.info("Storing evaluation for min images into the file %s", out_filename_min)
    with open(out_filename_min, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Sample name", "NGLV score", "Brenner score"])
        for line in min_images:
            csv_writer.writerow(line)


if __name__ == "__main__":
    args = parse_args()
    main(args)
