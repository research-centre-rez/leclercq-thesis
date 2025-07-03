import argparse
import logging
import os
import sys
import csv
import re

from image_evaluation import (
    NGLV,
    BrennerMethod,
)
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


def write_scores_to_csv(csv_filename, columns: list[str], data: list):
    with open(csv_filename, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(columns)
        for line in data:
            csv_writer.writerow(line)


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    min_scores = []
    max_scores = []
    var_scores = []
    mean_scores = []

    nglv = NGLV()
    brenner = BrennerMethod()

    for filename in tqdm(args.input, desc="Evaluating images"):

        nglv_score = round(nglv.calculate_metric(img_path=filename, normalise=True), 4)
        brenner_score = round(brenner.calculate_metric(img_path=filename, normalise=True), 4)

        _, name = os.path.split(filename)
        base, _ = os.path.splitext(name)

        base_split = base.split("_")

        sample_name = base_split[0]
        sample_name = re.sub(r'[-_]part\d+$', ' ', sample_name)
        fuse_type = base_split[-1]

        if fuse_type == "MAX":
            max_scores.append(
                [
                    sample_name,
                    nglv_score,
                    brenner_score,
                ]
            )
        elif fuse_type == "MIN":
            min_scores.append(
                [
                    sample_name,
                    nglv_score,
                    brenner_score,
                ]
            )
        elif fuse_type == "VAR":
            var_scores.append(
                [
                    sample_name,
                    nglv_score,
                    brenner_score,
                ]
            )
        elif fuse_type == "MEAN":
            mean_scores.append(
                [
                    sample_name,
                    nglv_score,
                    brenner_score,
                ]
            )

    out_filename_max = create_out_filename(args.output, [], [args.method, "max"])
    out_filename_max = append_file_extension(out_filename_max, "csv")

    out_filename_min = create_out_filename(args.output, [], [args.method, "min"])
    out_filename_min = append_file_extension(out_filename_min, "csv")

    out_filename_mean = create_out_filename(args.output, [], [args.method, "mean"])
    out_filename_mean = append_file_extension(out_filename_mean, "csv")

    out_filename_var = create_out_filename(args.output, [], [args.method, "var"])
    out_filename_var = append_file_extension(out_filename_var, "csv")

    columns = [
        "Sample name",
        "NGLV",
        "Brenner",
    ]

    logger.info("Storing evaluation for max images into the file %s", out_filename_max)
    write_scores_to_csv(out_filename_max, columns, max_scores)

    logger.info("Storing evaluation for min images into the file %s", out_filename_min)
    write_scores_to_csv(out_filename_min, columns, min_scores)

    logger.info("Storing evaluation for min images into the file %s", out_filename_mean)
    write_scores_to_csv(out_filename_mean, columns, mean_scores)


if __name__ == "__main__":
    args = parse_args()
    main(args)
