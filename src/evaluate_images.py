import argparse
import logging
import os
import sys
import csv
import re
from collections import defaultdict
from datetime import datetime

from image_evaluation import (
    NGLV,
    BrennerMethod,
)
from tqdm import tqdm
from image_evaluation.metrics import normType
from utils import pprint
from utils.filename_builder import append_file_extension, create_out_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fused images using NGLV and Brenner metrics. "
            "Creates separate CSV files per fusion method (MAX, MIN, VAR, MEAN)."
            "We assume that fused images were created with fuse_video_stack.py"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")

    required.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Paths to fused image(s). Example: -i ../data/images/*.png",
    )
    required.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Base name of the output CSV files (fusion type is auto-appended). If you pass in 'auto' then the program will automatically create an output filename.",
    )
    required.add_argument(
        "-n",
        "--normalisation_type",
        type=str,
        required=True,
        choices=["std", "minmax", "equalHist", "l2", "grad_mag"],
        help=(
            "What image normalisation type would you like to use? The options are:"
            "std: (image - mean(image)) / (std(image))"
            "minmax: (image - image.min) / (image.max - image.min)"
            "equalHist: cv.equalizeHist(image)"
            "l2: image - (np.linalg.norm(image) + 1e-8)"
            "grad_mag: gradient normalisation with the use of Sobel filters"
        ),
    )
    return parser.parse_args()


def write_scores_to_csv(csv_filename, columns: list[str], data: list):
    with open(csv_filename, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(columns)
        csv_writer.writerows(data)


def extract_sample_and_fuse_type(filename):
    name = os.path.basename(filename)
    base, _ = os.path.splitext(name)
    base_split = base.split("_")

    sample_name = re.sub(r"[-_]part\d+$", " ", base_split[0])
    fuse_type = base_split[-1].upper()

    return sample_name, fuse_type


def main(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

    logger = logging.getLogger(__name__)
    pprint.log_argparse(args)

    metric_functions = {"NGLV": NGLV(), "Brenner": BrennerMethod()}

    results = defaultdict(list)
    valid_fuse_types = ["MAX", "MIN", "VAR", "MEAN"]
    normalisation_type = normType[args.normalisation_type]

    # Store the file as date-month_hour-minute-second_evaluation
    if args.output.lower() == "auto":

        timestamp = datetime.now().strftime("%d-%m_%H-%M-%S")
        args.output = f"./evaluation_runs/{timestamp}_evaluation"
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for filename in tqdm(args.input, desc="Evaluating image"):
        sample_name, fuse_type = extract_sample_and_fuse_type(filename)

        if fuse_type not in valid_fuse_types:
            logger.warning("Skipping unknown fusion type: %s", fuse_type)

        nglv_score = round(
            metric_functions["NGLV"].calculate_metric(
                filename, normalise=True, normalisationType=normalisation_type
            ),
            4,
        )
        brenner_score = round(
            metric_functions["Brenner"].calculate_metric(
                filename, normalise=True, normalisationType=normalisation_type
            ),
            4,
        )

        results[fuse_type].append([sample_name, nglv_score, brenner_score])

    columns = ["Sample name", "NGLV", "Brenner"]
    base_output, _ = os.path.splitext(args.output)
    for fuse_type in sorted(valid_fuse_types):
        score_data = results[fuse_type]
        if not score_data:
            continue

        out_filename = create_out_filename(base_output, [], [fuse_type])
        out_filename = append_file_extension(out_filename, "csv")

        logger.info("Storing evaluation for %s images into %s", fuse_type, out_filename)
        write_scores_to_csv(out_filename, columns, score_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)
