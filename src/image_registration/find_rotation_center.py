import sys
import argparse
import logging
from circle_fit import taubinSVD, plot_data_circle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel

from utils.disp_utils import extract_medians
from utils import pprint
from utils.loaders import load_npz_disp

def parse_args():
    parser = argparse.ArgumentParser(description="Fits a circle / ellipse to the displacement data. The geometric shape is a parameter that the user can specify.")

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # Required arguments
    required.add_argument('-i', '--input', required=True, help='Path to the .npz displacement file')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--fit_ellipse', action=argparse.BooleanOptionalAction, help='Do you want to fit an ellipse to the displacement data?' )
    optional.add_argument('--fit_circle', action=argparse.BooleanOptionalAction, help='Do you want to fit a circle to the displacement data?' )
    optional.add_argument('--show', action=argparse.BooleanOptionalAction, help='Show the circle / ellipse' )
    return parser.parse_args()

# TODO: Add docstring
def plot_ellipse(disp, xc,yc,a,b,theta):
    _, ax = plt.subplots()
    ax.plot(disp[:,0], disp[:,1], 'bo', label='Data')

    # Generate fitted ellipse outline
    fit_t = np.linspace(0, 2*np.pi, 300)
    x_fit = xc + a * np.cos(fit_t) * np.cos(theta) - b * np.sin(fit_t) * np.sin(theta)
    y_fit = yc + a * np.cos(fit_t) * np.sin(theta) + b * np.sin(fit_t) * np.cos(theta)
    ax.plot(x_fit, y_fit, 'r-', label='Fitted ellipse')
    ax.plot(xc,yc, 'o', label='Center of ellipse')

    ax.set_aspect('equal', 'datalim')
    ax.legend()
    plt.show()

def fit_ellipse(disp:np.ndarray, show=False):

    logger = logging.getLogger(__name__)
    model = EllipseModel()
    success = model.estimate(disp)

    if not success:
        logger.error('failed to find an ellipse')
        sys.exit(-1)

    xc, yc, _, _, _ = model.params

    if show:
        plot_ellipse(disp, *model.params)

    return xc,yc

def fit_circle(displacement:np.ndarray, show=False):

    xc, yc, r, _ = taubinSVD(displacement)
    if show:
        plot_data_circle(displacement, xc, yc, r)

    return xc,yc

def main(args):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    disp, _ = load_npz_disp(args.input, squeeze=True)

    med_disp = extract_medians(disp)

    if args.fit_ellipse:
        fit_ellipse(med_disp, True)
    if args.fit_circle:
        fit_circle(med_disp, True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
