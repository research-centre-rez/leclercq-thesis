import os
import sys
import argparse
import logging
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform

from image_registration.register_matrix import compute_affine_transform
from utils import pprint
from utils.filename_builder import create_out_filename
from utils.loaders import load_npz_disp

def parse_args():
    parser = argparse.ArgumentParser(description='Creates a video from a given image stack. There are options to display the displacement as well and to show the estimated center of rotation.')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', required=True, help='Path to .npz displacement file')
    required.add_argument('-f', '--frame_id', type=int, required=True, help='Which index do you want to check?')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help='Show the affine estimation in a GUI')
    optional.add_argument('--save', default=False, action=argparse.BooleanOptionalAction, help='Whether to save the final graph')
    return parser.parse_args()

def display_affine_for_OF(opt_flow, idx:int, graph_config=None) -> None:

    if graph_config is None:
        graph_config = {
            'title': f'Affine transformation for index {idx}',
            'save_as': None,
            'show': True
        }
    fixed  = opt_flow[:, 0,   :] # Registering to these points
    moving = opt_flow[:, idx, :]

    fixed  = fixed.astype(np.float32)
    moving = moving.astype(np.float32)

    print(fixed.shape) #(18,2)
    print(moving.shape) #(18,2)

    #T = estimate_transform('euclidean', moving, fixed)
    T, _ = cv.estimateAffinePartial2D(moving, fixed, method=cv.RANSAC)

    ones    = np.ones((moving.shape[0], 1))
    mov_hom = np.hstack([moving, ones])
    mov_T   = (T @ mov_hom.T).T

    plt.scatter(fixed[:, 0], fixed[:, 1], label="Original Points", alpha=0.5)
    plt.scatter(moving[:, 0], moving[:, 1], label="Displaced Points", alpha=0.5)
    plt.scatter(mov_T[:, 0], mov_T[:, 1], label="Transformed Points", alpha=0.5)
    plt.plot()
    plt.legend()
    plt.title(graph_config['title'])

    if graph_config['save_as']:
        plt.savefig(graph_config['save_as'])
    if graph_config['show']:
        plt.show()

    plt.close()
def display_affine_for_point(disp:np.ndarray, mesh:np.ndarray, idx:int, graph_config:dict[str,str]) -> None:
    '''
    Displays how keypoints are mapped for a single frame (starting frame vs given frame).
    Args:
        disp (np.ndarray): Displacement matrix
        mesh (np.ndarray): Mesh for the displacement
        idx (int): Which frame id you want to see
        graph_config: Dictionary containing graph config, should contain these keys: ['title', 'save_as', 'show']. You can set 'save_as' as None if you don't want to save the graph.
    Returns:
        None

    '''
    if graph_config is None:
        graph_config = {
            'title': f'Affine transformation for index {idx}',
            'save_as': 'Affine check',
            'show': True
        }

    mesh_time_t = disp[...,idx]
    x_mesh      = mesh_time_t[0].flatten()
    y_mesh      = -mesh_time_t[1].flatten() # y=0 at the top
    mesh_stack  = np.stack([x_mesh, y_mesh], axis=-1)
    mesh_stack  = mesh_stack + mesh

    T = compute_affine_transform(mesh, disp, idx)

    # Apply T
    ones     = np.ones((mesh_stack.shape[0], 1))
    mesh_hom = np.hstack([mesh_stack, ones])
    mesh_tra = (T @ mesh_hom.T).T

    plt.scatter(mesh[:, 0], mesh[:, 1], label="Original Points", alpha=0.5)
    plt.scatter(mesh_stack[:, 0], mesh_stack[:, 1], label="Displaced Points", alpha=0.5)
    plt.scatter(mesh_tra[:, 0], mesh_tra[:, 1], label="Transformed Points", alpha=0.5)
    plt.legend()
    plt.title(graph_config['title'])

    if graph_config['save_as']:
        plt.savefig(graph_config['save_as'])
    if graph_config['show']:
        plt.show()

    plt.close()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    _, name = os.path.split(args.input)
    base, _ = os.path.splitext(name)

    disp, mesh = load_npz_disp(args.input, True)

    if args.save:
        save_as = create_out_filename(f'./images/{base}', [], ['affine', 'check'])
    else:
        save_as = None

    g_config = {
        'title': f'Affine transformation for {base} on index {args.frame_id}',
        'save_as': save_as,
        'show': args.show
    }

    display_affine_for_point(disp, mesh, args.frame_id, g_config) 

if __name__ == "__main__":
    args = parse_args()
    main(args)
