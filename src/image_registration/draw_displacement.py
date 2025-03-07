import logging
import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from tqdm import tqdm

from utils import filename_builder, pprint

def parse_args():
    parser = argparse.ArgumentParser(description='Draws displacement from a given `displacement.npy` file. Note: The file thats passed into this should contain the calculated displacement!!')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', required=True, help='Path to the .npz file that contains the calculated displacements')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help='Show the displacement in a GUI')
    optional.add_argument('--save', default=False, action=argparse.BooleanOptionalAction, help='Whether to save the final graph')
    optional.add_argument('--first_n', type=int, help='Whether to save the final graph')
    return parser.parse_args()

def draw_displacement(displacement:np.ndarray, grid:np.ndarray, sample_name:str, show:bool, save:bool) -> None:
    '''
    Draws a displacement graph for the sample. To make the graph easier for reading, frame index is used to colour the points with a `viridis` heatmat.

    Args:
        displacement (np.ndarray): Displacement matrix that is created by `mudic_correlation`
        grid (np.ndarray): Grid that determines spaces between displacement points, also created by `mudic_correlation`
        sample_name (str): Name of the sample, will be displayed in the graph title
        show (bool): Whether the graph should be displayed
        save (bool): Whether the graph should be saved

    Returns:
        None
    '''

    logger = logging.getLogger(__name__)
    x_disp = displacement[0,0]
    y_disp = displacement[0,1]
    
    grid_reshaped = grid.reshape(x_disp.shape[:2] + (2,))
    mesh_x        = grid_reshaped[:,:,0]
    mesh_y        = grid_reshaped[:,:,1]

    num_frames = x_disp.shape[-1]
    cmap       = plt.get_cmap('viridis', num_frames)

    logger.info(f'Max displacement over x axis: {np.diff(x_disp, axis=-1).max()}')
    logger.info(f'Max displacement over y axis: {np.diff(y_disp, axis=-1).max()}')

    _, ax = plt.subplots()

    norm = mcolors.Normalize(vmin=0, vmax=num_frames-1)
    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in tqdm(range(x_disp.shape[-1]), desc='Plotting displacement'):
        disp_x = x_disp[:, :, i]
        disp_y = y_disp[:, :, i]

        tmp_x = mesh_x + disp_x
        tmp_y = mesh_y - disp_y # y=0 at the top

        plt.plot(tmp_x, tmp_y, 'o', color=cmap(i), linestyle='none', label=f'Step {i}')

        if i > 0:
            prev_x = mesh_x + x_disp[:,:, i-1]
            prev_y = mesh_y - y_disp[:,:, i-1] # y=0 at the top

            for j in range(tmp_x.shape[0]):
                for k in range(tmp_x.shape[1]):
                    plt.arrow(prev_x[j, k], prev_y[j, k],
                              tmp_x[j, k] - prev_x[j, k], tmp_y[j, k] - prev_y[j, k],
                              color=cmap(i), head_width=0.03, head_length=0.06, alpha=0.7)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frame number")

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title(f"Disp over {x_disp.shape[-1]} frames for {sample_name}")

    if save:
        save_as = filename_builder.create_out_filename(f'./images/{sample_name}', [], ['displacement', 'graph'])
        plt.savefig(save_as)
    if show:
        plt.show()

    plt.close()

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    data         = np.load(args.input)
    displacement = data['displacement']
    mesh_nodes   = data['mesh_nodes']
    base_name    = os.path.basename(args.input).split('_displacement')[0]

    if args.first_n:
        displacement = displacement[...,:args.first_n]

    draw_displacement(displacement, mesh_nodes, base_name, args.show, args.save)

if __name__ == "__main__":
    args = parse_args()
    main(args)
