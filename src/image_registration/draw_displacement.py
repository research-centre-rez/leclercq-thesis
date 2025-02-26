import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from tqdm import tqdm

# FIX: this is typically solved by __init__.py and calling scripts from the root folder.
# No need to do this:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

# Again wrap this paragraph into if __name__ == "__main__": or another function to prevent run on file import
parser = argparse.ArgumentParser(
    description="""
        Draws displacement from a given `displacement.npy` file. 
        Note: The file thats passed into this should contain the calculated displacement!!
    """
)
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument(
    '-i',
    '--input',
    required=True,
    help='Path to the .npz file that contains the calculated displacements'
)
optional = parser.add_argument_group('optional arguments')
optional.add_argument(
    '--show',
    default=False,
    action=argparse.BooleanOptionalAction,
    help='Show the displacement in a GUI'
)
optional.add_argument(
    '--save',
    default=False,
    action=argparse.BooleanOptionalAction,
    help='Whether to save the final graph'
)

# Do you need types here? Maybe better to use function doc and describe those fields (see suggestion)
def draw_displacement(displacement:np.ndarray, grid:np.ndarray, sample_name, show, save):
    """
        Draw displacement graph for a given set of displacements over a grid.

        This function generates a graphical representation of the displacement
        of points on a grid over multiple frames. It computes and plots the
        positions at each frame based on the displacement provided and displays
        or saves the resulting graph. The function highlights the evolution
        of displacement over time and provides the option to visualize
        displacements inline or save them to a file.

        Parameters:
            displacement (np.ndarray): A 3D numpy array containing the displacement
                values for X and Y axes in the last dimension over multiple frames.
            grid (np.ndarray): A 2D numpy array defining the initial grid points
                before displacement, with shape (n_points, 2).
            sample_name: A string representing the name of the sample being visualized.
            show: A boolean flag to determine whether to display the plot inline or not.
            save: A boolean flag indicating whether to save the plot as a file.

        Returns:
            None

        Raises:
            None
    """
    #displacement = np.load(args.input)

    x_disp = displacement[0,0]
    y_disp = displacement[0,1]
    
    grid_reshaped = grid.reshape(x_disp.shape[:2] + (2,))
    mesh_x = grid_reshaped[:,:,0]
    mesh_y = grid_reshaped[:,:,1]

    # TODO: remove obsolete/debugging code. If you need it create a function for that purpose and name it as debugging.
    #mesh_x.shape => (6,6)
    #mesh_y.shape => (6,6)
    #grid.shape => (36,2)
    #x_disp.shape => (6,6, 1316)

    num_frames = x_disp.shape[-1]
    cmap = plt.get_cmap('viridis', num_frames)

    # FIXME: Use logger or remove in production quality code.
    print(f'Max displacement over x axis: {np.diff(x_disp, axis=-1).max()}')
    print(f'Max displacement over y axis: {np.diff(y_disp, axis=-1).max()}')

    _, ax = plt.subplots()

    norm = mcolors.Normalize(vmin=0, vmax=num_frames-1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i in tqdm(range(x_disp.shape[-1]), desc='Plotting displacement'):
        disp_x = x_disp[:, :, i]
        disp_y = y_disp[:, :, i]

        tmp_x = mesh_x + disp_x
        tmp_y = mesh_y - disp_y#since muDIC has y=0 at the top

        plt.plot(tmp_x, tmp_y, 'o', color=cmap(i), linestyle='none', label=f'Step {i}')

        if i > 0:
            prev_x = mesh_x + x_disp[:,:, i-1]
            prev_y = mesh_y - y_disp[:,:, i-1]#since muDIC has y=0 at the top

            for j in range(tmp_x.shape[0]):
                for k in range(tmp_x.shape[1]):
                    plt.arrow(prev_x[j, k], prev_y[j, k],
                              tmp_x[j, k] - prev_x[j, k], tmp_y[j, k] - prev_y[j, k],
                              color=cmap(i), head_width=0.03, head_length=0.06, alpha=0.7)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frame number")

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Disp over {x_disp.shape[-1]} frames for {sample_name}")

    if save:
        save_as = f'./images/{sample_name}_displacement_graph'
        plt.savefig(save_as)
    if show:
        plt.show()
    # I think that you want to close it in every scenario
    plt.close()

def main(args):
    # use logger
    print('Running with the following parameters:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')
    data = np.load(args.input)
    displacement = data['displacement']
    mesh_nodes   = data['mesh_nodes']
    base_name = os.path.basename(args.input).split('_displacement')[0]

    draw_displacement(displacement, mesh_nodes, base_name, args.show, args.save)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
