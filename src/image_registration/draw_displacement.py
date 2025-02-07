import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

parser = argparse.ArgumentParser(description='Draws displacement from a given `displacement.npy` file. Note: The file thats passed into this should contain the calculated displacement!!')

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input', type=str, required=True, help='Path to the .npy file that contains the calculated displacements')
required.add_argument('-o', '--output', type=str, required=True, help='Name of the output file')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help='Show the displacement in a GUI')

def main(args):
    print('Running with the following parameters:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')
    displacement = np.load(args.input)

    x_disp = displacement[0,0]
    y_disp = displacement[0,1]

    length = x_disp.shape[0] + 1
    height = y_disp.shape[0] + 1
    mesh_x, mesh_y = np.meshgrid(np.arange(1,length), np.arange(1,height))

    num_frames = x_disp.shape[-1]
    cmap = plt.get_cmap('viridis', num_frames)

    print(f'Max displacement over x axis: {x_disp.max()}')
    print(f'Max displacement over y axis: {y_disp.max()}')

    fig, ax = plt.subplots()

    #plt.tight_layout()

    norm = mcolors.Normalize(vmin=0, vmax=num_frames-1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    for i in range(x_disp.shape[-1]):
        disp_x = x_disp[:, :, i]
        disp_y = y_disp[:, :, i]

        tmp_x = mesh_x + disp_x
        tmp_y = mesh_y + disp_y

        plt.plot(tmp_x, tmp_y, 'o', color=cmap(i), linestyle='none', label=f'Step {i}')

        if i > 0:
            prev_x = mesh_x + x_disp[:,:, i-1]
            prev_y = mesh_y + y_disp[:,:, i-1]

            for j in range(tmp_x.shape[0]):
                for k in range(tmp_x.shape[1]):
                    plt.arrow(prev_x[j, k], prev_y[j, k],
                              tmp_x[j, k] - prev_x[j, k], tmp_y[j, k] - prev_y[j, k],
                              color=cmap(i), head_width=0.03, head_length=0.06, alpha=0.7)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Frame number")


    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(f"Displacement visualisation over first {x_disp.shape[-1]} frames")

    plt.savefig(args.output)
    if args.show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
