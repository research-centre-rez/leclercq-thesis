import sys
import os
from matplotlib.pyplot import viridis
import numpy as np
import argparse
import matrix_processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers


parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')

def main(args):
    img_stack = np.load(args.input)

    increment = 200
    gallery = {}
    for idx in range(increment,img_stack.shape[0],increment):
        gallery[f'{idx-increment}_to_{idx}_frames'] = matrix_processing.max_image(img_stack[idx-increment:idx])

    gallery['all_frames'] = matrix_processing.max_image(img_stack)

    for key, value in gallery.items():
        gallery[key] = matrix_processing.gray_to_rgb(value)

    base_name = args.input.split('/')[-1].split('.')[0]
    save_as = f'{base_name}_slices'
    visualisers.imshow(save_as, **gallery)

if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
