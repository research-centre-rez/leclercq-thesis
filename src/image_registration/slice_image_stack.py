import sys
import os
from matplotlib.pyplot import viridis
import numpy as np
import argparse
import matrix_processing

# use __init__.py instead and run the script from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

# wrap parser into a function to prevent run on module loading
parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')


FRAME_STEP = 200


def process_image_slices(image_stack, frame_step):
    """
    Generate and process gallery of image slices.
    Args:
        image_stack (np.ndarray): The stack of input images.
        frame_step (int): Number of frames per slice.
    Returns:
        dict: Processed gallery of image slices.
    """
    gallery = {}
    # Process slices of the image stack in chunks of `frame_step`
    for start_idx in range(0, image_stack.shape[0], frame_step):
        end_idx = start_idx + frame_step
        if end_idx > image_stack.shape[0]:  # Ensure we don't go out of bounds
            break
        slice_key = f'{start_idx}_to_{end_idx}_frames'
        gallery[slice_key] = matrix_processing.max_image(image_stack[start_idx:end_idx])

    # Process full image stack
    gallery['all_frames'] = matrix_processing.max_image(image_stack)

    # Convert all images to RGB
    for key, value in gallery.items():
        gallery[key] = matrix_processing.gray_to_rgb(value)
    return gallery


def main(args):
    image_stack = np.load(args.input)
    base_filename = args.input.split('/')[-1].split('.')[0]
    gallery = process_image_slices(image_stack, FRAME_STEP)
    output_filename = f'{base_filename}_slices'  # create a utils/filename_builder.py for this purpose
    visualisers.imshow(output_filename, **gallery)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
