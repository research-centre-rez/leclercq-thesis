import os
import cv2 as cv
import numpy as np
from image_registration.matrix_processing import max_image
from utils.filename_builder import create_out_filename, append_file_extension

# TODO: Make this into a proper file or delete
def show_maxs_from_image_stacks(dir_name, out):
    npy_files = os.listdir(dir_name)

    files = [os.path.join(dir_name, file) for file in npy_files]

    for i, file in enumerate(files):
        img_stack = np.load(file)
        img = max_image(img_stack)
        out = f'{out}_{i}'
        out = append_file_extension(out, '.png')
        cv.imwrite(out, img)
