import logging
import numpy as np
import cv2 as cv
import argparse
import os
import sys
from tqdm import tqdm

from image_registration.find_circle import find_ellipse
from image_registration.register_matrix import extract_medians
from utils.filename_builder import create_out_filename, append_file_extension
from utils.loaders import load_npz_disp

def parse_args():
    parser = argparse.ArgumentParser(description='Creates a video from a given image stack. There are options to display the displacement as well and to show the estimated center of rotation.')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--input', required=True, help='Path to .npy image stack file')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('--disp_path', default=None, help='Path to the .npz displacement file')
    return parser.parse_args()

def create_video_from_stack(image_stack, disp, mesh, sample_name):
    # TODO: Add docstring
    '''
    ...
    '''
    logger      = logging.getLogger(__name__)
    h,w         = image_stack[0].shape

    show_disp   = (disp is not None)

    if show_disp:
        x_disp = disp[0]
        y_disp = disp[1]

        grid_reshaped = mesh.reshape(x_disp.shape[:2] + (2,))
        mesh_x        = grid_reshaped[:,:,0]
        mesh_y        = grid_reshaped[:,:,1]

        center_h = h // 2
        center_w = w // 2

        rot_centre_x, rot_centre_y = find_ellipse(extract_medians(disp))
        rot_centre_x = int(center_w + rot_centre_x)
        rot_centre_y = int(center_h + rot_centre_y)


    save_as = create_out_filename(f'./videos/{sample_name}', [], ['video'])
    save_as = append_file_extension(save_as, '.mp4')

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video  = cv.VideoWriter(save_as, fourcc, 35, (w,h))

    for frame_id in tqdm(range(image_stack.shape[0]), desc='Writing to video'):
        frame_bgr = cv.cvtColor(image_stack[frame_id], cv.COLOR_GRAY2BGR)

        if show_disp:
            disp_x = x_disp[:,:,frame_id]
            disp_y = y_disp[:,:,frame_id]

            tmp_x = (mesh_x + disp_x).flatten()
            tmp_y = (mesh_y + disp_y).flatten()

            zipped = np.array((tmp_x, tmp_y)).transpose(1,0)

            for pt in zipped:
                frame_bgr = cv.circle(frame_bgr, pt.astype(int), radius=5, color=(0,255,0), thickness=-1)

            frame_bgr = cv.circle(frame_bgr, (rot_centre_x, rot_centre_y), radius=5, color=(0,0,255), thickness=-1)

        video.write(frame_bgr)

    logger.info('Video successfully created')
    cv.destroyAllWindows()
    video.release()


def main(args):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    img_stack = np.load(args.input)[15:]

    if args.disp_path:
        disp, mesh = load_npz_disp(args.disp_path, squeeze=True)
    else:
        disp, mesh = None, None

    _, name = os.path.split(args.input)
    base, _ = os.path.splitext(name)
    create_video_from_stack(img_stack, disp, mesh, base)

if __name__ == "__main__":
    args = parse_args()
    main(args)
