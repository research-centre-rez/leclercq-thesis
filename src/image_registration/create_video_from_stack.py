import numpy as np
import cv2 as cv
import argparse
import os
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Draws displacement from a given `displacement.npz` file.')

required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input', required=True, help='Path to .npy image stack file')

def main(args):

    base_name = os.path.basename(args.input).split('.npy')[0]
    print(base_name)
    img_stack = np.load(args.input)[15:]
    h,w = img_stack[0].shape

    disp = f'{base_name}_displacement.npz'
    data = np.load(disp)
    mesh = data['mesh_nodes']
    disp = data['displacement'].squeeze()

    save_as = f'{base_name}_video.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(save_as, fourcc, 35, (w,h))

    center_h = h // 2
    center_w = w // 2

    x_disp = disp[0]
    y_disp = disp[1]
    grid_reshaped = mesh.reshape(x_disp.shape[:2] + (2,))
    mesh_x = grid_reshaped[:,:,0]
    mesh_y = grid_reshaped[:,:,1]

    #Calculated centre
    rot_centre_x = 991
    rot_centre_y = 534

    for frame_id in tqdm(range(img_stack.shape[0]), desc='Writing to video'):
        frame_bgr = cv.cvtColor(img_stack[frame_id], cv.COLOR_GRAY2BGR)

        disp_x = x_disp[:,:,frame_id]
        disp_y = y_disp[:,:,frame_id]

        tmp_x = (mesh_x + disp_x).flatten()
        tmp_y = (mesh_y + disp_y).flatten()

        zipped = np.array((tmp_x, tmp_y)).transpose(1,0)

        for pt in zipped:
            frame_bgr = cv.circle(frame_bgr, pt.astype(int), radius=5, color=(0,255,0), thickness=-1)

        frame_bgr = cv.circle(frame_bgr, (center_w, center_h), radius=5, color=(255,0,0), thickness=-1)
        frame_bgr = cv.circle(frame_bgr, (rot_centre_x, rot_centre_y), radius=5, color=(0,0,255), thickness=-1)

        video.write(frame_bgr)

    print('Video successfully created')
    cv.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
