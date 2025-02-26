import numpy as np
import cv2 as cv
import argparse
import os
import sys
from tqdm import tqdm

# Pack this into if __name__ == "__main__": to prevent running on module import
parser = argparse.ArgumentParser(
    description="""
        This script converts stack of images stored in {input} file into a video.
        Video is augmented with circles showing the calculated displacement (source is in {input}_displacement.npz 
        in the same folder). Displacement file is a product of previous step of image registration (@see ...).
        
        TODO: describe format of the video
        TODO: describe hardcoded values of the centre of the rotation or create input parameter for that
    """
)

required = parser.add_argument_group('required arguments')
# Describe structure of the npy file in the help message
# There is missing info that you expect "_displacement.npz", link source of the file here.
required.add_argument('-i', '--input', required=True, help='Path to .npy image stack file')

def main(args):
    # You build a video, please add info here about that.
    # Split the method to reading function and writing function

    base_name = os.path.basename(args.input)[:-4]
    # Use logger if you need this in production quality code or remove
    print(base_name)
    img_stack = np.load(args.input)[15:]  # maybe this "[15:]" should be parameter of the function/input parameter
    h,w = img_stack[0].shape

    disp = f'{base_name}_displacement.npz'  # put this into some config file, describe structure of the *.npz file
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
    rot_centre_x = 991  # This should be in a config file or somewhere close to import section
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
