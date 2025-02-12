import sys
import os
import numpy as np
import cv2 as cv
from scipy.interpolate import griddata
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')
required.add_argument('-d', '--displacement', type=str, required=True, help='Path to the displacement matrix')

optional.add_argument('-o', '--save_as', type=str, default='registered_out', help='Name of the saved displacement matrix')
parser._action_groups.append(optional)

def apply_displacement(image:np.ndarray, displacement:np.ndarray):
    h, w = image.shape[-2:]

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    grid_x = np.linspace(0, w, displacement.shape[2])
    grid_y = np.linspace(0, h, displacement.shape[3])


    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    disp_x = griddata((grid_xx.ravel(), grid_yy.ravel()),
                      displacement[0,0].ravel(),
                      (x,y), method='cubic')
    disp_y = griddata((grid_xx.ravel(), grid_yy.ravel()),
                      displacement[0,1].ravel(),
                      (x,y), method='cubic')

    map_x = np.clip(x + disp_x, 0, w-1).astype(np.float32)
    map_y = np.clip(y + disp_y, 0, h-1).astype(np.float32)

    warped_image = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR)

    return warped_image

def process_image(args):
    i, img, disp = args
    return i, apply_displacement(img, disp)

def register_image_stack(img_stack, displacement):
    num_processes = cpu_count() // 2
    args = [(i, img_stack[i], displacement[..., i]) for i in range(img_stack.shape[0])]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image,args), total=img_stack.shape[0]))

    for i, result in results:
        img_stack[i] = result

    return img_stack



def main(args):
    img_stack = np.load(args.input, mmap_mode='r+')[15:]
    displacement = np.load(args.displacement, mmap_mode='r')
    if (img_stack.shape[0] != displacement.shape[-1]):
        print('Img stack shape does not match displacement shape')
        print(f'Number of images: {img_stack.shape[0]} should be the same as the number of displacements: {displacement.shape[-1]}')
        sys.exit()

    img_stack = register_image_stack(img_stack, displacement)

    np.save(f'{args.save_as}', img_stack)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

