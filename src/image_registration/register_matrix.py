import sys
import os
import numpy as np
import cv2 as cv
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from find_circle import find_circle_for_center, find_ellipse
from utils import visualisers

parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')
required.add_argument('-d', '--displacement', type=str, required=True, help='Path to the displacement matrix')

optional.add_argument('-o', '--save_as', type=str, default=None, help='Name of the saved displacement matrix')
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

def extract_medians(displacement:np.ndarray):
    medians = []
    for i in range(displacement.shape[-1]):
        x_med = np.median(displacement[0, :, :, i])
        y_med = np.median(displacement[1, :, :, i])
        medians.append([x_med, y_med])

    meds = np.array(medians)

    return meds

def extract_means(displacement:np.ndarray):
    means = []
    for i in range(displacement.shape[-1]):
        x_mean = displacement[0, :, :, i].mean()
        y_mean = displacement[1, :, :, i].mean()
        #x_med = np.median(displacement[0, :, :, i])
        #y_med = np.median(displacement[1, :, :, i])
        means.append([x_mean, y_mean])
        #means.append([x_med, y_med])

    means = np.array(means)

    return means

def draw_means(means):
    plt.plot(means[:,0], means[:,1])
    plt.show()

def shift_by_vector(image_stack, displacement, mesh):
    n, h, w = image_stack.shape

    x_c, y_c = find_ellipse(displacement)
    for i in tqdm(range(n)):
        image = image_stack[i]
        x_d, y_d = displacement[i]
        #x_d = x_c - x_d
        #y_d = y_c - y_d

        T = np.array([[1, 0, -x_d],
                      [0, 1, -y_d]])

        im_translated = cv.warpAffine(image, T, (w,h))
        image_stack[i] = im_translated

    return image_stack

def compute_affine_transform(mesh_nodes:np.ndarray, disp:np.ndarray, frame_idx:int):
    src_pts = mesh_nodes.copy()

    dx = -disp[0, :, :, frame_idx].flatten()
    dy = -disp[1, :, :, frame_idx].flatten()

    dst_pts = np.stack([src_pts[:,0] + dx, src_pts[:,1] + dy], axis=-1)

    A, _ = cv.estimateAffine2D(src_pts, dst_pts, method=cv.RANSAC)
    return A

def apply_transformations(video_frames:np.ndarray, mesh_nodes:np.ndarray, disp:np.ndarray):
    n,h,w = video_frames.shape
    for frame_id in tqdm(range(n), desc='Registering'):
        affine_mat = compute_affine_transform(mesh_nodes, disp, frame_id)
        if affine_mat is not None:
            transformed = cv.warpAffine(video_frames[frame_id], affine_mat, (w, h))
        else:
            print('Failed')
            sys.exit()
        video_frames[frame_id] = transformed

    return video_frames


def main(args):
    img_stack = np.load(args.input)[15:]
    data = np.load(args.displacement)
    displacement = data['displacement']
    mesh_nodes = data['mesh_nodes']
    displacement = displacement.squeeze()
    if (img_stack.shape[0] != displacement.shape[-1]):
        print('Img stack shape does not match displacement shape')
        print(f'Number of images: {img_stack.shape[0]} should be the same as the number of displacements: {displacement.shape[-1]}')
        sys.exit()

    #means = extract_means(displacement)
    meds = extract_medians(displacement)
    base_name = args.input.split('/')[-1].split('.')[0]
    #img_stack = apply_transformations(img_stack, mesh_nodes, displacement)

    img_stack = shift_by_vector(img_stack, meds, mesh_nodes)

    #img_stack = register_image_stack(img_stack, displacement)

    if args.save_as is None:
        base_name = args.input.split('/')[-1].split('.')[0]
        save_as = f'{base_name}_registered'
    else:
        save_as = args.save_as
    np.save(save_as, img_stack)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

