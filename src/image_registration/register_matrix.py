import logging
import sys
import os
import numpy as np
import cv2 as cv
from scipy.interpolate import griddata
from multiprocessing import Pool, cpu_count
import argparse
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform
from tqdm import tqdm

from image_registration.disp_utils import extract_medians, extract_means
from image_registration.find_rotation_center import fit_ellipse
from utils import pprint
from utils.filename_builder import create_out_filename
from utils.loaders import load_npz_disp

def parse_args():
    parser = argparse.ArgumentParser(description='Estimating correlation between individual frames of the video matrix')

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Required arguments
    required.add_argument('-i', '--input', type=str, required=True, help='Path to the input video, can be .npy file or .mp4')
    required.add_argument('-d', '--displacement', type=str, required=True, help='Path to the displacement matrix')

    optional.add_argument('-o', '--save_as', type=str, default=None, help='Name of the saved displacement matrix')
    parser._action_groups.append(optional)
    return parser.parse_args()

def apply_displacement(image:np.ndarray, displacement:np.ndarray):
    '''
    Very slow, very bad function that applies a displacement map to an image. Not used currently
    '''
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

def _process_image(args):
    i, img, disp = args
    return i, apply_displacement(img, disp)

def _register_image_stack(img_stack, displacement):
    num_processes = cpu_count() // 2
    args = [(i, img_stack[i], displacement[..., i]) for i in range(img_stack.shape[0])]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(_process_image,args), total=img_stack.shape[0]))

    for i, result in results:
        img_stack[i] = result

    return img_stack

def draw_means(means):
    '''
    Plots means/medians on a graph
    '''
    plt.plot(means[:,0], means[:,1])
    plt.xlabel('X displacement (px)')
    plt.ylabel('Y displacement (px)')
    plt.title('Mean displacement')
    plt.show()

def shift_by_vector(image_stack, displacement, mesh):
    n, h, w = image_stack.shape

    x_c, y_c = fit_ellipse(displacement) 
    
    for i in tqdm(range(n), desc='Registering by shift'):
        image = image_stack[i]
        x_d, y_d = displacement[i]

        T = np.array([[1, 0, -x_d + x_c],
                      [0, 1, -y_d + y_c]])

        im_translated = cv.warpAffine(image, T, (w,h))
        image_stack[i] = im_translated

    return image_stack

def compute_affine_transform(mesh_nodes:np.ndarray, disp:np.ndarray, frame_idx:int):
    logger = logging.getLogger(__name__)
    src_pts = mesh_nodes.copy()

    dx = disp[0, :, :, frame_idx].flatten()
    dy = disp[1, :, :, frame_idx].flatten()

    dst_pts = np.stack([src_pts[:,0] + dx, src_pts[:,1] - dy], axis=-1)

    logger.debug(f'source pts:\n {src_pts}')
    logger.debug(f'------------')
    logger.debug(f'dest pts:\n {dst_pts}')

    #A, _ = cv.estimateAffinePartial2D(dst_pts, src_pts, method=cv.RANSAC)
    #return A
    A = estimate_transform('euclidean', dst_pts, src_pts)
    logger.debug('A: \n %s',A.params[:2])
    logger.debug('A shape: %s',A.params[:2].shape)

    return A.params[:2]

def apply_transformations(video_frames:np.ndarray, mesh_nodes:np.ndarray, disp:np.ndarray):
    n,h,w = video_frames.shape
    for frame_id in tqdm(range(n), desc='Reg via affine'):
        affine_mat = compute_affine_transform(mesh_nodes, disp, frame_id)
        if affine_mat is not None:
            transformed = cv.warpAffine(video_frames[frame_id], affine_mat, (w, h))
        else:
            print('Failed')
            sys.exit()
        video_frames[frame_id] = transformed

    return video_frames

def main(args):
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint.pprint_argparse(args, logger)

    img_stack = np.load(args.input)[15:]
    displacement, mesh_nodes = load_npz_disp(args.displacement, squeeze=True)

    if (img_stack.shape[0] != displacement.shape[-1]):
        logging.error('Img stack shape does not match displacement shape')
        logging.error(f'Number of images: {img_stack.shape[0]} should be the same as the number of displacements: {displacement.shape[-1]}')
        sys.exit()

    meds = extract_medians(displacement)

    _, name = os.path.split(args.input)
    base_name, _ = os.path.splitext(name)

    img_stack = shift_by_vector(img_stack, meds, mesh_nodes)
    #img_stack = apply_transformations(img_stack, mesh_nodes, displacement)

    if args.save_as is None:
        save_to = create_out_filename(f'./npy_files/{base_name}', [], ['registered'])
    else:
        save_to = args.save_as
    np.save(save_to, img_stack)

if __name__ == "__main__":
    args = parse_args()
    main(args)

