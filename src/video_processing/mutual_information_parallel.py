from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

import numpy as np
from skimage import transform
from scipy.optimize import minimize
from skimage.metrics import normalized_mutual_information
import cv2 as cv
from tqdm import tqdm

def mi_loss(params, fixed, moving):
    angle, tx, ty = params
    tform = transform.AffineTransform(
        rotation=np.deg2rad(angle), translation=(tx,ty)
    )
    moved = transform.warp(moving, tform)
    loss  = normalized_mutual_information(fixed, moved, bins=640)
    return -loss

def align_frame(args):
    fixed, moving = args
    initial_params = [0,0,0]
    result = minimize(mi_loss, initial_params, method='Nelder-Mead', args=(fixed, moving))
    if not result.success:
        raise RuntimeError('Optimisation failed')

    angle, tx, ty = result.x
    tform = transform.AffineTransform(
        rotation=np.deg2rad(angle), translation=(tx,ty)
    )
    aligned = transform.warp(moving, tform)
    return (aligned * 255).astype(np.uint8)

def process_block_parallel(vid_path, start_idx, end_idx, num_workers=4):
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        print('Could not load video')
        return []

    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx - 1)
    num_frames = end_idx - start_idx
    image_stack = []

    ret, fixed = cap.read()
    if not ret:
        print('Error reading video')
        return []
    
    fixed = cv.cvtColor(fixed, cv.COLOR_BGR2GRAY)
    image_stack.append(fixed)

    frames = []
    for _ in range(num_frames-1):
        ret, moving = cap.read()
        if not ret:
            break
        moving = cv.cvtColor(moving, cv.COLOR_BGR2GRAY)
        frames.append(moving)

    cap.release()

    args_list = [(fixed, frame) for frame in frames]

    with tqdm(total=len(args_list), desc='Registering a block') as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(align_frame, args): args for args in args_list}

            for future in concurrent.futures.as_completed(futures):
                try:
                    image_stack.append(future.result())
                except Exception as e:
                    print(f'Error in alignment: {e}')
                pbar.update(1)

    print(f'Finished block {start_idx}-{end_idx}')

    return image_stack
