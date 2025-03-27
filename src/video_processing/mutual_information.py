import sys
import numpy as np
from skimage import transform, io
from scipy.optimize import minimize
from skimage.metrics import normalized_mutual_information
import cv2 as cv
from tqdm import tqdm
from image_registration.matrix_processing import max_image

from video_processing.mutual_information_parallel import process_block_parallel

def mi_loss(params, fixed, moving):
    angle, tx, ty = params
    tform = transform.AffineTransform(
        rotation=np.deg2rad(angle), translation=(tx,ty)
    )
    moved = transform.warp(moving, tform)
    loss  = normalized_mutual_information(fixed, moved, bins=640)
    return -loss

def process_block(video_path, start_idx, end_idx):

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print('Could not load video')
        sys.exit(-1)

    cap.set(cv.CAP_PROP_POS_FRAMES, start_idx-1)

    num_frames  = end_idx - start_idx
    image_stack = []

    ret, fixed = cap.read()
    if not ret:
        print('Error when reading video')
        sys.exit(-2)

    fixed = cv.cvtColor(fixed, cv.COLOR_BGR2GRAY)

    image_stack.append(fixed)

    with tqdm(desc='Calculating MI', total=num_frames) as pbar:
        pbar.update(1) # Since first frame has been processed
        initial_params = [0,0,0]
        i = 1
        while True:
            ret, moving = cap.read()
            if not ret or i >= end_idx:
                break
            moving = cv.cvtColor(moving, cv.COLOR_BGR2GRAY)

            result = minimize(mi_loss,
                              initial_params,
                              method='Nelder-Mead',
                              args=(fixed, moving))
            angle, tx, ty = result.x
            if not result.success:
                print('Error')
                sys.exit(-2)
            pbar.set_postfix({'success': f'{result.success}'})
            tform = transform.AffineTransform(
                rotation=np.deg2rad(angle), translation=(tx,ty)
            )
            aligned_image  = transform.warp(moving, tform)
            aligned_image  = (aligned_image * 255).astype(np.uint8)
            image_stack.append(aligned_image)
            pbar.update(1)
            i += 1
    print(f'Finished block {start_idx}-{end_idx}')
    cap.release()
    return image_stack

def create_blocks(video_path, block_size, start_at=15):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video")
        sys.exit(-1)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f'Found a total of {total_frames} frames')
    block_ids = [(idx, min(total_frames, idx + block_size)) for idx in range(start_at,total_frames, block_size)]
    return block_ids

def process_blocks(video_path, block_ids:list[tuple]):
    for start, end in tqdm(block_ids, desc='Processing blocks'):
        img_stack = process_block(video_path, start, end)
        np.save('tmp_test.npy', img_stack)
        sys.exit()


# TODO: Add argparse functionalities

def main():
    vid_path   = './videos/test_video_cropped_video.mp4'
    block_size = 150
    blocks     = create_blocks(vid_path, block_size)
    #process_blocks(vid_path, blocks)
    for i, (start, end) in enumerate(blocks):
        image_stack = process_block_parallel(vid_path, start, end, 10)
        np.save(f'./npy_files/block_test/block_{i}', image_stack)

if __name__ == "__main__":
    main()
