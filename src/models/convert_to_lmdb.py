import lmdb
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

def convert_to_lmdb(csv_file:str, image_dir:str, mask_dir:str, lmdb_path:str, map_size=1e12):
    """
    Converts images and masks into an LMDB database.

    Args:
        csv_file:  Path to the CSV file containing image and mask pairs
        image_dir: Directory containing images
        mask_dir:  Directory containing masks
        lmdb_path: Path where the LMDB database will be saved
        map_size:  Max size of the LMDB database in bytes
    """

    data = pd.read_csv(csv_file, header=None)
    num_entries = len(data)

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            img_path  = os.path.join(image_dir, row[0])
            mask_path = os.path.join(mask_dir, row[1])

            img  = cv.imread(img_path)
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print(f'Error loading image or mask for {row[0]} or {row[1]}')
                sys.exit(1)

            img_encoded  = cv.imencode('.jpg', img)[1].tobytes()
            mask_encoded = cv.imencode('.png', mask)[1].tobytes()

            img_key  = f'image_{idx}'.encode()
            mask_key = f'mask_{idx}'.encode()

            txn.put(img_key, img_encoded)
            txn.put(mask_key, mask_encoded)

        txn.put(b'__len__', str(num_entries).encode('utf-8'))

    env.close()
    print(f'LMDB dataset saved at {lmdb_path}')

if __name__ == '__main__':
    csv_file  = './dataset/concrete_segmentation_camvid/image_mask_pairs.txt'
    img_dir   = './dataset/concrete_segmentation_camvid/'
    mask_dir  = './dataset/concrete_segmentation_camvid/'
    lmdb_path = 'test_lmdb'

    map_size = 1 * 1024 * 1024 * 1024

    convert_to_lmdb(csv_file, img_dir, mask_dir, lmdb_path, map_size)
