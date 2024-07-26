import os
import numpy as np
import cv2 as cv

def crop_images(img_path, mask_path, output_dir):
    img = cv.imread(img_path)
    mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

    # masks the image such that everything but the mask is turned to 0s
    img_masked = cv.bitwise_and(img, img, mask=mask)

    x,y = np.nonzero(mask)
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()

    # crop the image
    cropped = img_masked[xl : xr+1, yl : yr+1]
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0])
    out_base = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = f'{output_dir}-cropped.jpg'

    print('\n', output_dir)
    cv.imwrite(output_dir, cropped)

def load_images(directory_path):
    output_dir = './imgs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(directory_path):
        if file.endswith('.jpg'):
            print(f'\rprocessing file {file}', end='', flush=False)
            img_path = os.path.join(directory_path, file)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_name = f'{base_name}_mask.png'
            mask_path = os.path.join(directory_path, mask_name)

            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                crop_images(img_path, mask_path, output_dir)
    print('')

load_images('../data/dev/temp-test')
