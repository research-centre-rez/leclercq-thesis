import os
import sys
from matplotlib.pyplot import imshow
import numpy as np
import buffer_video
import cv2 as cv
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers

def min_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.min(axis=0)

def max_image(video_mat:np.ndarray) -> np.ndarray:
    return video_mat.max(axis=0)

if __name__ == "__main__":
    vid_mat = np.load('calib_vid_rotated.npy')

    min_img = min_image(vid_mat)
    max_img = max_image(vid_mat)
    max_min = max_img - min_img

    cylinder_mask = (min_img > 0).astype(dtype=np.uint8)
    masked_maxmin = cv.bitwise_and(max_min, max_min, mask=cylinder_mask)
    hist, bins = np.histogram(masked_maxmin[masked_maxmin > 0], bins=256, range=(0,256))

    plt.figure()
    plt.title('Histogram of masked Max-Min image')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')
    plt.bar(bins[:-1], hist, width=1, align='center')
    plt.savefig('hist')

    visualisers.imshow('masked maxmin', masked_maxmin=masked_maxmin)

    min_img = cv.cvtColor(min_img, cv.COLOR_GRAY2RGB)
    max_img = cv.cvtColor(max_img, cv.COLOR_GRAY2RGB)
    max_min = cv.cvtColor(max_min, cv.COLOR_GRAY2RGB)

#    visualisers.imshow(title='min_img', min=min_img)
#    visualisers.imshow(title='max_img', max=max_img)
#    visualisers.imshow(title='max-min_img', max_minus_min=max_min)
#    visualisers.imshow(title='min-max-img', min=min_img, max=max_img, max_minus_min=max_min)

