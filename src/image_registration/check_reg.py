import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import compare_images

from utils.visualisers import imshow

img_stack = np.load('./npy_files/temp_3A-part0_rotated_short_registered.npy')

frame_A = img_stack[0]
frame_B = img_stack[450]


crop_x = slice(360, 860)
crop_y = slice(350, 850)

markers = np.array([[133, 392], [132, 395]])
center  = np.array([930.46740187, 532.53527425])

p1 = np.array([crop_x.start + markers[0, 0], crop_y.start + markers[0, 1]]) - center
p2 = np.array([crop_x.start + markers[1, 0], crop_y.start + markers[1, 1]]) - center

err = np.rad2deg(
    np.arccos(np.matmul(p1,p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
) / 350
print(f'Error when doing - center {err}')

p1 = np.array([crop_x.start + markers[0, 0], crop_y.start + markers[0, 1]]) + center
p2 = np.array([crop_x.start + markers[1, 0], crop_y.start + markers[1, 1]]) + center

err = np.rad2deg(
    np.arccos(np.matmul(p1,p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
) / 450
print(f'Error when doing + center {err}')


ax = plt.subplot(1,2,1)
ax.imshow(frame_A[crop_y, crop_x], cmap='gray')
ax.scatter(markers[:,0], markers[:,1], marker='+', color='red')
ax = plt.subplot(1,2,2)
ax.imshow(frame_B[crop_y, crop_x], cmap='gray')
ax.scatter(markers[:,0], markers[:,1], marker='+', color='red')
plt.tight_layout()
plt.show()

frame_A = img_stack[1]
frame_B = img_stack[-1]

checker_img = compare_images(frame_A, frame_B, method='checkerboard', n_tiles=(8,16))
int_img     = (checker_img * 255).astype(np.uint8)
colored_img = cv.cvtColor(int_img, cv.COLOR_GRAY2RGB)
imshow('reg_test', checkerboard_test=colored_img)

