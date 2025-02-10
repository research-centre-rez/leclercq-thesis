import sys
import os
import numpy as np
import cv2 as cv
from scipy.interpolate import griddata
from tqdm import tqdm
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualisers
import matrix_processing

def apply_displacement(image:np.ndarray, displacement:np.ndarray):
    h, w = image.shape[:2]

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

def torch_displacement(image_stack, displacement):

    n,h,w = image_stack.shape

    # Create normalized pixel grid
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(n, 1, 1, 1).cuda()

    # Normalize displacement
    disp_x = F.interpolate(displacement[0, 0], size=(h, w), mode='bilinear', align_corners=False)
    disp_y = F.interpolate(displacement[0, 1], size=(h, w), mode='bilinear', align_corners=False)

    # Convert displacement to normalized coordinates
    disp_x = disp_x.permute(2, 3, 0, 1).squeeze(-1) / (w / 2)
    disp_y = disp_y.permute(2, 3, 0, 1).squeeze(-1) / (h / 2)

    # Apply displacement
    warped_grid = grid + torch.stack((disp_x, disp_y), dim=-1)
    warped_images = F.grid_sample(image_stack.unsqueeze(1).float().cuda(), warped_grid, mode='bilinear', align_corners=False)

    return warped_images.squeeze(1).cpu().numpy()


if __name__ == "__main__":
    img_stack = np.load('1A-part1_rotated.npy')[15:]
    img_stack = torch.tensor(img_stack).cuda()
    displacement = np.load('1A_disp.npy')
    displacement = torch.tensor(displacement).cuda()

    print(img_stack.shape)
    print(displacement.shape)
    if (img_stack.shape[0] != displacement.shape[-1]):
        print('Img stack shape does not match displacement shape')
        print(f'Number of images: {img_stack.shape[0]} should be the same as the number of displacements: {displacement.shape[-1]}')
        sys.exit()
    #output = np.zeros_like(img_stack)
    print('Doing calculations..')
    output = torch_displacement(img_stack, displacement)
#    for i in tqdm(range(displacement.shape[-1]), desc='Registering images'):
#
#        output[i] = apply_displacement(img_stack[i], displacement[...,i])

    np.save('reg_out', output)


