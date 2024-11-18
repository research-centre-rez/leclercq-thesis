import os

from utils import visualisers
from utils import loaders
import torch
import cv2 as cv
from utils import visualisers
import argparse
from tqdm import tqdm 

argparser = argparse.ArgumentParser()
argparser.add_argument('--dir_path', default='../datasets/concrete_crack_segmentation', type=str)
argparser.add_argument('--model_path', default='../weights/unet_efficientnet-b7', type=str)
argparser.add_argument('--resize', default=416*5, type=int)
argparser.add_argument('--t', default=0.65, type=float)
argparser.add_argument('--patch_size', default=416, type=int)

# from https://stackoverflow.com/a/76982095
def image_to_patches(image:torch.Tensor, res_patch:int):
    '''
    Given a single [1, C, H, W] image, patch it up into patches of size `res_patch`.
    Args:
        image: torch tensor of shape [1, C, H, W]
        res_patch: desired shape of the patch 
    Returns:
        patches: all patches in a single tensor with shape: [nb_patches, C, H // res_patch, W // res_patch]
    '''

    image       = image.permute(0, 2, 3, 1)
    N, H, W, C  = image.shape
    patch_width = res_patch
    n_rows      = H // patch_width
    n_cols      = W // patch_width


    cropped_img = image[:,:n_rows * patch_width, :n_cols * patch_width, :]
    patches     = torch.empty(N, n_rows, n_cols, patch_width, patch_width, C).to(image.dtype)

    for chan in range(C):
        patches[..., chan] = (
            cropped_img[..., chan]
            .reshape(N, n_rows, patch_width, n_cols, patch_width)
            .permute(0, 1, 3, 2, 4)
        )

    patches = patches.view(N, -1, patch_width, patch_width, C)
    patches = patches.permute(0, 1, -1, 2, 3)
    return patches


def patches_to_image(patches, res_patch, n_rows, n_cols):
    '''
    Merges patches together into a single image.
    '''
    nb_patches, C, H_patch, W_patch = patches.shape
    assert H_patch == res_patch and W_patch == res_patch, "Patch size must match res_patch"
    assert nb_patches == n_rows * n_cols, "Number of patches must match n_rows * n_cols"

    patches = patches.view(n_rows, n_cols, C, res_patch, res_patch)
    image = patches.permute(2,0,3,1,4).contiguous().view(
        C, n_rows * res_patch, n_cols * res_patch
    )

    return image

def main(args):
    dir_path   = args.dir_path
    model_path = args.model_path
    res        = args.resize
    threshold  = args.t
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_s    = args.patch_size

    mask_path   = os.path.join(dir_path, 'masks')
    imgs_path   = os.path.join(dir_path, 'images')
    sample_path = os.path.join(dir_path, 'samples')

    model, mean, std = loaders.load_smp_model(model_path, device)

    i = 0
    for file in tqdm(os.listdir(imgs_path), desc='Processing images'):
        if file.endswith('.jpg'):
            img_path  = os.path.join(imgs_path, file)

            #[1, C, H, W]
            x       = loaders.load_img_to_tensor(img_path, std, mean, res, device)
            h, w    = x.shape[2:]
            n_rows  = h // patch_s
            n_cols  = w // patch_s
            patches = image_to_patches(x, patch_s).squeeze(0).to(device)

            with torch.no_grad():
                out = model(patches).sigmoid()

            image = patches_to_image(out, patch_s, n_rows, n_cols)
            image = (image > threshold).type(torch.uint8)
            image = image.permute(1,2,0)
            image = image.cpu().numpy()
            image = image * 255

            resized_mask = cv.resize(image, (2200, 2200), interpolation=cv.INTER_CUBIC)

            saving_dest = os.path.join(mask_path, file)
            cv.imwrite(saving_dest, resized_mask) # saving the resized mask

            if i % 20 == 0:
                raw_img = cv.imread(img_path)
                dest = os.path.join(sample_path, file)
                visualisers.show_overlay(dest, raw_img, resized_mask, alpha=0.5)

            i += 1

if __name__ == "__main__":

    args = argparser.parse_args()
    main(args)

