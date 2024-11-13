import os
from utils import visualisers
from utils import loaders
import torch
import cv2 as cv
from utils import visualisers
import csv
import argparse
from tqdm import tqdm 
import sys
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--dir_path', default='../datasets/concrete_crack_segmentation', type=str)
argparser.add_argument('--model_path', default='../weights/unet_efficientnet-b7', type=str)
argparser.add_argument('--resize', default=1056*2, type=int)
argparser.add_argument('--t', default=0.5, type=float)
argparser.add_argument('--patch_size', default=(1056,1056), type=tuple[int, int])

def main(args):
    dir_path   = args.dir_path
    model_path = args.model_path
    res        = args.resize #2080
    threshold  = args.t
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_s    = args.patch_size #416
    stride     = args.patch_size #416

    mask_path   = os.path.join(dir_path, 'masks')
    imgs_path   = os.path.join(dir_path, 'images')
    sample_path = os.path.join(dir_path, 'samples')

    model, mean, std = loaders.load_smp_model(model_path, device)

    im_size      = args.resize
    p_h,p_w      = args.patch_size[0], args.patch_size[1] #patch height, width
    in_c         = 3
    out_c        = 1
    nb_patches_h = im_size // p_h
    nb_patches_w = im_size // p_w


    i = 0
    for file in tqdm(os.listdir(imgs_path), desc='Processing images'):
        if file.endswith('.jpg'):
            img_path  = os.path.join(imgs_path, file)
            x = loaders.load_img_to_tensor(img_path, std, mean, res, device)

            # [1, C, im_size/patch_size, im_size/patch_size, patch_size, patch_size]
            patches = x.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
            unfold_shape = torch.Size((1, out_c, nb_patches_h, nb_patches_w, p_h, p_w))

            # [num_patches, C, patch_size, patch_size]
            patches = patches.contiguous().view(-1, in_c, p_h, p_w)
            print(patches.shape)


            out = []
            with torch.no_grad():
                for patch in patches:
                    result = model(patch.unsqueeze(0))
                    out.append(result.squeeze(0))

            print("length: ", len(out))
            out = torch.stack(out, dim=0)
            out = out.view(unfold_shape)
            print(f'after view {out.shape}')
            out = out.permute(0,1,2,4,3,5).contiguous()
            print(f'after permute {out.shape}')
            out = out.view(1, im_size, im_size)

            print(f'final {out.shape}')
            test_img = cv.imread(img_path)
            test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
            with torch.no_grad():
                temp = model(x).squeeze(0)
            visualisers.imshow('res', input=test_img, target=temp.cpu().permute(1,2,0), reconstructed=out.cpu().permute(1,2,0))


            print('---------------------------------------')
            # [BS, C, H, W] = [3, 2080, 2080]
            # patch = 416
            size = 2080
            patch = 416
            channel, h, w = 1, patch, patch
            x = torch.arange(channel * size * size).view(1, channel, size, size)

            out = x.unfold(2, h, h).unfold(3, w, w)
            unfold_shape = out.size()

            out = out.contiguous().view(-1, channel, h, w)

            res = []
            for sample in out:
                res.append(sample)

            out = torch.stack(res, dim=0)

            out = out.view(unfold_shape)
            out = out.permute(0,1,2,4,3,5).contiguous()
            out = out.view(1,channel,size, size)


            #patches_orig = patches_orig.view(1, output_c, output_h, output_w)
            print(f'Does x and reconstructed_x have the same dims? {x.shape == out.shape}')
            print(f'Are all elements equal?: {torch.sum(torch.eq(x, out)) == (channel * size * size)}')
            sys.exit()

#            with torch.no_grad():
#                logits = model(torch_img)
#
#            prob_mask = logits.sigmoid()
#            prob_mask = (prob_mask > threshold).type(torch.uint8)
#            prob_mask = prob_mask.squeeze().cpu().numpy()
#            prob_mask = prob_mask * 255
#
#            resized_mask = cv.resize(prob_mask, (2200, 2200), interpolation=cv.INTER_CUBIC)
#
#            saving_dest = os.path.join(mask_path, file)
#            cv.imwrite(saving_dest, resized_mask)
#
#            if i % 20 == 0:
#                raw_img = cv.imread(img_path)
#                raw_img = cv.resize(raw_img, (res, res))
#
#                colour_mask = cv.merge([prob_mask * 0, prob_mask, prob_mask * 0])
#                alpha = 0.5
#                overlayed = cv.addWeighted(raw_img, 1, colour_mask, alpha, 0)
#                dest = os.path.join(sample_path, file)
#                cv.imwrite(dest, overlayed)
#
#            i += 1


if __name__ == "__main__":

    args = argparser.parse_args()
    main(args)

