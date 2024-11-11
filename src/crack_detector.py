import os
from utils import visualisers
from utils import loaders
import torch
import cv2 as cv
from utils import visualisers
import csv
import argparse
from tqdm import tqdm 

argparser = argparse.ArgumentParser()
argparser.add_argument('--dir_path', default='../datasets/concrete_crack_segmentation', type=str)
argparser.add_argument('--model_path', default='../weights/fpn_timm-mobilenetv3_large_100', type=str)
argparser.add_argument('--resize', default=1280, type=int)
argparser.add_argument('--t', default=0.7, type=float)

def main(args):
    dir_path   = args.dir_path
    model_path = args.model_path
    res        = args.resize
    threshold  = args.t
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    mask_path   = os.path.join(dir_path, 'masks')
    imgs_path   = os.path.join(dir_path, 'images')
    sample_path = os.path.join(dir_path, 'samples')

    model, mean, std = loaders.load_smp_model(model_path, device)

    i = 0
    for file in tqdm(os.listdir(imgs_path), desc='Processing images'):
        if file.endswith('.jpg'):
            img_path  = os.path.join(imgs_path, file)
            torch_img = loaders.load_img_to_tensor(img_path, std, mean, res, device)

            with torch.no_grad():
                logits = model(torch_img)

            prob_mask = logits.sigmoid()
            prob_mask = (prob_mask > threshold).type(torch.uint8)
            prob_mask = prob_mask.squeeze().cpu().numpy()
            prob_mask = prob_mask * 255

            resized_mask = cv.resize(prob_mask, (2200, 2200), interpolation=cv.INTER_CUBIC)

            saving_dest = os.path.join(mask_path, file)
            cv.imwrite(saving_dest, resized_mask)

            if i % 20 == 0:
                raw_img = cv.imread(img_path)
                raw_img = cv.resize(raw_img, (res, res))

                colour_mask = cv.merge([prob_mask * 0, prob_mask, prob_mask * 0])
                alpha = 0.5
                overlayed = cv.addWeighted(raw_img, 1, colour_mask, alpha, 0)
                dest = os.path.join(sample_path, file)
                cv.imwrite(dest, overlayed)

            i += 1


if __name__ == "__main__":

    args = argparser.parse_args()
    main(args)

