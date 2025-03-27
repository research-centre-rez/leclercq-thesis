import csv
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd

def glue_image_stack(img_stack:list[str], config=None):
    extractor = SuperPoint(max_num_keypoints=3000).eval().cuda()  # load the extractor
    matcher   = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().cuda()  # load the matcher

    fixed_image = load_image(img_stack[0]).cuda()
    fixed_feats = extractor.extract(fixed_image)

    correspondences = {}
    for moving in tqdm(img_stack[1:], desc='Glueing'):
        moving_image = load_image(moving).cuda()
        moving_feats = extractor.extract(moving_image)
        matches      = matcher({'image0': fixed_feats, 'image1': moving_feats})

        feats0, feats1, matches01 = [rbd(x) for x in [fixed_feats, moving_feats, matches]]  # remove batch dimension

        matches = matches01['matches']  # indices with shape (K,2)
        points_fixed = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points_moved = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        correspondences[moving] = (points_fixed, points_moved) # points in fixed, moving

    for key, (val1,val2) in correspondences.items():
        correspondences[key] = (val1.cpu().numpy(), val2.cpu().numpy())

def main():
    # SuperPoint+LightGlue
    extractor = SuperPoint(max_num_keypoints=3000).eval().cuda()  # load the extractor
    matcher   = LightGlue(features='superpoint').eval().cuda()  # load the matcher

    img_list    = sorted(os.listdir('./test_img_stack'))
    img_list    = [os.path.join('./test_img_stack', file) for file in img_list]

    fixed_image = load_image(img_list[0]).cuda()
    fixed_feats = extractor.extract(fixed_image)

    f_np = (fixed_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    img_stack = [cv2.cvtColor(f_np, cv2.COLOR_RGB2GRAY)]

    homs = []

    for moving in tqdm(img_list[1:], desc='Glueing'):
        moving_image = load_image(moving).cuda()
        moving_feats = extractor.extract(moving_image)
        matches      = matcher({'image0': fixed_feats, 'image1': moving_feats})

        feats0, feats1, matches01 = [rbd(x) for x in [fixed_feats, moving_feats, matches]]  # remove batch dimension

        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

        moving_np = (moving_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        H, mask     = cv2.estimateAffinePartial2D(points1, points0, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        moving_warp = cv2.warpAffine(moving_np, H, (f_np.shape[1], f_np.shape[0]))
        img_stack.append(cv2.cvtColor(moving_warp, cv2.COLOR_RGB2GRAY))
        homs.append(H)

    img_stack = np.array(img_stack)
    np.save('../tmp_stack', img_stack)

    hom_im_pair = zip(img_list[1:], homs)
    with open('homographies.csv', mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'h11', 'h12', 'h13', 'h21', 'h22', 'h23', 'h31', 'h32', 'h33'])  # Header row

        for im_path, H in hom_im_pair:
            flattened_H = H.flatten().tolist()
            writer.writerow([im_path] + flattened_H)


if __name__ == "__main__":
    main()
