import csv
import os
import cv2 as cv
from utils import visualisers
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', default='../dev_dataset/triples.csv', type=str, help='Location of the csv_file')

def register_images():
    ...
def rotate_image(image, angle):
    """
    Rotates an image around its center by `angle` degrees
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat      = cv.getRotationMatrix2D(image_center, angle, 1.0)

    return cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_CUBIC)

def load_csv(csv_path):
    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)  # For reading files use pands.read_csv - IMO best API
        ncols = len(next(reader))
        csvfile.seek(0)
        print(ncols)
        i = 0
        for row in reader:
            first, second, third = row
            print(f'Processing: {first}, {second}, {third}')

            target    = cv.imread(first)
            rotated_1 = cv.imread(second)
            rotated_2 = cv.imread(third)

            rotated_1 = rotate_image(rotated_1, 120)
            rotated_2 = rotate_image(rotated_2, 240)

            overlay = cv.addWeighted(target.copy(), 0.5, rotated_1.copy(), 0.5, 0.0)
            #overlay = cv.addWeighted(overlay, 0.5, rotated_2, 0.5, 0.0)
            cv.imwrite(f'overlay_test_{i}.jpg', overlay)

            #target_gray      = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
            #rotated_1_gray   = cv.cvtColor(rotated_1, cv.COLOR_BGR2GRAY)
            #rotated_2_gray   = cv.cvtColor(rotated_2, cv.COLOR_BGR2GRAY)

            #target_gray      = cv.equalizeHist(target_gray)
            #rotated_1_gray   = cv.equalizeHist(rotated_1_gray)
            #rotated_2_gray   = cv.equalizeHist(rotated_2_gray)

            sift = cv.SIFT_create()
            kps_a, desc_a = sift.detectAndCompute(target, None)
            kps_b, desc_b = sift.detectAndCompute(rotated_1, None)

            index_params  = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)

            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_b, desc_a, k=2)

            good_matches = []
            for m,n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m)

            good_matches = sorted(good_matches, key=lambda x: x.distance)
            good_matches = good_matches[:100]
            if len(good_matches) < 4:
                continue

            img_matches = cv.drawMatches(rotated_1,
                                         kps_b,
                                         target,
                                         kps_a, good_matches,
                                         None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #cv.namedWindow('img', cv.WINDOW_NORMAL)
            #cv.imshow('img', img_matches)
            #cv.waitKey(0)
            cv.imwrite(f'matches{i}.png', img_matches)

            src_pts = np.float32([kps_a[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kps_b[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)

            H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0, confidence=0.999)
            h,w = target.shape[:2]

            aligned = cv.warpPerspective(rotated_1, H, (w,h))

            sift_overlay = cv.addWeighted(target, 0.5, aligned, 0.5, 0.0)
            cv.imwrite(f'overlay_sift{i}.jpg', sift_overlay)
            target  = cv.cvtColor(target, cv.COLOR_BGR2RGB)
            aligned = cv.cvtColor(aligned, cv.COLOR_BGR2RGB)
            visualisers.imshow(f'test_output{i}',target=target, sift_aligned=aligned)

            i += 1
            if i == 20:
                sys.exit(1)



def main(args) -> None:
    load_csv(args.csv_file)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
