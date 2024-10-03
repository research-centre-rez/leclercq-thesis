import os
import sys
import cv2 as cv
import numpy as np
import csv
import masking
import argparse

#This is for convenience since extracting screenshots takes some time and we dont need to 
#always do it, therefore it is set to False by default
parser = argparse.ArgumentParser()
parser.add_argument('--crop', default=True, help='Do you want to crop images')
parser.add_argument('--extract', default=False, help='Do you want to extract images from videos')

def extract_screenshots(vid_path, first_frame_id, distance_id, mismatch_id):
    '''
    Is given the path to a video and the length of the full rotation. Takes it and extracts the screenshots, saves them in the same directory. 
    '''
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    output_dir = os.path.dirname(vid_path)
    print(f"Extracting images from {vid_path}..")

    cap = cv.VideoCapture(vid_path)
    cap.set(1, int(first_frame_id))
    ret, first_frame = cap.read()
    if ret:
        first_frame_path = os.path.join(output_dir, f"{base_name}-0deg.jpg")
        cv.imwrite(first_frame_path, first_frame)

    halfway_frame_id = int(distance_id) // 2
    cap.set(1,halfway_frame_id)

    ret, halfway_frame = cap.read()
    if ret:
        image_path = os.path.join(output_dir, f"{base_name}-180deg.jpg")
        cv.imwrite(image_path, halfway_frame)

    cap.release()

def read_csv(csv_path):
    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skips the header
        for row in reader:
            name, first_frame_id, distance_id, mismatch_id = row
            extract_screenshots(name, first_frame_id, distance_id, mismatch_id)

def crop_images(root_dir):
    model = masking.my_model()
    for dir_, _, files in os.walk(root_dir, topdown=True):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, os.getcwd())
            rel_file = os.path.join(rel_dir, file_name)

            if file_name.endswith('.jpg') and "cropped" not in file_name:
                print(f"Cropping {file_name}")
                masking.get_circle(model=model, img_path=rel_file, output_dir=rel_dir)

def main(args:argparse.Namespace) -> None:
    if args.extract:
        #Extract images
        read_csv('full_rotation_times.csv')
    if args.crop:
        #Crop images
        crop_images('../data')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
