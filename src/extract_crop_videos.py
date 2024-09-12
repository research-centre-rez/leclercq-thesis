import os
import sys
import cv2 as cv
import numpy as np
import csv

import segformer_masking

def extract_screenshots(vid_path, frame_id, time):
    '''
    Is given the path to a video and the length of the full rotation. Takes it and extracts the screenshots, saves them in the same directory. 
    '''
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    output_dir = os.path.dirname(vid_path)
    print(f"Extracting images from {vid_path}..")

    cap = cv.VideoCapture(vid_path)
    ret, first_frame = cap.read()
    if ret:
        first_frame_path = os.path.join(output_dir, f"{base_name}-0deg.jpg")
        cv.imwrite(first_frame_path, first_frame)

    halfway_frame_id = int(frame_id) // 2
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
            name, frame_id, time = row
            extract_screenshots(name, frame_id, time)

def crop_images(root_dir):
    for dir_, _, files in os.walk(root_dir, topdown=True):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, os.getcwd())
            rel_file = os.path.join(rel_dir, file_name)
            if file_name.endswith('.jpg'):
                print(f"Cropping {file_name} via SegFormer..")
                segformer_masking.get_circle(model=None, img_path=rel_file, output_dir=rel_dir)

read_csv('final_data.csv')
crop_images('../data')
