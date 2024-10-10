import cv2 as cv
import os
import csv
from image_processing.extract_frames_from_video import extract_frames_from_video

def load_csv(csv_path):
    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skips the header
        for row in reader:
            name, first_frame_id, distance_id, mismatch_id = row
            end_frame = (int(distance_id) + int(mismatch_id) ) // 2

            print(f'Processing: {name}')

            extract_frames_from_video(name, first_frame_id, end_frame, "../new_dataset")

load_csv('./full_rotation_times.csv')
