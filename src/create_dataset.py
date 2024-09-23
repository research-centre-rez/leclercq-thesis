import cv2 as cv
import os
import csv

# Creates a dataset for manual labeling, screenshots are taken at regular interval

def extract_frames(video_path, start_frame, end_frame, howMany = 5):
    base_name  = os.path.splitext(os.path.basename(video_path))[0]
    cap        = cv.VideoCapture(video_path)
    output_dir = './models/dataset'

    currentFrame = int(start_frame)

    cap.set(1, currentFrame)

    total_frames = int(end_frame) - int(start_frame)

    frame_increment = total_frames // howMany

    for i in range(howMany):
        cap.set(1, currentFrame)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"{base_name}-{i+1}.jpg")
            cv.imwrite(frame_path, frame)

        currentFrame += frame_increment
        if currentFrame > end_frame:
            break

def load_csv(csv_path):
    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skips the header
        for row in reader:
            name, first_frame_id, distance_id, mismatch_id = row
            end_frame = (int(distance_id) + int(mismatch_id) ) // 2

            print(f'Processing: {name}')

            extract_frames(name, first_frame_id, end_frame)

load_csv('./final_data.csv')
