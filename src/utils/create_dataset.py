import cv2 as cv
import os
import csv
import sys
from image_processing.extract_frames_from_video import extract_frames_from_video

# TODO: Figure out what to do with this. Either delete or fix
def load_csv(csv_path, new_dataset_path:str, howMany:int):
    def map_index(sample:str, index:str):
        return sample + index

    name_triples = []
    indices      = [f'-{i+1}-cropped.jpg' for i in range(howMany)]

    with open(csv_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) #skips the header
        for row in reader:
            name, first_frame_id, distance_id, mismatch_id = row
            end_frame = (int(distance_id) + int(mismatch_id) ) // 2

            print(f'Processing: {name}')
            sample_name = name.split('/')[-1].split('.')[0]
            sample_dest = os.path.join(new_dataset_path, sample_name)
            table_row   = list(map_index(sample_dest, index) for index in indices)

            name_triples.append(table_row)

            extract_frames_from_video(name, first_frame_id, end_frame, new_dataset_path, howMany)
    csv_filename = f'{new_dataset_path}/triples.csv'
    create_csv(name_triples, csv_filename)

def create_csv(table, destination):
    with open(destination, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


if __name__ == "__main__":
    load_csv('./full_rotation_times.csv', '../dev_dataset', 3)
