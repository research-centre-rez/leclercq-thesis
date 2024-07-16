import numpy as np
import cv2 as cv
from moviepy.editor import VideoFileClip
import os

def detect_black_frames(vid_path, threshold=1):
    prev_black = True

    cap = cv.VideoCapture(vid_path)
    print(f"loaded video {os.path.splitext(os.path.basename(vid_path))[0]}")
    black_frame_indices = []
    frame_count = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    for i in range(frame_count):
        print(f'\rworking on frame: {i} out of {frame_count}', end='', flush=True)
        ret, frame = cap.read()
        if not ret:
            break

        if i == frame_count-1 and not prev_black:
            black_frame_indices.append(i)

        elif prev_black and np.mean(frame) > threshold:
            black_frame_indices.append(i-1)
            prev_black = False

        elif not prev_black and np.mean(frame) < threshold:
            black_frame_indices.append(i)
            prev_black = True

    cap.release()
    print(f'\rnumber of black frames: {black_frame_indices}')
    return black_frame_indices

def split_video(video_path, frame_idx, output_dir):
    clip = VideoFileClip(video_path)
    fps = clip.fps

    index = 0
    for i in range(0, len(frame_idx), 2):
        print('indices of frames: ', (frame_idx[i], frame_idx[i+1]))

        start, end = frame_idx[i], frame_idx[i+1]
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        print(base_name)

        start_t = start / fps
        end_t = end / fps

        subclip = clip.subclip(start_t, end_t).without_audio()
        subclip.write_videofile(f'{output_dir}{base_name}-part{index}.mp4')
        index += 1

def process_videos(directory_path):
    output_dir = '../data-copy/split-videos/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    failed_files = []

    for file in os.listdir(directory_path):
        if file.endswith('.MP4'):
            video_path = os.path.join(directory_path, file)

            black_frame_idx = detect_black_frames(video_path)

            if len(black_frame_idx) % 2 == 0:
                split_video(video_path, black_frame_idx, output_dir)
            else:
                print(f'Was not able to split file {file}.')
                failed_files.append(file)

    with open('failed_files.log', 'w') as log_file:
        for file in failed_files:
            log_file.write(file + '\n')

process_videos('../data-copy/')
