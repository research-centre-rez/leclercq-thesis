import subprocess
import os
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "100000"
import numpy as np
import cv2 as cv
from tqdm import tqdm

'''
Video preprocessing

OpenCV has a default setting that specifies how many frames it will read from a video, you should change this before running any of these functions with:

export OPENCV_FFMPEG_READ_ATTEMPTS=100000

While the os.environ written above should do it for you, it is possible that it might not work.
'''

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f'{hours:02}:{minutes:02}:{secs:05.3f}'

def detect_black_frames(vid_path, threshold=5):
    """
    Given a video path, identify the frames where there the screen turns black. These indices are later converted 
    to time stamps that are passed to ffmpeg, where the video is separated into 2 parts.
    """
    prev_black = True

    cap = cv.VideoCapture(vid_path, cv.CAP_FFMPEG)
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps != 0:
        print(f"Successfully loaded video {os.path.splitext(os.path.basename(vid_path))[0]}")

    black_frame_indices = []
    frame_count         = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    center_x, center_y = width // 2, height // 2
    half_size = 500

    start_x = center_x - half_size
    start_y = center_y - half_size
    end_x   = center_x + half_size
    end_y   = center_y + half_size

    progress = tqdm(range(frame_count), leave=False)
    for i in progress:
        ret, frame = cap.read()
        mean = np.mean(frame[start_y:end_y, start_x:end_x])
        progress.set_postfix(mean=mean)

        if not ret:
            break

        # end of video and its not black
        if i == frame_count-1 and not prev_black:
            black_frame_indices.append(i)

        # previous frame was black and the current one is not
        elif prev_black and mean > threshold:
            black_frame_indices.append(i)
            prev_black = False

        # previous frame wasn't black and the current one is
        elif not prev_black and mean < threshold:
            black_frame_indices.append(i-1)
            prev_black = True

    cap.release()
    print(f'Indices where there are black frames: {black_frame_indices}')
    return black_frame_indices, fps

def split_video(video_path, frame_idx, output_dir, fps):
    """"
    Given the path of a video, split it into separate videos such that each sub-video
    contains only one angle of lighting.
    ffmpeg is used in order to preserve the quality of the video(s).
    """

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_dir, (base_name.split('-')[0]))

    print(f'{base_name} is going to be saved to {output_dir}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_index = 0 #index of video part
    for i in range(0, len(frame_idx), 2):
        print('indices of frames: ', (frame_idx[i], frame_idx[i+1]))

        start, end = frame_idx[i], frame_idx[i+1]

        # calculate the start and end times
        start_t = round((start / fps), 2)
        end_t = round((end / fps), 2)

        # ffmpeg uses hms time format so we convert
        start_t = seconds_to_hms(start_t)
        end_t = seconds_to_hms(end_t)

        output_file = os.path.join(output_dir, f'{base_name}-part{video_index}.mp4')
        subprocess.call([
            'ffmpeg', '-i', video_path, '-ss', str(start_t), '-to', str(end_t), '-c', 'copy', '-n', output_file
        ])
        video_index += 1

def process_videos(directory_path):
    output_dir = '../data-copy/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    failed_files = []

    for file in os.listdir(directory_path):
        if file.endswith('.MP4'):
            video_path = os.path.join(directory_path, file)

            black_frame_idx, fps = detect_black_frames(video_path)

            if len(black_frame_idx) % 2 == 0:
                split_video(video_path, black_frame_idx, output_dir, fps)
            else:
                print(f'Was not able to split file {file}.')
                failed_files.append(file)

    with open('failed_files.log', 'w') as log_file:
        for file in failed_files:
            log_file.write(file + '\n')

# TODO: Add argparse functionalities
if __name__ == "__main__":
    vid_path = '../../data/calibration_video.MP4'
    out = './calibration_video/'
    os.makedirs(out, exist_ok=True)
    black_f_id, fps = detect_black_frames(vid_path)
    split_video(vid_path, black_f_id, out, fps)
