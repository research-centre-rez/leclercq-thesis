import cv2 as cv
import os

def extract_frames_from_video(video_path:str, start_frame:int, end_frame:int, output_dir:str, how_many = 7):
    """
    Creates a dataset for manual labeling, screenshots are taken at regular interval between `start_frame` and `end_frame`. If `output_dir` does not exist, it will be created

    :video_path: Relative path to the video for extraction
    :start_frame: Frame from which extraction should start
    :end_frame: Extraction won't be happening past this frame
    :output_dir: Where to output the newly created dataset
    :how_many: How many frames to be extracted
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name  = os.path.splitext(os.path.basename(video_path))[0]
    cap        = cv.VideoCapture(video_path)

    curr_frame = int(start_frame)

    cap.set(1, curr_frame)

    total_frames    = int(end_frame) - int(start_frame)
    frame_increment = total_frames // how_many

    for i in range(how_many):
        cap.set(1, curr_frame)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"{base_name}-{i+1}.jpg")
            print(f"saving to: {frame_path}")
            cv.imwrite(frame_path, frame)

        curr_frame += frame_increment
        if curr_frame > end_frame:
            break
