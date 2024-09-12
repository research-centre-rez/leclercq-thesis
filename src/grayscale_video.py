import cv2 as cv
import numpy as np
import os

#This was test of something that I am no longer using
def grayscale_video(vid_path):
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    cap = cv.VideoCapture(vid_path)
    print(f"loaded video {base_name}")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print(width, height, fps)
    fourcc = cv.VideoWriter_fourcc(*'MJPG')

    writer = cv.VideoWriter('test.avi', fourcc, fps, (width, height), isColor=False)

    i = 0
    while cap.isOpened():
        print(f'\rprocessing frame: {i}', end='', flush=True)
        ret, frame = cap.read()
        if not ret:
            break
        print(frame.shape)
        exit()

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        writer.write(gray_frame)
        i += 1

    cap.release()
    writer.release()

grayscale_video('../data-copy/dev/9-vrsek-part0-gray.mp4')
grayscale_video('test.avi')
print("done!")
