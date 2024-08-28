import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video_path1 = '../data/1/1-vrsek-part0.mp4'
video_path2 = '../data/1/1-vrsek-part1.mp4'

def plot_mean_dist(mean_dist):
    print("Making the plot")
    print(f"number of points to plot: {len(mean_dist)}")
    plt.plot(mean_dist, '-o', color='blue', markersize=0.1)
    plt.title("Mean distance vs Frame Index")
    plt.xlabel("Frame id")
    plt.ylabel("Mean distance")
    plt.savefig("mean_distances_plot.png")
    plt.close()

def measure_video_length(video_path):

    cap = cv.VideoCapture(video_path)

    orb = cv.ORB_create()

    ret, firt_frame = cap.read()

    gray_frame = cv.cvtColor(firt_frame, cv.COLOR_BGR2GRAY)

    kp_og, dp_og = orb.detectAndCompute(gray_frame, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    rot_ts = []

    frame_idx = 0
    fps = cap.get(cv.CAP_PROP_FPS)

    frame_count= int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    mean_distances = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        kp_f, dp_f = orb.detectAndCompute(gray_frame, None)

        matches = bf.match(dp_og, dp_f)

        matches = sorted(matches, key=lambda x: x.distance)

        progress = (frame_idx / frame_count) * 100
        mismatch = len(kp_og) - len(matches)
        mean_distance = np.mean([m.distance for m in matches]).round(2)
        mean_distances.append(float(mean_distance))

        print(f'\rProgress: {progress:04.2f}% Mismatches: {mismatch} Mean Distance: {mean_distance}',
              end='',
              flush=True)

        frame_idx += 1

    cap.release()
    print(rot_ts)
    plot_mean_dist(mean_distances)
    min_dist = np.array_split(mean_distances, 2)
    min_dist = min_dist[1].min()
    min_dist_frame_id = np.where(mean_distances == min_dist)[0][0]
    time = min_dist_frame_id / fps
    print(time)

measure_video_length(video_path1)
measure_video_length(video_path2)

