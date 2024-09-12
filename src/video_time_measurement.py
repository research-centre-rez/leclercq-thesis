import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Debug variables
VIDEO_PATH1 = '../data/10/10-vrsek-part0.mp4'
VIDEO_PATH2 = '../data/10/10-vrsek-part1.mp4'

def plot_accum_graph(data):
    print("Creating accumulation graph..")
    plt.figure()
    for timeline, distances, vid_name in data:
        plt.plot(timeline, distances, '-', linewidth=0.2, label=vid_name)

    plt.title("Mean dist vs frame id acculumated")
    plt.xlabel('Frame id')
    plt.ylabel('Mean distance')
    plt.legend()
    plt.savefig('accumulated_graph.png')
    plt.close()

def plot_simple_graph(data, title, xlabel, ylabel, saveToFile):
    print(f"Plotting {title}..")

    plt.plot(data[0], data[1], '-o', color='blue', markersize=0.01)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(saveToFile)
    plt.close()

def write_to_csv(data, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Video Name', 'Frame ID', 'Duration'])

        csv_writer.writerows(data)

def measure_video_length(video_path, plot_graphs=False):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video {base_name}..")

    cap = cv.VideoCapture(video_path)

    # Initialise ORB and a brute-force matcher
    orb = cv.ORB_create(nfeatures=1000)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    ret, first_frame = cap.read()

    gray_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    kp_og, dp_og = orb.detectAndCompute(first_frame, None)

    fps         = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))
    start_frame = frame_count - 400
    frame_idx   = start_frame
    timeline    = np.arange(start_frame, frame_count)

    cap.set(1, start_frame)

    mean_distances = []
    mismatches     = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        _, dp_f = orb.detectAndCompute(gray_frame, None)

        matches = bf.match(dp_og, dp_f)

        #idk whether this is needed at the moment
        #matches = sorted(matches, key=lambda x: x.distance)

        # Variables for printing in the console
        progress      = (frame_idx / frame_count) * 100
        mismatch      = len(kp_og) - len(matches)
        mean_distance = np.mean([m.distance for m in matches]).round(2)

        mean_distances.append(float(mean_distance))
        mismatches.append(mismatch)

        print(f'\rProgress: {progress:04.2f}% Mismatches: {mismatch} Mean Dist: {mean_distance}',
              end='',
              flush=True)

        frame_idx += 1

    print("")
    cap.release()

    if plot_graphs:
        plot_simple_graph((timeline,mean_distances),
                          title=f"Mean distance vs. frame index for {base_name}",
                          xlabel="Frame id",
                          ylabel="Mean distance",
                          saveToFile=f'mean-dist-{base_name}.png')

        plot_simple_graph((timeline, mismatches),
                          title=f"# mismatches vs frame index for {base_name}",
                          xlabel="Frame id",
                          ylabel="# mismatches",
                          saveToFile=f'mismatches-{base_name}.png')

    mean_distances = np.array(mean_distances)
    min_dist = mean_distances.min()
    min_dist_frame_id = timeline[mean_distances.argmin()]
    time = min_dist_frame_id / fps
    print(f"min dist: {min_dist} with time {time:05.3f}")
    return min_dist_frame_id, time, (timeline, mean_distances, base_name)

def process_videos_in_directories(root_dir):
    counter = 0
    min_ids = []
    times   = []
    vidids  = []
    csv_path = 'tmp_data.csv'
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Vid name', 'Min dist ID', 'Duration'])

        for dir_, _, files in os.walk(root_dir, topdown=True):
            for file_name in files:
                if file_name.endswith('.mp4'):
                    rel_dir = os.path.relpath(dir_, os.getcwd())
                    rel_file = os.path.join(rel_dir, file_name)
                    min_dist_id, time, _ = measure_video_length(rel_file, False)
                    counter += 1

                    min_ids.append(min_dist_id)
                    times.append(round(time, 3))
                    vidids.append(rel_file)

                    print("writing to csv file")
                    csv_writer.writerow([file_name, min_dist_id, round(time, 3)])
                    print('--------------------------------------------')

    timeline = np.arange(len(min_ids))
    plot_simple_graph((timeline, min_ids),
                      title='min frame id per video',
                      xlabel='video "id"',
                      ylabel='frame id',
                      saveToFile='min-frame-vs-time.png')

    plot_simple_graph((timeline, times),
                      title='time for full rotation per video',
                      xlabel='video "id"',
                      ylabel='video length in seconds',
                      saveToFile='rotation-length.png')

    csv_data = list(zip(vidids, min_ids, times))
    write_to_csv(csv_data, './final_data.csv')

process_videos_in_directories("../data")
