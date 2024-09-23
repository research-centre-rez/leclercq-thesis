import os
import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Test variables
VIDEO_PATH1 = '../data/10/10-vrsek-part0.mp4'
VIDEO_PATH2 = '../data/10/10-vrsek-part1.mp4'
VIDEO_PATH3 = '../data/1A/1A-exp-part1.mp4'

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

    plt.plot(data[0], data[1], '-', color='blue', linewidth=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(saveToFile)
    plt.close()

def write_to_csv(data, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Vid name', 'Starting frame', 'Min dist ID', 'Min mismatch ID'])

        csv_writer.writerows(data)

def measure_full_rotation_time(video_path, plot_graphs=False):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video {base_name}..")

    #Camera variables
    cap         = cv.VideoCapture(video_path)
    fps         = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get((cv.CAP_PROP_FRAME_COUNT)))

    # Initialise ORB and a brute-force matcher
    orb = cv.ORB_create(nfeatures=1000)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    start_time_seconds = 2
    starting_frame = int(round(start_time_seconds * fps))
    cap.set(1, starting_frame)

    ret, first_frame = cap.read()

    gray_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    kp_og, dp_og = orb.detectAndCompute(first_frame, None)

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

    mean_distances    = np.array(mean_distances)
    min_dist          = mean_distances.min()
    min_dist_frame_id = timeline[mean_distances.argmin()]
    time              = min_dist_frame_id / fps

    mismatches            = np.array(mismatches)
    min_mismatch_frame_id = timeline[mismatches.argmin()]
    mismatch_time         = min_mismatch_frame_id / fps

    print(f"min dist: {min_dist} time: {time:.3f} mismatch time: {mismatch_time:.3f}")

    return starting_frame, min_dist_frame_id, min_mismatch_frame_id, time

def process_videos_in_directories(root_dir):
    counter      = 0
    min_ids      = []
    min_mismatch = []
    times        = []
    vidids       = []
    start_frames = []

    csv_path = 'full_rotation_times.csv'

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Vid name', 'Starting frame', 'Min dist ID', 'Min mismatch ID'])

        for dir_, _, files in os.walk(root_dir, topdown=True):
            for file_name in files:
                if file_name.endswith('.mp4'):
                    rel_dir = os.path.relpath(dir_, os.getcwd())
                    rel_file = os.path.join(rel_dir, file_name)

                    start_frame, min_dist_id, min_mismatch_id, time = measure_full_rotation_time(rel_file, False)
                    counter += 1

                    min_ids.append(min_dist_id)
                    times.append(round(time, 3))
                    vidids.append(rel_file)
                    start_frames.append(start_frame)
                    min_mismatch.append(min_mismatch_id)

                    print("writing to csv file")
                    csv_writer.writerow([file_name, start_frame, min_dist_id ,min_mismatch_id])
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

    csv_data = list(zip(vidids, start_frames, min_ids, min_mismatch))
    write_to_csv(csv_data, './final_data.csv')

#measure_full_rotation_time(VIDEO_PATH1, True)
#measure_full_rotation_time(VIDEO_PATH2, True)
#measure_full_rotation_time(VIDEO_PATH3, True)
process_videos_in_directories("../data")
