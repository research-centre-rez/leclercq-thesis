import json

def calc_frame_duration(pts_file):
    with open(pts_file, 'r') as f:
        data = json.load(f)

    frames = data['frames']
    durs   = []

    for i in range(1, len(frames)):
        pts_curr = int(frames[i]['pkt_pts'])
        pts_prev = int(frames[i-1]['pkt_pts'])

        durs.append(pts_curr - pts_prev)

    return durs

def calc_average_duration(durations):
    return sum(durations) / len(durations)

def duration_to_framerate(avg_duration, timebase):
    return timebase / avg_duration

train_timebase = 90000
calib_timebase = 24000

train_video = calc_frame_duration('train_video_pkt_pts.json')
calib_video = calc_frame_duration('calib_video_pkt_pts.json')

train_avg = calc_average_duration(train_video)
calib_avg = calc_average_duration(calib_video)

print(f'Average frame duration of video1: {train_avg}')
print(f'Average frame duration of video2: {calib_avg}')

train_fps = duration_to_framerate(train_avg, train_timebase)
cal_fps   = duration_to_framerate(calib_avg, calib_timebase)

print(f'Train video framerate: {train_fps:.4f} fps')
print(f'Calibration video framerate: {cal_fps:.4f} fps')
