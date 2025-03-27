import logging
import os
import pprint
import sys
import argparse

from cv2.gapi import BGR2Gray

from circle_fit import plot_data_circle, taubinSVD
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.signal import find_peaks

from image_registration.video_matrix import rotate_frames_optical_flow
from utils.filename_builder import append_file_extension, create_out_filename
import utils.visualisers
from utils.pprint import pprint_argparse, pprint_dict
from utils.tqdm_utils import tqdm_generator

def parse_args():
    # Argparse configuration
    argparser = argparse.ArgumentParser(description='Creating a video matrix and rotating it')

    optional = argparser._action_groups.pop()
    req = argparser.add_argument_group('required arguments')

    # Required arguments
    req.add_argument('-i', '--input', type=str, help='Path to the input video', required=True)
    optional.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help='Show the displacement in a GUI')
    optional.add_argument('--save', default=False, action=argparse.BooleanOptionalAction, help='Whether to save the final graph')
    optional.add_argument('--num_kp', type=int, default=15, help='How many keypoints do you want to track?')
    optional.add_argument('--save_video', default=False, action=argparse.BooleanOptionalAction, help='Whether to save the ')
    optional.add_argument('--analyse', default=False, action=argparse.BooleanOptionalAction, help='Perform angular motion analysis')

    return argparser.parse_args()

def fit_circle_from_points(points:np.ndarray):
    '''
    Given a set of points in cartesian coordinates, try to fit a circle that describes them. 
    Args:
        points (np.ndarray): Points in 2D cartesian coordinates
    Returns:
        xc,yc: Center of the found circle
        R: Radius
        error: Residual error of the solution
    '''
    xc,yc, R, error = taubinSVD(points)
    return xc,yc,R,error

def estimate_rotation_center_individually(trajectories:list[np.ndarray], return_mean=True):
    '''
    Estimates center for each of the trajectories, returns their mean or median.
    Args:
        trajectories (list[np.ndarray]): List of trajectories, each one containing an `np.ndarray` that describes the trajectory of the point.
        return_mean: Whether to return mean or median of the circles
    Returns:
        center: Of all the trajectories
        error: Mean / median of the residual errors
    '''
    centers = []
    err     = []

    for traj in tqdm(trajectories, desc='Finding center of rotation'):
        # Need at least 3 points
        if len(traj) < 3:
            continue

        xc,yc, _, error = fit_circle_from_points(traj)
        centers.append((xc,yc))
        err.append(error)

    err_arr     = np.array(err)
    centers_arr = np.array(centers)

    if return_mean:
        center = centers_arr.mean(axis=0)
        error  = err_arr.mean()
        return center, error

    center = np.median(centers_arr, axis=0)
    error  = np.median(err_arr)

    return center, error

def estimate_rotation_center(trajectories):
    """
    Estimate the center of rotation for a set of trajectories.
    
    Args:
        trajectories (list): List of trajectory arrays with shape (n_frames, 2)
        
    Returns:
        tuple: (center_x, center_y) - the estimated center of rotation
        float: redidual error
    """
    # Combine all trajectory points for initial guess
    all_points = np.vstack([traj for traj in trajectories])

    xc, yc, _, error = taubinSVD(all_points)

    center = (xc,yc)

    return center, error

def calculate_angular_movement(trajectories, center):
    '''
    Calculates angular movement around a center of rotation.
    '''
    center_x, center_y = center

    max_frames = max(len(traj) for traj in trajectories)

    frame_angles = [[] for _ in range(max_frames - 1)]
    all_angles   = [] #list of all trajectories and their respective angle changes

    for traj in trajectories:
        if len(traj) < 2:
            continue

        dx     = traj[:, 0] - center_x
        dy     = traj[:, 1] - center_y
        angles = np.arctan2(dy,dx)

        angle_changes = np.diff(np.unwrap(angles))

        for i, change in enumerate(angle_changes):
            frame_angles[i].append(change)

        all_angles.extend(angle_changes)

    avg_angle_per_frame = []
    var_angle_per_frame = []
    for angles in frame_angles:
        if angles:
            avg_angle = np.mean(angles)
            var_angle = np.var(angles)
        else:
            avg_angle = 0.0
            var_angle = 0.0

        avg_angle_per_frame.append(avg_angle)
        var_angle_per_frame.append(var_angle)

    if avg_angle_per_frame:
        # The first frame has angular movement of 0.0
        avg_angle_per_frame.insert(0, 0.0)
        var_angle_per_frame.insert(0, 0.0)

    cum_angles = np.cumsum(avg_angle_per_frame)

    avg_angle_per_frame_deg = [angle * 180 / np.pi for angle in avg_angle_per_frame]
    cum_angles_deg          = cum_angles * 180 / np.pi

    results = {
        'center': (center_x, center_y),
        'average_angle_per_frame_rad': avg_angle_per_frame,
        'average_angle_per_frame_deg': avg_angle_per_frame_deg,
        'variance_angle_per_frame': var_angle_per_frame,
        'cumulative_angles_rad': cum_angles,
        'cumulative_angles_deg': cum_angles_deg,
        'mean_angular_velocity_deg_per_frame': np.mean(avg_angle_per_frame_deg),
        'median_angular_velocity_deg_per_frame': np.median(avg_angle_per_frame_deg),
        'total_rotation_deg': cum_angles_deg[-1],
        'frame_count': max_frames
    }
    return results


def analyse_sparse_optical_flow(video_path, num_points=10, f_params=None, lk_params=None):
    '''
    Analyses optical flow in a video by tracking feature points. Can be used for registered and non-registered videos. In a registered video this will effectively measure how well is the video registered, in a non-registered video it will measure the optical flow of the video.

    Args:
        video_path (str): Path to the registered video
        num_points (int): Number of feature points to track
        plot_results (bool): Whether to plot the results

    Returns:
        Dictionary containing jitter metrics and trajectories
    '''

    # Init logger
    logger = logging.getLogger(__name__)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f'Error: Could not open file {video_path}')
        return None
    
    ret, first_frame = cap.read()
    if not ret:
        logger.error('Error with reading the first frame')
        return None

    # Convert first frame to grayscale
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    #first_gray = cv.equalizeHist(first_gray)

    # Feature params
    if f_params is None:
        f_params = dict(maxCorners=num_points,
                        qualityLevel=0.001,
                        minDistance=7,
                        blockSize=7)

    # Lukas-kane sparse optical flow params
    if lk_params is None:
        lk_params = dict(winSize=(40,40),
                         maxLevel=4,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.001)) 

    pprint_dict(f_params, 'Feature params: ')
    pprint_dict(lk_params, 'LK params:')

    # Detecting feature points in the first frame
    corners = cv.goodFeaturesToTrack(first_gray, mask=None, **f_params)
    
    trajectories = [[] for _ in range(len(corners))]
    for i, corner in enumerate(corners):
        x,y = corner.ravel()
        trajectories[i].append((x,y))

    frame_count = 0
    prev_gray   = first_gray

    with tqdm(desc='Opt flow', total=int(cap.get(cv.CAP_PROP_FRAME_COUNT))) as pbar:
        i = 0
        while True:
            ret,frame = cap.read()
            if not ret:
                break

            frame_count += 1

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
            #gray = cv.equalizeHist(gray)

            new_c, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, **lk_params)

            mean_error = np.mean(error[status==1])
            pbar.set_postfix(mean_error=f'{mean_error:.4f}')

            # Select good points
            good_new = new_c[status==1]

            # Store trajectories
            for i, new in enumerate(good_new):
                if i < len(trajectories):
                    x,y = new.ravel()
                    trajectories[i].append((x, y))

            prev_gray = gray.copy()
            corners = good_new.reshape(-1, 1, 2)

            pbar.update(1)
            i += 1

        pbar.close()
        cap.release()

    # Convert to np array for analysis
    np_trajectories = []
    for traj in trajectories:
        logger.debug('Length of trajectory %s', len(traj))
        if len(traj) > 0:
            np_trajectories.append(np.array(traj))

    return np_trajectories


def analyse_dense_optical_flow(video_path, opt_flow_params=None):
    if opt_flow_params is None:
        farneback_p = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Could not open video file {video_path}')
        return None
    
    ret, prev_frame = cap.read()
    if not ret:
        return None

    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    flows = []
    f_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    with tqdm(total=f_count, desc='Dense optical flow') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **farneback_p)
            flows.append(flow)

            prev_gray = gray.copy()
            pbar.update(1)

    cap.release()
    return flows

def visualise_dense_optical_flow(video_path, output_video_path=None):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties for writing output video if needed
    frame_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv.CAP_PROP_FPS)

    writer = None
    if output_video_path is not None:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Read the first frame and convert to grayscale
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        cap.release()
        return
    
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
    # Prepare an HSV image for visualization
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # Set saturation to maximum

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Compute dense optical flow using Farneback's method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                           pyr_scale=0.5, levels=3, winsize=15, 
                                           iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        
        # Compute magnitude and angle of 2D flow vectors
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Set hue according to the optical flow direction
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # Set value according to the normalized magnitude (motion intensity)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        
        # Convert HSV image to BGR format for display
        bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # Display the flow
        cv.imshow('Dense Optical Flow', bgr_flow)
        if writer is not None:
            writer.write(bgr_flow)
        
        # Break loop on 'Esc' key press
        if cv.waitKey(30) & 0xFF == 27:
            break
        
        # Use the current frame as previous for the next iteration
        prev_gray = gray.copy()
    
    cap.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()


def calculate_jitter_metrics(trajectories):

    metrics = {}

    traj_metrics = []

    num_frames = trajectories[0].shape[0]
    frames     = np.arange(num_frames) # For timeline

    for traj in trajectories:
        if len(traj) < 10:
            continue

        x = traj[:, 0]
        y = traj[:, 1]

        # Smooth the trajectories (ideal path without jitter)
        window_length = min(15, len(x) - (1 if len(x) % 2 == 0 else 0))
        if window_length < 3:
            window_length = 3
        
        try:
            # Apply Savitzky-Golay filter for smoothing
            x_smooth = savgol_filter(x, window_length, 2)
            y_smooth = savgol_filter(y, window_length, 2)
            
            # Calculate frame-to-frame displacement
            dx = np.diff(x)
            dy = np.diff(y)
            displacements = np.sqrt(dx**2 + dy**2)
            
            # Calculate jitter as deviation from smoothed path
            x_jitter = x - x_smooth
            y_jitter = y - y_smooth
            jitter_magnitudes = np.sqrt(x_jitter**2 + y_jitter**2)
            
            traj_metrics.append({
                "mean_jitter": np.mean(jitter_magnitudes),
                "max_jitter": np.max(jitter_magnitudes),
                "std_jitter": np.std(jitter_magnitudes),
                "mean_displacement": np.mean(displacements),
                "jitter_to_motion_ratio": np.mean(jitter_magnitudes) / (np.mean(displacements) + 1e-6),
                "x_jitter": x_jitter,
                "y_jitter": y_jitter,
                "jitter_magnitudes": jitter_magnitudes,
                "x_smooth": x_smooth,
                "y_smooth": y_smooth,
                "x": x,
                "y": y,
                "frames": frames
            })
        except:
            # Skip trajectories that cause errors in smoothing
            continue

    # Aggregate metrics across all trajectories
    if traj_metrics:
        metrics["mean_jitter"] = np.mean([tm["mean_jitter"] for tm in traj_metrics])
        metrics["max_jitter"] = np.max([tm["max_jitter"] for tm in traj_metrics])
        metrics["std_jitter"] = np.mean([tm["std_jitter"] for tm in traj_metrics])
        metrics["jitter_to_motion_ratio"] = np.mean([tm["jitter_to_motion_ratio"] for tm in traj_metrics])
        metrics["trajectories"] = traj_metrics
    else:
        metrics["error"] = "Not enough valid trajectories for analysis"

    return metrics

def visualize_dense_optical_flow_with_arrows(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        cap.release()
        return
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(first_frame)
    hsv[..., 1] = 255  # full saturation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv[..., 0] = ang * 180 / np.pi / 2  # map angle to hue
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        # Overlay arrows every few pixels for clarity
        step = 16
        for y in range(0, gray.shape[0], step):
            for x in range(0, gray.shape[1], step):
                fx, fy = flow[y, x].astype(np.int32)
                cv.arrowedLine(bgr_flow, (x, y), (x+fx, y+fy), (255,255,255), 1, tipLength=0.3)
        
        cv.imshow('Dense Optical Flow with Arrows', bgr_flow)
        if cv.waitKey(30) & 0xFF == 27:
            break
        
        prev_gray = gray.copy()
    
    cap.release()
    cv.destroyAllWindows()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint_argparse(args, logger)

    temp = visualize_dense_optical_flow_with_arrows(args.input)
    sys.exit()
    np_trajectories = analyse_sparse_optical_flow(args.input, num_points=args.num_kp)
    jitter_metrics  = calculate_jitter_metrics(np_trajectories)

    _, name = os.path.split(args.input)
    base, _ = os.path.splitext(name)

    if args.save_video:
        path = utils.visualisers.draw_optical_flow_video(args.input, np_trajectories)

    graph_config = {
        'title': f'Optical flow for {base}',
        'save_as': create_out_filename(f'./images/{base}', [], ['opt', 'flow']),
        'save': args.save,
        'show': args.show
    }

    if args.save or args.show:
        utils.visualisers.plot_optical_flow(np_trajectories, jitter_metrics, graph_config)

    if jitter_metrics:
        logger.info(f"Mean jitter: {jitter_metrics['mean_jitter']:.2f}")
        logger.info(f"Max jitter: {jitter_metrics['max_jitter']:.2f}" )
        logger.info(f"Jitter-to-motion ratio: {jitter_metrics['jitter_to_motion_ratio']:.4f}")

    if args.analyse:
        # Estimate center of rotation
        center, quality = estimate_rotation_center_individually(np_trajectories)
        logger.info(f"Estimated rotation center: ({center[0]:.2f}, {center[1]:.2f})")
        logger.info(f"Center quality metric: {quality:.6f} (lower is better)")
        
        # Calculate angular movement
        rotation_results = calculate_angular_movement(np_trajectories, center)

        peaks, _ = find_peaks(rotation_results['average_angle_per_frame_deg'])
        print(f'Number of peaks found: {len(peaks)}')

        if args.save:
            rotate_frames_optical_flow(args.input, rotation_results['average_angle_per_frame_deg'], 1)
        
        # Print results
        logger.info(f"Mean angular velocity: {rotation_results['mean_angular_velocity_deg_per_frame']:.5f}° per frame")
        logger.info(f"Total rotation: {rotation_results['total_rotation_deg']:.5f}°")
        
        graph_config = {
            'save_as': create_out_filename(f'./images/{base}', [], ['rotation', 'analysis']),
            'save': args.save,
            'show': args.show,
        }
        
        utils.visualisers.visualize_rotation_analysis(np_trajectories, rotation_results, graph_config=graph_config)

    if args.save:
        np.save('opt_flow_temp', np_trajectories)

if __name__ == "__main__":
    args = parse_args()
    main(args)
