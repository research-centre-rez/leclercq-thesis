import logging
import os
import pprint
import sys
import argparse

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from scipy.optimize import minimize

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


def estimate_rotation_center(trajectories, frames_to_use=None):
    """
    Estimate the center of rotation for a set of trajectories using optimization.
    
    Args:
        trajectories (list): List of trajectory arrays with shape (n_frames, 2)
        frames_to_use (int): Number of frames to use for estimation. If None, use all.
        
    Returns:
        tuple: (center_x, center_y) - the estimated center of rotation
        float: quality metric (lower is better)
    """
    # Combine all trajectory points for initial guess
    all_points = np.vstack([traj[:frames_to_use] if frames_to_use else traj for traj in trajectories])
    initial_center = np.mean(all_points, axis=0)
    
    # Define objective function to minimize
    def objective(center):
        center_x, center_y = center
        variance_sum = 0
        count = 0
        
        # For each trajectory, calculate variance of distances from center
        for traj in trajectories:
            points = traj[:frames_to_use] if frames_to_use else traj
            if len(points) < 2:
                continue
                
            # Calculate distances from center to each point
            dx = points[:, 0] - center_x
            dy = points[:, 1] - center_y
            distances = np.sqrt(dx**2 + dy**2)
            
            # Variance of distances should be minimal for a true rotation center
            variance_sum += np.var(distances)
            count += 1
            
        return variance_sum / max(count, 1)
    
    # Optimize to find the center
    result = minimize(objective, initial_center, method='Nelder-Mead')
    center = result.x
    quality = result.fun
    
    return center, quality

def calculate_angular_movement(trajectories, center, smooth_window=None):
    """
    Calculate the angular movement of keypoints around a center for each frame.
    
    Args:
        trajectories (list): List of trajectory arrays with shape (n_frames, 2)
        center (tuple): (x, y) coordinates of the rotation center
        smooth_window (int): Window size for smoothing angles, None for no smoothing
        
    Returns:
        dict: Dictionary containing angular movements and related metrics
    """
    center_x, center_y = center
    
    # Find the maximum frame count across all trajectories
    max_frames = max(len(traj) for traj in trajectories)
    
    # Initialize arrays for storing angular data
    all_angles = []  # All individual angle changes
    frame_angles = [[] for _ in range(max_frames-1)]  # Angle changes by frame
    cumulative_angles = np.zeros(max_frames)  # Cumulative rotation
    
    # Process each trajectory
    for traj in trajectories:
        if len(traj) < 2:
            continue
            
        # Calculate the angle of each point relative to center
        dx = traj[:, 0] - center_x
        dy = traj[:, 1] - center_y
        angles = np.arctan2(dy, dx)
        
        # Calculate angle changes between frames (unwrapped to handle -π to π transitions)
        angle_changes = np.zeros(len(angles)-1)
        for i in range(len(angles)-1):
            change = angles[i+1] - angles[i]
            # Handle wrapping around +/- π
            if change > np.pi:
                change -= 2 * np.pi
            elif change < -np.pi:
                change += 2 * np.pi
            angle_changes[i] = change
            
            # Store by frame for averaging
            if i < len(frame_angles):
                frame_angles[i].append(change)
                
        all_angles.extend(angle_changes)
    
    # Calculate average angle change per frame
    average_angle_per_frame = []
    variance_angle_per_frame = []
    for frame_idx, angles in enumerate(frame_angles):
        if angles:
            avg_angle = np.mean(angles)
            var_angle = np.var(angles)
            average_angle_per_frame.append(avg_angle)
            variance_angle_per_frame.append(var_angle)
            
            # Update cumulative angle
            cumulative_angles[frame_idx+1:] += avg_angle
    
    # Apply smoothing if requested
    if smooth_window and len(average_angle_per_frame) > smooth_window:
        from scipy.signal import savgol_filter
        # Make sure window length is odd
        if smooth_window % 2 == 0:
            smooth_window += 1
        average_angle_per_frame = savgol_filter(
            average_angle_per_frame, smooth_window, 2
        ).tolist()
    
    # Convert to degrees
    average_angle_per_frame_deg = [angle * 180 / np.pi for angle in average_angle_per_frame]
    cumulative_angles_deg = cumulative_angles * 180 / np.pi
    
    # Prepare results
    results = {
        "center": (center_x, center_y),
        "average_angle_per_frame_rad": average_angle_per_frame,
        "average_angle_per_frame_deg": average_angle_per_frame_deg,
        "variance_angle_per_frame": variance_angle_per_frame,
        "cumulative_angles_rad": cumulative_angles,
        "cumulative_angles_deg": cumulative_angles_deg,
        "mean_angular_velocity_deg_per_frame": np.mean(average_angle_per_frame_deg),
        "median_angular_velocity_deg_per_frame": np.median(average_angle_per_frame_deg),
        "total_rotation_deg": cumulative_angles_deg[-1],
        "frame_count": max_frames
    }
    
    return results


def analyse_optical_flow(video_path, num_points=10, f_params=None, lk_params=None):
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

    pprint_dict(f_params, 'feature params: ')
    pprint_dict(lk_params, 'LK params:')

    # Detecting feature points in the first frame
    corners = cv.goodFeaturesToTrack(first_gray, mask=None, **f_params)
    
    trajectories = []
    for i in range(len(corners)):
        trajectories.append([])

    frame_count = 0
    prev_gray   = first_gray

    frame_counter = tqdm(desc='Calculating optical flow')
    for _ in tqdm_generator():
        ret,frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    

        new_c, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, corners, None, **lk_params)
        logger.debug('Error: \n%s', error)

        mean_error = np.mean(error[status==1])
        frame_counter.set_postfix(mean_error=f'{mean_error:.4f}')

        # Select good points
        good_new = new_c[status==1]

        # Store trajectories
        for i, new in enumerate(good_new):
            if i < len(trajectories):
                x,y = new.ravel()
                trajectories[i].append((x, y))

        prev_gray = gray.copy()
        corners = good_new.reshape(-1, 1, 2)

        frame_counter.update(1)

    frame_counter.close()
    cap.release()

    # Convert to np array for analysis
    np_trajectories = []
    for traj in trajectories:
        logger.debug('Length of trajectory %s', len(traj))
        if len(traj) > 0:
            np_trajectories.append(np.array(traj))

    return np_trajectories

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

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
    logger = logging.getLogger(__name__)
    pprint_argparse(args, logger)

    np_trajectories = analyse_optical_flow(args.input, num_points=args.num_kp)
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
        center, quality = estimate_rotation_center(np_trajectories)
        logger.info(f"Estimated rotation center: ({center[0]:.2f}, {center[1]:.2f})")
        logger.info(f"Center quality metric: {quality:.6f} (lower is better)")
        
        # Calculate angular movement
        rotation_results = calculate_angular_movement(np_trajectories, center, smooth_window=5)
        
        # Print results
        logger.info(f"Mean angular velocity: {rotation_results['mean_angular_velocity_deg_per_frame']:.5f}° per frame")
        logger.info(f"Total rotation: {rotation_results['total_rotation_deg']:.5f}°")
        
        graph_config = {
            'save_as': create_out_filename(f'./images/{base}', [], ['rotation', 'analysis']),
            'save': args.save,
            'show': args.show,
        }
        
        utils.visualisers.visualize_rotation_analysis(np_trajectories, rotation_results, graph_config=graph_config)

    np.save('opt_flow_temp', np_trajectories)

if __name__ == "__main__":
    args = parse_args()
    main(args)
