import os
import sys
import logging

from circle_fit.circle_fit import taubinSVD
import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils.pprint import pprint_dict
from utils.prep_cap import prep_cap

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
        'average_angle_per_frame_deg': avg_angle_per_frame_deg,
        'variance_angle_per_frame': var_angle_per_frame,
        'cumulative_angles_deg': cum_angles_deg,
        'mean_angular_velocity_deg_per_frame': np.mean(avg_angle_per_frame_deg),
        'median_angular_velocity_deg_per_frame': np.median(avg_angle_per_frame_deg),
        'total_rotation_deg': cum_angles_deg[-1],
        'frame_count': max_frames
    }
    return results


def analyse_sparse_optical_flow(vid_path, num_points=10, f_params=None, lk_params=None) -> list[np.ndarray]:
    """
    Analyses optical flow in a video by tracking feature points. Can be used for registered and non-registered videos. In a registered video this will effectively measure how well is the video registered, in a non-registered video it will measure the optical flow of the video.

    Args:
        video_path (str): Path to the video
        num_points (int): Number of feature points to track
        plot_results (bool): Whether to plot the results

    Returns:
        List of all trajectories
    """

    # Init logger
    logger = logging.getLogger(__name__)
    cap = prep_cap(vid_path, 15)

    ret, first_frame = cap.read()
    if not ret:
        logger.error("Error with reading the first frame of video")
        sys.exit(-1)

    # Convert first frame to grayscale
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Feature params
    if f_params is None:
        f_params = {
            "maxCorners": num_points,
            "qualityLevel": 0.001,
            "minDistance": 7,
            "blockSize": 7,
        }

    # Lukas-kane sparse optical flow params
    if lk_params is None:
        lk_params = {
            "winSize": (40, 40),
            "maxLevel": 4,
            "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.001),
        }

    pprint_dict(f_params, "Feature params: ")
    pprint_dict(lk_params, "LK params:")

    # Detecting feature points in the first frame
    corners = cv.goodFeaturesToTrack(first_gray, mask=None, **f_params)

    trajectories = [[] for _ in range(len(corners))]
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        trajectories[i].append((x, y))

    frame_count = 0
    prev_gray = first_gray

    with tqdm(desc="Calculating optical flow", total=int(cap.get(cv.CAP_PROP_FRAME_COUNT))) as pbar:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            new_c, status, error = cv.calcOpticalFlowPyrLK(prev_gray,
                                                           gray,
                                                           corners,
                                                           None,
                                                           **lk_params)

            mean_error = np.mean(error[status == 1])
            pbar.set_postfix(mean_error=f"{mean_error:.4f}")

            # Select good points
            good_new = new_c[status == 1]

            # Store trajectories
            for i, new in enumerate(good_new):
                if i < len(trajectories):
                    x, y = new.ravel()
                    trajectories[i].append((x, y))

            prev_gray = gray.copy()
            corners = good_new.reshape(-1, 1, 2)

            pbar.update(1)
            i += 1

        pbar.close()
        cap.release()

    # Convert to a list of np arrays for analysis
    # Not converting directly to an np array because it is
    # not guaranteed that all trajectories have the same length
    np_trajectories = []
    for traj in trajectories:
        logger.debug("Length of trajectory %s", len(traj))
        if len(traj) > 0:
            np_trajectories.append(np.array(traj))

    return np_trajectories

def analyse_dense_optical_flow(video_path, farneback_p=None):
    if farneback_p is None:
        farneback_p = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    cap = prep_cap(video_path, set_to=15)
    
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
