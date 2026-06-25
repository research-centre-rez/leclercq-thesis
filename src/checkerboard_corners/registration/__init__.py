import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize

DEFAULT_SAMPLING = np.linspace(-90, 90, 360).reshape(-1, 1)


def compute_line_orientations(checkerboard_corners):
    """
    An image with a checkerboard is represented by the set of corner points. These points define
    multiple lines. This method computes orientation of each of these lines.
    @param checkerboard_corners list of points
    @return list of orientations
    """
    x = checkerboard_corners[:, 0].reshape(-1,1)
    y = checkerboard_corners[:, 1].reshape(-1,1)
    delta = []
    for dim in [x,y]:
        extension_vector = np.ones((len(checkerboard_corners),1))
        vertical_extension = np.matmul(extension_vector, dim.T)
        horizontal_extension = np.matmul(dim, extension_vector.T)
        delta.append(vertical_extension - horizontal_extension)
    delta = np.array(delta)
    angles = np.zeros_like(delta[0])
    angles[np.isclose(delta[0], 0)] = 90
    angles[delta[0] != 0] = np.atan(delta[1][delta[0] != 0] / delta[0][delta[0] != 0])
    angles = np.rad2deg(angles)
    out = []
    for i in np.arange(len(x)):
        out.extend(angles[i, i+1:].tolist())
    return out


def checkerboard_rotation(angles, est_angle=None, delta=None, precision=1):
    """
    Let's say, that we computed orientations of lines defined by checkerboard corners. This method
    convert this "list of line orientations" into one number - the image orientation.

    @param angles list of line orientations expected in degrees
    @param est_angle (optional) expected angle in degrees (e.g. orientation of the previous frame in video sequence)
    @param delta (optional) expected maximal deviation from the est_angle in degrees
    @param precision histogram density (angle precision) in degrees
    @return orientation in degrees
    """
    if est_angle is None or delta is None:
        bins_conf = np.arange(-90, +90, precision)
    else:
        bins_conf = np.arange(est_angle - delta, est_angle + delta, precision)
    counts, bins = np.histogram([angle % 45 for angle in angles], bins=bins_conf)
    most_frequent_angle_degrees = bins[np.argmax(counts)]
    return most_frequent_angle_degrees


def estimate_rotation_per_frame(checkerboard_corners):
    """
    Incoming parameters contains checkerboard corners from a video sequence. These checkerboards are
    continuously rotated. Purpose of this method is to estimate the speed of rotation.

    Algorithm works as follows:
        1. For each checkerboard estimates orientation (in range 0°-45° because the corners well
        define rows, columns and diagonals - strongest responses shifted by 45°).
        2. Continuous sequence of checkerboard angles is created (45° is added whenever it is
        necessary for preserving continuity).
        3. The sequence is approximated linearly and the slope is returned.
    @param checkerboard_corners list of checkerboard's list of points
    @return angle diff per item in a list i.e. degrees per frame.
    """
    corners_orientation_raw = [checkerboard_rotation(compute_line_orientations(
        np.array(checkerboard_corners[0][1])))]

    for pts in tqdm(checkerboard_corners):
        corners_orientation_raw.append(
            checkerboard_rotation(compute_line_orientations(np.array(pts[1])))
        )

    rotation_steps = np.diff(corners_orientation_raw)
    # NOTE: tglimits denotes areas in a sequence where is the step change in value.
    # In these indexes we need to fix the monotonicity of the sequence.
    tglimits = np.where(np.abs(rotation_steps) > 40)[0] + 1
    direction = - np.sign(rotation_steps[tglimits[0] - 1])

    corners_orientation = np.array(corners_orientation_raw)
    for low in tglimits:
        corners_orientation[low:] += 45 * direction
    poly = np.polyfit(np.arange(len(corners_orientation)), corners_orientation, deg=1)

    # For easy evaluation what happen in the function a plot is generated. Useful just in notebook env.
    ax = plt.subplot(111)
    ax.plot(corners_orientation, label="measured angles")
    ax.plot(np.polyval(poly, np.arange(len(corners_orientation))), label="linear approximation")
    ax.set_ylabel("cummulative angle")
    plt.legend()
    ax = ax.twinx()
    ax.plot(corners_orientation - np.polyval(poly, np.arange(len(corners_orientation))),
            label="diff", color="gray", alpha=0.2)
    ax.set_ylabel("Error in degrees")
    ax.set_xlabel("Frame")
    plt.title("Approximation of the rotation by constant angle per frame.")
    plt.show()

    return poly[0] # this is angle in degrees rotated per frame


def projection_error(ref_points_2D, obj_points_2D, tform, center=(1920, 1080)):
    """
    For the given ref points computes obj_points_projected by using tform and uses MSE to compare
    their position with the position of given obj_points_2D.

    @param ref_points_2D array of reference points
    @param obj_points_2D array of object points
    @param tform transformation matrix 3x3
    @param center (optional) center of the image where obj_points are. Default value is half of 5K.
    @return square error for each point, and its distance from the center
    """
    ref_homogeneous = np.insert(ref_points_2D, 2, values=1, axis=1)
    obj_reprojected = np.matmul(tform, ref_homogeneous.T).T
    if np.any(obj_reprojected[:, 2] == 0): # invalid projection
        return np.inf, None
    euclidean_coords = np.stack([
            obj_reprojected[:, 0]/obj_reprojected[:, 2],
            obj_reprojected[:, 1]/obj_reprojected[:, 2]
        ], axis=1)
    square_error = np.linalg.norm(euclidean_coords - obj_points_2D, axis=1)
    center_distance = np.sqrt((np.array(obj_points_2D)[:, 0] - center[0]) ** 2 +
                              (np.array(obj_points_2D)[:, 1] - center[1]) ** 2)
    return square_error, center_distance


def get_perspective_by_ransac(ref_points, obj_points, iterations):
    """
    RANSAC for the perspective transformation. From the pool of the points selects quadruple and
    computes reprojection error serveral times. Best result after number of iterations is returned.

    @ref_points list of 3D points
    @obj_points list of 2D points
    @iterations number of random quadruple selections
    @return perspective transformation matrix 3x3 and its reprojection error for whole points set
    """
    def random_partition(ref_points, obj_points):
        indices = np.random.randint(0, len(ref_points), 4)
        return ref_points[indices], obj_points[indices]

    best_reprojection_error = np.inf
    for _ in range(iterations):
        estim_ref_points, estim_obj_points = random_partition(ref_points, obj_points)
        tform = cv2.getPerspectiveTransform(estim_ref_points, estim_obj_points)
        projection_errors, center_distance = projection_error(ref_points, obj_points, tform)
        projection_error_sum = np.sum(projection_errors)

        if projection_error_sum < best_reprojection_error:
            mse_best = tform
            best_reprojection_error = projection_error_sum

    return mse_best, best_reprojection_error


def estimate_center_of_rotation(pts_A, pts_B, rotation_center, angle, correspondence_limit=30):
    """
    For two checkerboards (pts_A and pts_B) and given rotation (angle) the method estimates
    the translation.

    @param pts_A checkerboard A corener points (some can be missing)
    @param pts_B checkerboard B corener points (some can be missing)
    @param image_shape image shape of checkerboard image (center is used as a center of rotation)
    @param angle angle of rotation (in degrees)
    @param correspondence_limit maximum distance between checkerboards corners to be denoted as
        corresponding
    @return translation of the center of rotation
    """

    # First, we simply put the center into the center of the image
    rform = cv2.getRotationMatrix2D(
        rotation_center,
        angle, scale=1)

    # A points are still the A points
    ptsA = np.array(pts_A)
    # But B points are rotated according to rform (image center)
    ptsB = np.matmul(rform, np.insert(np.array(pts_B), 2, values=1, axis=1).T).T

    # This block computes correspondece of the points in the sets. This approach is valid until
    # the distance between A points and corresponding rotated B points is lower correspondence_limit.
    coresp = []
    for ptA in ptsA:
        best_dist = np.inf
        for ptB in ptsB:
            dist = np.linalg.norm(ptA - ptB)
            if dist < correspondence_limit:
                coresp.append((ptA, ptB))
                break

    def shift_err(shift):
        if len(coresp) == 0:
            return np.inf
        dx, dy = shift
        return np.sum([(ptA[0] - ptB[0] - dx) ** 2 + (ptA[1] - ptB[1] - dy) ** 2 for ptA, ptB in coresp])

    # For the correspondence find out the best translation minimizing SSE
    best_shift = minimize(shift_err, x0=[0, 0])

    return best_shift