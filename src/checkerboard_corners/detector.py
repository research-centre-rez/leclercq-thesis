import cv2
import numpy as np
from checkerboard_corners import INVERT_BLOCK_RADIUS
from skimage.morphology import label
from checkerboard_corners.topology import (
    corner_region_association,
    region_corners_association,
    sides_and_adjoining_regions,
    regions_collection,
    place_square_regions,
    point_pairs_from_component,
    corners_neighborhood
)
import logging
from checkerboard_corners.visualizations import checkerboard_component


def local_maxima(img, radius=3, num=100):
    """
    Locate the most salient local maxima of an image, using a kernel of given size
    :param img : numpy array storing a single-channel image.
    :param radius : distance of the corners from the other closest
    :param num : maximum number of most salient local maxima to return.
    :returns list of at most num (x, y) tuples containing the image coordinates of the found maxima.
    """
    # Dilate and match image and dilated version, getting _all_ the local maxima
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated_img = cv2.dilate(img, structuring_element)
    mx = np.where(img == dilated_img,
                  np.ones(img.shape, dtype=np.int32),
                  np.zeros(img.shape, dtype=np.int32))
    height, width = mx.shape
    local_maxima_coords = []
    for y in range(height):
        for x in range(width):
            if mx[y, x] == 1:
                local_maxima_coords.append((x, y))
    # Sort the local maxima by descending image intensity.
    local_maxima_coords = sorted(local_maxima_coords, key=lambda p: img[p[1], p[0]], reverse=True)
    # Traverse the sorted list, suppress a window of the same radius about each maximum, and
    # select the first "num" that have not been suppressed.
    lms=[]
    for x, y in local_maxima_coords:
        if mx[y, x] == 0:
            # It has already been suppressed, discard it.
            continue
        lms.append((x, y))
        if len(lms) == num:
            break
        # Suppress neighbors to avoid duplicating nearby maxima.
        xmin = max(0, x - radius)
        xmax = min(width, x + radius + 1)
        ymin = max(0, y - radius)
        ymax = min(height, y + radius + 1)
        mx[ymin:ymax, xmin:xmax] = 0
    return lms


def find_corners(grayscale_image, thr, mask,
                 subpix_window=9,
                 zero_zone=1,
                 num_iterations=10,
                 subpix_threshold=0.02,
                 harris_block_size=5,
                 harris_ksize=19,
                 harris_k=0.02,
                 local_maxima_prominence=7,
                 local_maxima_count=450):
    harris_corners = cv2.cornerHarris(thr, harris_block_size, harris_ksize, harris_k)
    corners = np.array(local_maxima(harris_corners, local_maxima_prominence, local_maxima_count))

    corners = np.array([
        corner
        for corner in corners
        if mask[corner[1], corner[0]] == 1
    ])

    if corners.shape[0] == 0:
        return np.array([])

    corners_subpix = cv2.cornerSubPix(
        cv2.GaussianBlur(grayscale_image, (11,11), 4),
        np.float32(np.array(corners).reshape(-1, 1, 2).tolist()),
        (subpix_window, subpix_window),
        (zero_zone, zero_zone),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iterations, subpix_threshold)
    )
    return corners_subpix

def bw_area_detection(corners, bw_img, invert_block_radius=INVERT_BLOCK_RADIUS):
    """
    This function cut off the corners neighborhood and segment the rest of the bw_img into white and black areas. Each area obtains label and color.
    """
    # Build corner neighborhood (square around with invert_block_radius)
    upper_left, lower_right = corners_neighborhood(corners, invert_block_radius, bw_img.shape)

    # remove the corner neighborhood from the white as well as black color
    white_square_segments = bw_img.copy()
    black_square_segments = bw_img.copy()
    for corner_id in np.arange(len(corners)):
        ul = upper_left[corner_id, :]
        lr = lower_right[corner_id, :]
        xx, yy = np.meshgrid(np.arange(ul[0], lr[0]),
                             np.arange(ul[1], lr[1]))
        white_square_segments[yy, xx] = 0
        black_square_segments[yy, xx] = 1

    # label the areas
    white_labels = label(white_square_segments, background=0)
    black_labels = label(black_square_segments, background=1)

    # Join black and white areas into one label image and save their color
    region_colors = {}
    for label_id in np.unique(white_labels) + 1 + np.max(black_labels):
        region_colors[label_id] = 1
    for label_id in - np.unique(black_labels) + 1 + np.max(black_labels):
        region_colors[label_id] = 0
    labels = white_labels - black_labels
    labels[labels != 0] += 1 + np.max(black_labels)

    return labels, region_colors


def detect_checkerboard(
        grayscale_image,
        thr,
        params
    ):
    corner_mask = np.zeros_like(grayscale_image)
    crop = [0, grayscale_image.shape[0], 0, grayscale_image.shape[1]]
    tblr = params["corner_detection_safety_margin_tblr_px"]
    corner_mask[crop[0] + tblr[0]:crop[1] - tblr[1], crop[2] + tblr[2]: crop[3] - tblr[3]] = 1
    corners_subpix = find_corners(grayscale_image, thr, corner_mask,
                                  harris_block_size=params["harris_block_size"] if "harris_block_size" in params else "7",
                                  subpix_threshold=params["subpix_threshold"],
                                  harris_ksize=params["harris_ksize"],
                                  harris_k=params["harris_k"],
                                  local_maxima_prominence=params["local_maxima_prominence"],
                                  local_maxima_count=params["local_maxima_count"])
    logging.debug(f"Found {corners_subpix.shape[0]} corners.")
    # region corresponds with a square of the checkerboard
    region_image, region_id_to_bw_color = bw_area_detection(corners_subpix, thr, invert_block_radius=params["corner_radius_px"])
    corner_id_to_regions_ids = corner_region_association(corners_subpix, region_image, corner_radius=params["corner_radius_px"] + 1)
    region_id_to_corners_ids_dict = region_corners_association(corner_id_to_regions_ids)

    # building of regions and sides must be done above all regions and corners (to prevent missing sides)
    region_sides, sides = sides_and_adjoining_regions(region_id_to_corners_ids_dict)
    regions = regions_collection(region_sides, sides)
    logging.debug(f"Found {len(regions)} regions.")

    # filtration of non-squares must be afterwards
    graph_components, oriented_regions = place_square_regions(regions)

    point_pairs = []
    for component in graph_components:
        try:
            ref_points, obj_points = point_pairs_from_component(component, oriented_regions, corners_subpix)
            point_pairs.append((ref_points, obj_points))
        except AssertionError as ae:
            checkerboard_component(grayscale_image, component, oriented_regions, corners_subpix)
    logging.debug(f"Found {len(point_pairs)} checkerboard components.")
    return point_pairs, corners_subpix

