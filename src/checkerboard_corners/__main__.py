import cv2
import numpy as np
from checkerboard_corners.preprocessor import otsu_based_thresholding, build_circular_mask
from checkerboard_corners.detector import find_corners, bw_area_detection
from checkerboard_corners.topology import corner_region_association, region_corners_association, sides_and_adjoining_regions, regions_collection, place_square_regions, point_pairs_from_component
from checkerboard_corners.visualizations import checkerboard_row, checkerboard_component, non_square_region_corners, corners_without_four_regions, corners
import logging
import multiprocessing
from tqdm.auto import tqdm
import json5 as json

logger = logging.getLogger("checkerboard_corners_parallel_processor")

CROP_CORNERS_DELTA = [18, 10, 15, 15]
CORNER_RADIUS_PX=3
EPS=1 # stability epsilon for division by lightness

def _image_preprocessing(image, crop, scale, light_balance_kernel=(131, 131), rotation_angle=0):
    assert crop is not None, "Crop of the ROI must be manually set."

    if rotation_angle != 0:
        rot_mat = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0]//2), rotation_angle, 1.0)
        im2 = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LANCZOS4)
    else:
        im2 = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)

    lightness = cv2.GaussianBlur(im2, light_balance_kernel, 0).astype(float) + EPS

    im2 = im2.astype(float)
    normalized = im2 / lightness
    return ((normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized)) * 255).astype(np.uint8)


def corners_single(frame_no, image, mask, kwargs):
    try:
        im2 = _image_preprocessing(image, crop, kwargs)
        point_pairs = detect_checkerboard(im2, mask, corner_radius_px=3)
        point_counts = [len(point_pair[0]) for point_pair in point_pairs]
        logger.debug(f"Frame {frame_no}: points detected ({np.sum(point_counts)}) <- {point_counts}")
        return point_pairs
    except Exception as e:
        return None


def detect_checkerboard(
        grayscale_image,
        mask,
        crop,
        corner_radius_px=CORNER_RADIUS_PX
    ):
    otsu_mask = mask
    thr, _ = otsu_based_thresholding(grayscale_image, otsu_mask, otsu_factor=1.5, kernel=(2, 2))
    corner_mask = np.zeros_like(grayscale_image)
    corner_mask[crop[0] + CROP_CORNERS_DELTA[0]: crop[1] - CROP_CORNERS_DELTA[1],
                crop[2] + CROP_CORNERS_DELTA[2]: crop[3] - CROP_CORNERS_DELTA[3]] = 1
    corners_subpix = find_corners(grayscale_image, thr, corner_mask,
                                  harris_block_size=7,
                                  subpix_threshold=0.01,
                                  harris_ksize=31,
                                  harris_k=0.07,
                                  local_maxima_prominence=10,
                                  local_maxima_count=350)
    # region corresponds with a square of the checkerboard
    region_image, region_id_to_bw_color = bw_area_detection(corners_subpix, thr, invert_block_radius=corner_radius_px)
    corner_id_to_regions_ids = corner_region_association(corners_subpix, region_image, corner_radius=corner_radius_px + 1)
    region_id_to_corners_ids_dict = region_corners_association(corner_id_to_regions_ids)

    # building of regions and sides must be done above all regions and corners (to prevent missing sides)
    region_sides, sides = sides_and_adjoining_regions(region_id_to_corners_ids_dict)
    regions = regions_collection(region_sides, sides)

    # filtration of non-squares must be afterwards
    graph_components, oriented_regions = place_square_regions(regions)

    point_pairs = []
    for component in graph_components:
        try:
            ref_points, obj_points = point_pairs_from_component(component, oriented_regions, corners_subpix)
            point_pairs.append((ref_points, obj_points))
        except AssertionError as ae:
            checkerboard_component(grayscale_image, component, oriented_regions, corners_subpix)
    return point_pairs


def video_parallel(input_video, crop, start_frame=0, kwargs={}):

    # Load video frames
    vidcap = cv2.VideoCapture(input_video)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True
    frames = []
    for i in tqdm(range(frame_count), desc="Reading frames"):
        success, frame = vidcap.read()
        if not success:
            break
        frames.append(frame)

    with multiprocessing.Pool(processes=kwargs["cpus"]) as pool:
        point_pairs = pool.map(corners_single, range(len(frames)), frames, crop)


if __name__ == "__main__":
    args = json.load(open("checkerboard_corners/config.json"))
    crop = np.array([715, 830, 880, 1920])
    cpus = multiprocessing.cpu_count() - 1
