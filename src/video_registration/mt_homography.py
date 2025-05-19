import sys
import cv2 as cv


def compute_and_apply_homography(
    fixed_features,
    moving_features,
    match,
    moving_frame,
    _hom_config,
    resolution,
    frame_id,
):
    """
    Utility function for performing homography matching with multithreading. Allows for processing batches directly.
    """
    if len(match) <= 4:
        return None

    try:
        points_fixed = fixed_features[match[:, 0]]
        points_moved = moving_features[match[:, 1]]
        H, _ = cv.findHomography(points_moved, points_fixed, **_hom_config)
        warped = cv.warpPerspective(moving_frame, H, resolution)

        return (frame_id, H, warped)
    except Exception as e:
        print(e)
        sys.exit()
