import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Polygon


def corners(background_image, corners_subpix):
    plt.imshow(background_image, cmap="gray")
    plt.scatter(corners_subpix[:, 0, 0], corners_subpix[:, 0, 1], c='r', marker='+')
    plt.title("Corners in non-masked area")
    plt.show()


def checkerboard_row(background_image, point_pairs, row_no=7):
    row_no = 7
    plt.imshow(background_image, cmap="gray")
    ref_points, obj_points = point_pairs
    to_draw = np.array([b
                        for a, b in zip(ref_points, obj_points)
                        if a[1] == row_no])
    plt.scatter(to_draw[:, 0], to_draw[:, 1], c='r', marker='+')
    plt.title(f"{row_no}th Row of the checkerboard")
    plt.show()


def checkerboard_component(background_image, graph_component, regions, corners_subpix):
    plt.figure(figsize=(15, 15))
    ax = plt.subplot(111)
    ax.imshow(background_image, cmap='gray')
    for reg_order, (pos, region_id) in enumerate(graph_component):
        cs = np.array(list([region.corners for region in regions if region.id == region_id][0]))
        coords = corners_subpix[cs, 0, :]
        rect = Polygon(coords, alpha=0.8, color="blue" if reg_order < 5 else "red" if reg_order < 25 else "yellow")
        ax.text(np.min(coords[:, 0]) + 10, np.mean(coords[:, 1]), f"{pos}")
        ax.add_patch(rect)
    plt.xlim(1000, 2800)
    plt.show()


def non_square_region_corners(background_image, non_square_regions, corners_subpix, region_id_to_corners_ids_dict):
    plt.imshow(background_image)
    non_rect = [corner_id for reg_id in non_square_regions for corner_id in region_id_to_corners_ids_dict[reg_id]]
    plt.scatter(corners_subpix[non_rect, :, 0], corners_subpix[non_rect, :, 1], c='r', marker='+')
    plt.title("Corners belonging to non-square regions")
    plt.show()


def corners_without_four_regions(background_image, corner_id_to_regions_ids, corners_subpix):
    plt.imshow(background_image)
    non_rect = np.array([corner[0] for corner in corner_id_to_regions_ids if len(corner) != 5])
    #corners_rect = np.array([corner for corner in corner_id_to_regions_ids if len(corner) == 5])
    plt.scatter(corners_subpix[non_rect, :, 0], corners_subpix[non_rect, :, 1], c='r', marker='+')
    plt.title("Corners where region count is not equal to 4.")
    plt.show()