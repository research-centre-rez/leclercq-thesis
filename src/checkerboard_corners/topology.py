import numpy as np
from checkerboard_corners import Region, Side
from checkerboard_corners import INVERT_BLOCK_RADIUS
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def corners_neighborhood(corners, radius, img_shape):
    upper_left = np.clip(corners[:, 0, :] - radius, 0, np.max(img_shape)).astype(int)
    lower_right = np.stack([
        np.clip(corners[:, 0, 0] + radius + 1, 0, img_shape[1]),
        np.clip(corners[:, 0, 1] + radius + 1, 0, img_shape[0])
    ], axis=1).astype(int)
    return upper_left, lower_right


def corner_region_association(corners, region_image, corner_radius=INVERT_BLOCK_RADIUS + 2):
    upper_left, lower_right = corners_neighborhood(corners, corner_radius, region_image.shape)
    corner_regions_ids = [np.unique(
        region_image[upper_left[cid, 1]: lower_right[cid, 1], upper_left[cid, 0]: lower_right[cid, 0]])
        for cid in np.arange(len(corners))
    ]

    corner_id_to_regions_ids = [np.concatenate([[cid], cri[1:]])
                                 for cid, cri in enumerate(corner_regions_ids)
                                 # if 0 in cri # the inverted area contains zero
                                 #    and len(cri) == 5 # there should be 4 neighboring regions
                                ]

    return corner_id_to_regions_ids


def region_corners_association(corner_id_to_regions_ids):
    # from regions attached to a corner create corners attached to a region
    region_id_to_corners_ids_dict = {}
    for corner_regions in corner_id_to_regions_ids:
        for rid in corner_regions[1:]:
            if rid not in region_id_to_corners_ids_dict:
                region_id_to_corners_ids_dict[rid] = set()
            region_id_to_corners_ids_dict[rid].add(corner_regions[0])

    return region_id_to_corners_ids_dict


def sides_and_adjoining_regions(region_id_to_corners_ids_dict):
    # adjoining regions and region sides
    corner_count = np.max([
        corner
        for region_id, corners in region_id_to_corners_ids_dict.items()
        for corner in corners
    ]) + 1

    region_sides = {}
    side_count = 0
    sides = np.empty((corner_count, corner_count), dtype=object)
    for r_id_ref, ref_corners in region_id_to_corners_ids_dict.items():
        for r_id, region_corners in region_id_to_corners_ids_dict.items():
            side_corners = list(ref_corners.intersection(region_corners))
            if len(side_corners) == 2:  # two corners are common => adjoining regions
                if sides[side_corners[0], side_corners[1]] is None:
                    side = Side(side_count, side_corners[0], side_corners[1], r_id_ref, r_id)
                    sides[side_corners[0], side_corners[1]] = side
                    sides[side_corners[1], side_corners[0]] = side
                    side_count += 1
                else:
                    side = sides[side_corners[0], side_corners[1]]
                if r_id_ref not in region_sides:
                    region_sides[r_id_ref] = set()
                #     adjoining_regions[r_id_ref] = set()
                if r_id not in region_sides:
                    region_sides[r_id] = set()
                #     adjoining_regions[r_id] = set()
                region_sides[r_id_ref].add(side.id)
                region_sides[r_id].add(side.id)
                # adjoining_regions[r_id_ref].add(r_id)
                # adjoining_regions[r_id].add(r_id_ref)
    return region_sides, sorted([side
                                 for side in list(set(sides.reshape(-1).tolist()))
                                 if side is not None], key=lambda x: x.id)


def regions_collection(regions_sides, sides):
    regions = []
    for region_id, side_ids in regions_sides.items():
        try:
            region = Region(region_id, [
                sides[side_id]
                for side_id in list(regions_sides[region_id])
            ])
            regions.append(region)
        except Exception:
            logger.debug(f"Region {region_id} is incomplete")
            logger.debug(f"""Sides {[
                str(sides[side_id])
                for side_id in list(regions_sides[region_id])]}""")
            # incomplete region
            pass

    return regions


def place_square_regions(regions):
    """
    Creates a checkerboard from the regions list (dropping non-squares).
    """
    directions = [
        (-1, 0),  # up
        (0, 1),  # right
        (1, 0),  # down
        (0, -1)  # left
    ]

    regions_buffer = deepcopy(regions)
    oriented_regions = []
    regions_graph = []

    while len(regions_buffer) > 0:
        # place first region
        starting_region = None
        while len(regions_buffer) > 0 and starting_region is None:
            region = regions_buffer.pop(0)
            if len(region.corners) == 4:
                logger.debug(f"Starting region found {region.id}")
                starting_region = region
                oriented_regions.append(starting_region)
                coordinate = (0, 0)
                regions_component = []
                break

        if starting_region is None:
            if len(regions_graph) != 0:
                logger.info("No more square regions found.")
                break
            else:
                raise ValueError("No square regions found. Improve your input data.")

        regions_component.append((coordinate, starting_region.id))

        regions_queue = [
            (adjoining_region_id, starting_region, coordinate, direction)
            for adjoining_regions_id, direction in zip(starting_region.adjoining_regions_cv(), directions)
            for adjoining_region_id in adjoining_regions_id
        ]

        while len(regions_queue) > 0:
            processed_region_id, ref_region, ref_region_coordinate, direction = regions_queue.pop(0)
            # Find out the region object
            processed_regions = [region for region in regions_buffer if region.id == processed_region_id]
            if len(processed_regions) == 0:
                # The region is non-square and was already removed from the regions_buffer
                continue
            assert len(processed_regions) == 1
            processed_region = processed_regions[0]
            regions_buffer.remove(processed_region)
            if len(processed_region.corners) != 4:
                logger.debug(f"Non-square region {processed_region_id} ({len(processed_region.corners)}) ... skipping.")
                continue

            processed_region.orient_region(
                ref_corners=ref_region.get_corners(direction),
                direction=direction
            )

            oriented_regions.append(processed_region)

            coordinate = (ref_region_coordinate[0] + direction[0],
                          ref_region_coordinate[1] + direction[1])
            regions_component.append((coordinate, processed_region.id))
            neighbors = [
                (region_id, processed_region, coordinate, direction)
                for regions_id, direction in zip(processed_region.adjoining_regions_cv(), directions)
                for region_id in regions_id
                if region_id not in [region_placed_id # already placed regions
                                     for _, region_placed_id in regions_component] and
                   region_id not in [region_queued_id # already planned regions
                                     for region_queued_id, _, _, _ in regions_queue]
            ]
            regions_queue.extend(neighbors)
        regions_graph.append(regions_component)

    return regions_graph, oriented_regions


def point_pairs_from_component(graph_component, regions, corners_subpix):
    ref_points = []
    obj_points = []

    mesh_x = [x for (y, x), reg_id in graph_component]
    mesh_y = [y for (y, x), reg_id in graph_component]

    grid_shift = np.min(mesh_y), np.min(mesh_x)
    for (y, x), reg_id in graph_component:
        ref_region = [region for region in regions if region.id == reg_id][0]
        if len(ref_region.corners) != 4:
            break
        for offset_y, offset_x, corner_id in [[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 0, 3]]:
            ref_point = (y - grid_shift[0] + offset_y, x - grid_shift[1] + offset_x)
            if ref_point not in ref_points:
                ref_points.append(ref_point)
                obj_points.append(corners_subpix[ref_region.corners[corner_id], 0, :])
            else:
                point_idx = ref_points.index(ref_point)
                if (obj_points[point_idx][0] != corners_subpix[ref_region.corners[corner_id], :, 0] or
                    obj_points[point_idx][1] != corners_subpix[ref_region.corners[corner_id], :, 1]):
                    logger.debug(f"Subpixels at the same position do not match attached x "
                        f"{obj_points[point_idx][0]} vs. incoming x {corners_subpix[ref_region.corners[corner_id], :, 0]}")
                    ref_points.pop(point_idx)
                    obj_points.pop(point_idx)
    return ref_points, obj_points
