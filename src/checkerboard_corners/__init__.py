import numpy as np
import logging

logger = logging.getLogger(__name__)
INVERT_BLOCK_RADIUS = 14

class Side:
    """
    Side class represents a side between two adjoining squares. I.e. it is two-cornered entity with a unique identifier.

    Methods:
    - __init__(id, corner_A, corner_B): Initializes a side with its unique id and defines its two corner points.
    - flip(): Swaps the two corners of the side.
    - __str__(): Returns a string representation of the side in the format 'side (id): [corner_A corner_B]'.
    """
    def __init__(self, id, corner_A, corner_B, region_A_id, region_B_id):
        self.id = id
        self.corner_A = corner_A
        self.corner_B = corner_B
        self.regions = [region_A_id, region_B_id]

    def flip(self):
        self.corner_A, self.corner_B = self.corner_B, self.corner_A

    def __str__(self):
        return f"side ({self.id}): [{self.corner_A} {self.corner_B}]"


class Region:
    """
    Represents a geometrical region (square). It is defined by corners, sides and adjoining regions.

    Attributes:
    id: Identifier for the region.
    sides: List of sides that make up the region (internally sorted CW or CCW).
    adjoining_regions: List of ids of adjoining region corresponding to each side.
    corners: List of corner points obtained from the sorted sides.

    Methods:
    __init__(id, sides, adjoining_regions):
        Initializes a Region instance with an id, a list of sides, and adjoining regions.
        The sides are sorted in order, and their corresponding corners are extracted.

    sort_sides(sides, return_order):
        A static method that sorts a list of sides in order based on their connectivity
        (i.e., linking corners). Optionally returns the original-to-sorted order mapping.

    orient_region(ref_corners, direction):
        Sets the orientation of the region relative to a set of reference corners
        and an incoming direction. Final orientation is CV or CCV as the reference region and has left-top corner
        as the first corner (orientation is taken from the reference region).

    set_orientation(left_top_corner, flip):
        Sets the left-top corner as the reference point for the region. Optionally flips
        the region's orientation.

    __str__:
        Returns a string representation of the region, detailing each side and its adjoining region.
    """
    def __init__(self, id, sides):
        self.id = id
        self.sides = self.sort_sides(sides)
        self.corners = np.array([side.corner_A for side in self.sides])

    @staticmethod
    def sort_sides(sides, return_order=False):
        last_added = sides[0]
        incoming_order = [side.id for side in sides]
        sorted_sides = [last_added]
        to_sort = sides[1:]
        while len(to_sort) > 0:
            side_added = False
            for side_id, side in enumerate(to_sort):
                if last_added.corner_B == side.corner_A:
                    sorted_sides.append(side)
                    last_added = to_sort.pop(side_id)
                    side_added = True
                    break
                if last_added.corner_B == side.corner_B:
                    side.flip()
                    sorted_sides.append(side)
                    last_added = to_sort.pop(side_id)
                    side_added = True
                    break
            if not side_added:
                raise Exception("Incomplete region")
        outgoing_order = [side.id for side in sorted_sides]
        if return_order:
            return sorted_sides, [incoming_order.index(o) for o in outgoing_order]
        return sorted_sides

    def adjoining_regions_cv(self):
        # We expect that region is already oriented
        out = []
        assert len(self.corners) == 4
        for corner_A, corner_B in zip(self.corners, np.roll(self.corners, -1)):
            adjoining_regions = []
            for side in self.sides:
                if ((side.corner_A == corner_A and side.corner_B == corner_B)
                    or (side.corner_B == corner_A and side.corner_A == corner_B)):
                    adjoining_regions.extend([int(region) for region in side.regions if region != self.id])
            out.append(adjoining_regions)
        return out

    def ref_corners_are_flipped(self, ref_corners):
        corner_A_index = np.where(self.corners == ref_corners[0])[0]
        corner_B_index = np.where(self.corners == ref_corners[1])[0]
        return (corner_A_index - 1) % 4 == corner_B_index

    def orient_region(self, ref_corners, direction):
        """
        Orientation of the region will be set accordingly
        """
        corner_A_index = np.where(self.corners == ref_corners[0])[0]
        corner_B_index = np.where(self.corners == ref_corners[1])[0]

        if direction == (1, 0):  # down
            if self.ref_corners_are_flipped(ref_corners):
                self.set_orientation(corner_B_index, False)
            else:
                self.set_orientation(corner_B_index, True)
        elif direction == (0, -1):  # left
            if self.ref_corners_are_flipped(ref_corners):
                self.set_orientation(corner_B_index - 1, False)
            else:
                self.set_orientation(corner_B_index + 1, True)
        elif direction == (-1, 0):  # up
            if self.ref_corners_are_flipped(ref_corners):
                self.set_orientation(corner_A_index + 1, False)
            else:
                self.set_orientation(corner_A_index - 1, True)
        elif direction == (0, 1):  # right
            if self.ref_corners_are_flipped(ref_corners):
                self.set_orientation(corner_A_index, False)
            else:
                self.set_orientation(corner_A_index, True)

    def set_orientation(self, left_top_corner_index, flip=False):
        self.corners = np.roll(self.corners, -left_top_corner_index, axis=0)
        if flip:
            self.corners = np.roll(self.corners[::-1], 1)

        sorted_sides = []
        for corner_A, corner_B in zip(self.corners, np.roll(self.corners, -1)):
            for side in self.sides:
                if ((side.corner_A == corner_A and side.corner_B == corner_B) or
                    (side.corner_A == corner_B and side.corner_B == corner_A)):
                    sorted_sides.append(side)
        self.sides = sorted_sides

    def get_corners(self, direction):
        """
        This method returns valid corner pair iff the corners are already oriented. Expected orientation is CV starting
        from the left top corner [0].
        Direction is defined from this region to the neighboring region.
        """
        if direction == (1, 0):  # down
            return self.corners[2], self.corners[3]
        elif direction == (0, -1):  # left
            return self.corners[3], self.corners[0]
        elif direction == (-1, 0):  # up
            return self.corners[0], self.corners[1]
        elif direction == (0, 1):  # right
            return self.corners[1], self.corners[2]


    def __str__(self):
        str_repr = f"region {self.id}:\n"
        for side, adjoining_region in zip(self.sides, self.adjoining_regions_cv()):
            str_repr += f"\t{side} \t region ({adjoining_region:d})\n"
        return str_repr
