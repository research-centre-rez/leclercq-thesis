import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator
import logging

logger = logging.getLogger(__name__)

class SmoothTPSWarp:
    def __init__(self, dst_points, src_points, smoothing=2.0):
        """
        Fit inverse warp: target plane -> source image.

        dst_points: (N, 2) target/output coordinates
        src_points: (N, 2) corresponding source/input coordinates
        smoothing: larger => smoother / less sensitive to noisy correspondences
        """
        self.dst_points = np.asarray(dst_points, dtype=np.float64)
        self.src_points = np.asarray(src_points, dtype=np.float64)

        # Fit x and y source coordinates as smooth functions of target coords
        self.fx = RBFInterpolator(
            self.dst_points,
            self.src_points[:, 0],
            kernel='thin_plate_spline',
            smoothing=smoothing
        )
        self.fy = RBFInterpolator(
            self.dst_points,
            self.src_points[:, 1],
            kernel='thin_plate_spline',
            smoothing=smoothing
        )

    def map_points(self, pts):
        """
        Map target/output points -> source/input points.
        pts: (M, 2)
        returns: (M, 2)
        """
        pts = np.asarray(pts, dtype=np.float64)
        x = self.fx(pts)
        y = self.fy(pts)
        return np.column_stack([x, y])

def warp_image_with_tps(image, warp, out_shape):
    """
    Warp an input image using a Thin Plate Spline (TPS) transformation.

    This function takes an input image and applies a warp transformation
    defined by a Thin Plate Spline model to produce the output image with
    the desired shape. The mapping from the target output coordinates to
    the source coordinates is performed using a callable 'warp' object,
    and the actual sampling from the source image is done via bilinear
    interpolation.

    :param image: The input image to be warped, represented as a
                  2D or 3D NumPy array.
    :param warp: A callable object that maps a set of target pixel
                 coordinates to corresponding source pixel coordinates
                 in the input image.
    :param out_shape: A tuple specifying the dimensions of the output
                      image in (height, width).
    :return: The warped output image represented as a transposed 2D or
             3D NumPy array, depending on the input image format.
    """
    out_h, out_w = out_shape

    # Grid of output pixel coordinates
    xx, yy = np.meshgrid(np.arange(out_w), np.arange(out_h))
    grid = np.column_stack([xx.ravel(), yy.ravel()])  # target coords

    # Map each output pixel to input image coordinates
    src_coords = warp.map_points(grid)

    map_x = src_coords[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_coords[:, 1].reshape(out_h, out_w).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped.T

def from_checkerboard_corners_to_image_coordinates(refs, objs):
    """
    Transforms the reference coordinates and object coordinates from the checkerboard corner
    system to image-based coordinates.

    This function calculates the spatial mapping by determining the bounds of the object
    coordinates and the steps needed in reference space. It interpolates the coordinate system
    to produce a transformed mapping that aligns with the image coordinates.

    :param refs: The reference corner indices in the checkerboard coordinate system.
    :type refs: numpy.ndarray
    :param objs: The object coordinates representing the positions in the physical system.
    :type objs: numpy.ndarray
    :return: A tuple containing the transformed image-based coordinates and the shifted object
        coordinates relative to the top-left corner.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]
    """
    left_top = np.min(objs, axis=0)
    right_bottom = np.max(objs, axis=0)
    steps = np.max(refs, axis=0) + 1

    x_space = np.linspace(0, right_bottom[0] - left_top[0], np.ceil(steps[0]).astype(int))
    y_space = np.linspace(0, right_bottom[1] - left_top[1], np.ceil(steps[1]).astype(int))

    xx, yy = np.meshgrid(x_space, y_space)

    xs = []
    ys = []
    for ref in refs:
        xs.append(xx[tuple(ref[::-1])])
        ys.append(yy[tuple(ref[::-1])])

    return np.stack([xs, ys], axis=1) + left_top, objs


import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator


class TPSWarp:
    """
    Thin-plate spline warp for 2D point sets with variable number of points per frame.

    Input to fit():
        src_points_list = [array(N0,2), array(N1,2), ...]
        dst_points_list = [array(N0,2), array(N1,2), ...]

    Convention:
        src = coordinates in original/distorted image
        dst = desired coordinates in corrected/reference space

    For image warping we fit the inverse map:
        dst -> src
    because cv2.remap needs, for each output pixel, where to sample in the input image.
    """

    def __init__(
        self,
        smoothing=0.0,
        neighbors=None,
        reduce_grid_step=None,
        dtype=np.float32,
    ):
        self.smoothing = float(smoothing)
        self.neighbors = neighbors
        self.reduce_grid_step = reduce_grid_step
        self.dtype = dtype

        self._rbf_inv = None
        self._map_cache = {}

    @staticmethod
    def _to_xy_array(pts, name="points"):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"{name} must have shape (N,2), got {pts.shape}")
        return pts

    @classmethod
    def _flatten_correspondence_lists(cls, src_points_list, dst_points_list):
        if len(src_points_list) != len(dst_points_list):
            raise ValueError(
                f"src_points_list and dst_points_list must have same length, "
                f"got {len(src_points_list)} and {len(dst_points_list)}"
            )

        src_all = []
        dst_all = []

        for i, (src, dst) in enumerate(zip(src_points_list, dst_points_list)):
            src = cls._to_xy_array(src, name=f"src_points_list[{i}]")
            dst = cls._to_xy_array(dst, name=f"dst_points_list[{i}]")

            if len(src) != len(dst):
                raise ValueError(
                    f"Frame {i}: src and dst must have same number of points, "
                    f"got {len(src)} and {len(dst)}"
                )

            if len(src) == 0:
                continue

            src_all.append(src)
            dst_all.append(dst)

        if not src_all:
            raise ValueError("No valid correspondences were provided.")

        src_all = np.vstack(src_all)
        dst_all = np.vstack(dst_all)
        return src_all, dst_all

    @staticmethod
    def _reduce_controls(src_xy, dst_xy, step):
        """
        Optionally average nearby control points in destination space.
        This is useful if you have a lot of repeated checkerboard observations.
        """
        if step is None or step <= 0:
            return src_xy, dst_xy

        cell = np.floor(dst_xy / float(step)).astype(np.int64)
        unique_cells, inv = np.unique(cell, axis=0, return_inverse=True)

        src_sum = np.zeros((len(unique_cells), 2), dtype=np.float64)
        dst_sum = np.zeros((len(unique_cells), 2), dtype=np.float64)
        counts = np.zeros(len(unique_cells), dtype=np.int64)

        for i, k in enumerate(inv):
            src_sum[k] += src_xy[i]
            dst_sum[k] += dst_xy[i]
            counts[k] += 1

        src_red = src_sum / counts[:, None]
        dst_red = dst_sum / counts[:, None]
        return src_red, dst_red

    def fit(self, src_points_list, dst_points_list):
        """
        Fit inverse TPS warp:
            dst -> src

        Parameters
        ----------
        src_points_list : list of (Ni,2) arrays
            Points in distorted/original image.
        dst_points_list : list of (Ni,2) arrays
            Corresponding points in corrected/reference space.
        """
        src_all, dst_all = self._flatten_correspondence_lists(
            src_points_list, dst_points_list
        )

        src_all, dst_all = self._reduce_controls(
            src_all, dst_all, self.reduce_grid_step
        )

        print(f"Fitting TPS warp with {len(src_all)} points.")

        self._rbf_inv = RBFInterpolator(
            y=dst_all,                  # input query points
            d=src_all,                  # predicted source coords
            kernel="thin_plate_spline",
            smoothing=self.smoothing,
            neighbors=self.neighbors,
            degree=1,
        )

        self._map_cache.clear()
        return self

    def transform_points(self, pts_dst):
        """
        Transform points from corrected/reference space to source/distorted space.
        """
        if self._rbf_inv is None:
            raise RuntimeError("TPSWarp is not fitted yet.")

        pts_dst = self._to_xy_array(pts_dst, name="pts_dst")
        return np.asarray(self._rbf_inv(pts_dst), dtype=np.float64)

    def build_remap(self, out_shape):
        """
        Build cv2.remap maps for output image of shape (H, W).
        """
        if self._rbf_inv is None:
            raise RuntimeError("TPSWarp is not fitted yet.")

        out_shape = tuple(map(int, out_shape))
        if out_shape in self._map_cache:
            return self._map_cache[out_shape]

        h, w = out_shape
        yy, xx = np.indices((h, w), dtype=np.float64)
        pts_dst = np.column_stack([xx.ravel(), yy.ravel()])

        pts_src = self._rbf_inv(pts_dst)
        map_x = pts_src[:, 0].reshape(h, w).astype(self.dtype)
        map_y = pts_src[:, 1].reshape(h, w).astype(self.dtype)

        self._map_cache[out_shape] = (map_x, map_y)
        return map_x, map_y

    def warp_image(
        self,
        image,
        out_shape=None,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=0,
    ):
        """
        Warp source image into corrected/reference geometry.
        """
        if out_shape is None:
            out_shape = image.shape[:2]

        map_x, map_y = self.build_remap(out_shape)

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

    def reprojection_error(self, src_points_list, dst_points_list):
        """
        Compute reprojection errors on given correspondences.
        Returns dict with rmse, median, max and all per-point errors.
        """
        src_all, dst_all = self._flatten_correspondence_lists(
            src_points_list, dst_points_list
        )
        pred_src = self.transform_points(dst_all)
        err = np.linalg.norm(pred_src - src_all, axis=1)

        return {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "median": float(np.median(err)),
            "max": float(np.max(err)),
            "n": int(len(err)),
            "errors": err,
        }