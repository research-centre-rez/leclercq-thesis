from enum import Enum
import logging
import os
from typing import Union

import numpy as np
import cv2 as cv

from utils.filename_builder import append_file_extension, create_out_filename
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD

logger = logging.getLogger(__name__)

class FuseMethod(Enum):
    MIN = 0
    MAX = 1
    VAR = 2
    PCA = 3
    MED = 4
    MEAN = 5


class ImageFuser:
    """
    Class responsible for creating fused images
    """

    def __init__(self):
        """
        Init method of the ImageFuser class.
        """

    def write_to_disc(self, fused_image: np.ndarray, save_as: str) -> None:
        """
        Writes the fused image to memory.
        """

        path, name = os.path.split(save_as)
        _, ext = os.path.splitext(name)
        if ext is None or ext not in [".png", ".jpeg"]:
            raise ValueError(f"Invalid extension, needs to be either `.png` or `.jpeg`. Got: {ext}")
        os.makedirs(path, exist_ok=True)
        cv.imwrite(save_as, fused_image)

    def write_gallery_to_disc(self, gallery:dict[str, np.ndarray], save_as:str) -> None:
        '''
        Writes the gallery to disc, automatically appending the fuse method to the end of the filename.
        Args:
            gallery (dict[str, np.ndarray]): Gallery obtained from `self.get_fused_gallery`
            save_as (str): how the gallery should be saved as. without the file extension
        Returns:
            None
        '''

        path, _ = os.path.split(save_as)
        os.makedirs(path, exist_ok=True)

        for method, image in gallery.items():
            out_file_name = create_out_filename(save_as, [], [method.lower()])
            out_file_name = append_file_extension(out_file_name, '.png')

            cv.imwrite(out_file_name, image)
            logger.info("Successfully saved %s", out_file_name)

    def get_fused_gallery(
        self, video_stack: Union[str, np.ndarray], methods: list[FuseMethod]
    ) -> dict[str, np.ndarray]:
        vid_stack = self._verify_video_stack(video_stack)
        gallery = {}

        if FuseMethod.MIN in methods:
            gallery["MIN"] = self._get_min_image(vid_stack)

        if FuseMethod.MAX in methods:
            gallery["MAX"] = self._get_max_image(vid_stack)

        if FuseMethod.VAR in methods:
            gallery["VAR"] = self._get_var_image(vid_stack)

        if FuseMethod.PCA in methods:
            gallery["PCA"] = self._get_pca_image(vid_stack)

        if FuseMethod.MED in methods:
            gallery["MED"] = self._get_median_image(vid_stack)

        if FuseMethod.MEAN in methods:
            gallery["MEAN"] = self._get_mean_image(vid_stack)

        return gallery

    def get_fused_image(self, video_stack: Union[str, np.ndarray], method: FuseMethod) -> np.ndarray:
        """
        Returns the fused image for a video stack. Used if you only want to use one image fusing method.
        """

        video_stack = self._verify_video_stack(video_stack)

        if method == FuseMethod.MAX:
            return self._get_max_image(video_stack)

        elif method == FuseMethod.MIN:
            return self._get_min_image(video_stack)

        elif method == FuseMethod.VAR:
            return self._get_var_image(video_stack)

        elif method == FuseMethod.PCA:
            return self._get_pca_image(video_stack)

        elif method == FuseMethod.MEAN:
            return self._get_mean_image(video_stack)

        elif method == FuseMethod.MED:
            return self._get_median_image(video_stack)

    def get_min_mask(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        The min image can be used to create a mask that will only contain the sample.
        """

        # First, we get the min image
        min_img = self._get_min_image(self._verify_video_stack(video_stack))

        # Create a mask
        mask = (min_img > 0).astype(dtype=np.uint8)
        kernel_size = (5, 5)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)
        morphed_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        return morphed_mask

    def _verify_video_stack(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        Utility function which allows for taking in a path to a video stack or an `np.ndarray` video stack.
        """
        if isinstance(video_stack, str):
            video_stack = np.load(video_stack)

        if not isinstance(video_stack, np.ndarray):
            raise TypeError(f"Input must be a file path (str) or a NumPy array. Got: {type(video_stack)}")

        return video_stack

    def _get_min_image(self, video_stack: np.ndarray) -> np.ndarray:
        """
        Returns the minima across the video stack
        """
        return video_stack.min(axis=0)

    def _get_max_image(self, video_stack: np.ndarray) -> np.ndarray:
        """
        Returns the maxima across the video stack
        """
        return video_stack.max(axis=0)

    def _get_var_image(self, video_stack: np.ndarray) -> np.ndarray:
        """
        Returns the variance across the video stack. 
        WARNING:
            `np.var` has a really high space complexity of O = n^2. Therefore it could happen that you run out of RAM memory.
        """
        var_img = video_stack.var(axis=0, dtype=np.float32)

        # normalise the image
        var_img = cv.normalize(var_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        return var_img.astype(np.uint8)

    def _get_pca_image(self, video_stack:np.ndarray) -> np.ndarray:
        n, h, w = video_stack.shape
        reshaped = video_stack.reshape(n, -1)  # shape: (n, h*w)

        # Perform SVD with 1 component
        svd = TruncatedSVD(n_components=1)

        # Use the first principal component to reconstruct the "fused" image
        # Multiply scores by component to get 1D image
        # fused_1d = (svd_img_1d @ svd.components_)  # shape: (n, h*w)
        fused_1d = svd.components_[0]  # shape: (h*w,)

        # Reshape to image
        fused_img = fused_1d.reshape(h, w)

        # Normalize to 8-bit image
        fused_img = cv.normalize(fused_img, None, 0, 255, cv.NORM_MINMAX)
        return fused_img.astype(np.uint8)

    def _get_mean_image(self, video_stack:np.ndarray) -> np.ndarray:
        return video_stack.mean(axis=0)

    def _get_median_image(self, video_stack:np.ndarray) -> np.ndarray:
        return np.median(video_stack, axis=0)
