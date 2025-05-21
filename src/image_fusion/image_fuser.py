from enum import Enum
import logging
import os
from typing import Callable, Union

import numpy as np
import cv2 as cv

from utils.filename_builder import append_file_extension, create_out_filename

logger = logging.getLogger(__name__)

class FuseMethod(Enum):
    MIN = 0
    MAX = 1
    VAR = 2


class ImageFuser:
    """
    Class responsible for creating fused images
    """

    def __init__(self, config):
        """
        Init method of the ImageFuser class.
        """

        self.config = config

    def write_to_memory(self, fused_image: np.ndarray, save_as: str) -> None:
        """
        Writes the fused image to memory.
        """

        path, name = os.path.split(save_as)
        _, ext = os.path.splitext(name)
        if ext is None or ext not in [".png", ".jpeg"]:
            raise ValueError(f"Invalid extension, needs to be either `.png` or `.jpeg`. Got: {ext}")
        os.makedirs(path, exist_ok=True)
        cv.imwrite(save_as, fused_image)

    def write_gallery_to_memory(self, gallery:dict[str, np.ndarray], save_as:str) -> None:

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

        return gallery

    def get_fused_image(self, video_stack: Union[str, np.ndarray], method: FuseMethod) -> np.ndarray:
        """
        Returns the fused image for a video stack. The `method` used is the one that was specified at initialisation.
        """

        video_stack = self._verify_video_stack(video_stack)

        if method == FuseMethod.MAX:
            return self._get_max_image(video_stack)

        if method == FuseMethod.MIN:
            return self._get_min_image(video_stack)

        if method == FuseMethod.VAR:
            return self._get_var_image(video_stack)

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
        Returns the variance across the video stack
        """
        return video_stack.var(axis=0, dtype=np.float16)
