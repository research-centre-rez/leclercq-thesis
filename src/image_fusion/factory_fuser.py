from enum import Enum, auto
import logging
import os
import sys
from typing import Union
import numpy as np
import cv2 as cv
from sklearn.decomposition import TruncatedSVD

from utils.filename_builder import append_file_extension, create_out_filename

logger = logging.getLogger(__name__)

class FuseMethod(Enum):
    MIN = auto()
    MAX = auto()
    VAR = auto()
    PCA = auto()
    MED = auto()
    MEAN = auto()

class DefaultFuser():
    def __init__(self, method:FuseMethod) -> None:
        self.method = method
        self.method_name = method.name

    def get_fused_image(self, video_stack:np.ndarray) -> np.ndarray:
        raise NotImplementedError("Each fuser should implement this on its own")

    def save_image_to_disc(self, fused_image:np.ndarray, save_as:str) -> None:
        path, name = os.path.split(save_as)
        base, ext = os.path.splitext(name)

        if ext is None or ext not in [".png", ".jpeg"]:
            raise ValueError(f"Invalid extension, must be one of ['.png', '.jpeg']. Got: {ext}")

        out_filename = os.path.join(path, base)
        out_filename = create_out_filename(out_filename, [], [self.method_name])
        out_filename = append_file_extension(out_filename, ext)

        os.makedirs(path, exist_ok=True)
        cv.imwrite(out_filename, fused_image)

    def _verify_video_stack(self, video_stack: Union[str, np.ndarray]) -> np.ndarray:
        """
        Utility function which allows for taking in a path to a video stack or an `np.ndarray` video stack.
        """
        if isinstance(video_stack, str):
            video_stack = np.load(video_stack)

        if not isinstance(video_stack, np.ndarray):
            raise TypeError(f"Input must be a file path (str) or a NumPy array. Got: {type(video_stack)}")

        return video_stack

class MinFuser(DefaultFuser):
    def __init__(self) -> None:
        super().__init__(FuseMethod.MIN)

    def get_fused_image(self, video_stack: np.ndarray) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        return verified_stack.min(axis=0)

class MaxFuser(DefaultFuser):
    def __init__(self) -> None:
        super().__init__(FuseMethod.MAX)

    def get_fused_image(self, video_stack: np.ndarray) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        return verified_stack.max(axis=0)

class SVDFuser(DefaultFuser):
    def __init__(self) -> None:
        super().__init__(FuseMethod.PCA)
    def get_fused_image(self, video_stack: np.ndarray) -> np.ndarray:
        video_stack = self._verify_video_stack(video_stack)
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

class MeanFuser(DefaultFuser):
    def __init__(self) -> None:
        super().__init__(FuseMethod.MEAN)

    def get_fused_image(self, video_stack: np.ndarray) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        return verified_stack.mean(axis=0)

class MedianFuser(DefaultFuser):
    def __init__(self) -> None:
        super().__init__(FuseMethod.MED)

    def get_fused_image(self, video_stack: np.ndarray) -> np.ndarray:
        verified_stack = self._verify_video_stack(video_stack)
        return np.median(verified_stack, axis=0)


class ImageFuserFactory:
    _registry = {
        FuseMethod.MIN: MinFuser,
        FuseMethod.MAX: MaxFuser, 
        FuseMethod.MEAN: MeanFuser,
        FuseMethod.MED: MedianFuser,
        FuseMethod.PCA: SVDFuser
    }

    @classmethod
    def get_strategy(cls, method: FuseMethod) -> DefaultFuser:
        if method not in cls._registry:
            raise ValueError(f'Fuse method {method} not supported.')
        return cls._registry[method]()
