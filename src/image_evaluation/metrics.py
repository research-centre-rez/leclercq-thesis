from enum import Enum, auto
from typing import Union
import numpy as np
import cv2 as cv
import sys

from torch.serialization import normalize_storage_type
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class normType(Enum):
    std = auto()
    minmax = auto()
    equalHist = auto()
    clahe = auto()
    mean_sub = auto()
    l2 = auto()
    grad_mag = auto()
    grad_equal_hist = auto()


class Metric:

    def __init__(self) -> None:
        pass

    def calculate_metric(self, img_path: str, normalise: bool) -> np.ndarray:
        raise ValueError("Each subclass should implement this on its own")

    def _normalise_img(self, image: np.ndarray, normType: normType):
        image = image.astype(np.float64)

        if normType == normType.std:
            return (image - np.mean(image)) / (np.std(image + 1e-8))

        if normType == normType.minmax:
            return (image - image.min()) / (image.max() - image.min() + 1e-8)

        if normType == normType.equalHist:
            return cv.equalizeHist(image.astype(np.uint8))

        if normType == normType.clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image.astype(np.uint8))

        if normType == normType.mean_sub:
            return image - image.mean()

        if normType == normType.l2:
            return image - (np.linalg.norm(image) + 1e-8)

        if normType == normType.grad_mag:
            gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mag /= grad_mag.max() + 1e-8
            return grad_mag

        if normType == normType.grad_equal_hist:
            gx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mag = 255 * (grad_mag / (grad_mag.max() + 1e-8))
            grad_mag_uint8 = grad_mag.astype(np.uint8)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(grad_mag_uint8)

        return image

    def _mask_sample(selt, img) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        radius = h // 2
        centre = (w // 2, h // 2)
        cv.circle(mask, centre, radius, 255, thickness=-1)
        masked_img = cv.bitwise_and(img, img, mask=mask)
        return masked_img

    def _load_img_to_memory(self, img_path: str, use_float: bool) -> np.ndarray:
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        if use_float:
            image = self._mask_sample(image)
            image = self._normalise_img(image, normType.grad_mag)
            return image

        return image

    def _preprocess_image(
        self, img_path: Union[str, np.ndarray], normalise: bool
    ) -> np.ndarray:
        if isinstance(img_path, str):

            image = self._load_img_to_memory(img_path, use_float=True)
            if normalise:
                return self._normalise_img(image, normType.grad_mag)
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0

        elif isinstance(img_path, np.ndarray):
            image = self._mask_sample(img_path)
            if normalise:
                image = self._normalise_img(image, normType.grad_mag)

            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0

        return image


class NGLV(Metric):

    def calculate_metric(
        self, img_path: Union[str, np.ndarray], normalise: bool
    ) -> np.ndarray:

        image = self._preprocess_image(img_path, normalise)

        mean, std_dev = cv.meanStdDev(image)
        result = std_dev[0] ** 2 / mean[0]
        return result[0]


class BrennerMethod(Metric):

    def calculate_metric(
        self, img_path: Union[str, np.ndarray], normalise: bool
    ) -> np.ndarray:
        image = self._preprocess_image(img_path, normalise)

        diff_x = np.abs(np.subtract(image[:-2, 2:], image[:-2, :-2], dtype=np.float64))
        diff_y = np.abs(np.subtract(image[2:, :-2], image[:-2, :-2], dtype=np.float64))

        raw_brenner = np.sum(np.maximum(diff_x, diff_y) ** 2)
        return raw_brenner / image.size


class AbsoluteGradient(Metric):

    def calculate_metric(
        self, img_path: Union[str, np.ndarray], normalise: bool
    ) -> np.ndarray:
        image = self._preprocess_image(img_path)

        I_x = np.abs(np.diff(image, 1, 1, 0))
        I_y = np.abs(np.diff(image, 1, 0, 0))
        grad = np.maximum(I_x, I_y)
        return np.sum(grad) / grad.size

        norm_grad = np.sum(grad) / grad.size  # average gradient per pixel
        return norm_grad


class VarianceOfLaplacian(Metric):

    def calculate_metric(self, img_path: str) -> np.ndarray:
        image = self._load_img_to_memory(img_path, True)

        laplacian = cv.Laplacian(image, cv.CV_64F)[1:-1, 1:-1].var()
        return laplacian


class Tenengrad(Metric):

    def calculate_metric(self, img_path: str, normalise: bool) -> np.ndarray:
        image = self._load_img_to_memory(img_path, True)
        grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, 3)
        grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, 3)
        return np.sum(grad_x**2 + grad_y**2)


class SSIM(Metric):
    def _load_video_stack(self, stack_path: str) -> np.ndarray:
        return np.load(stack_path)

    def calculate_metric(self, stack_path: str, normalise: bool) -> np.ndarray:
        video_stack = self._load_video_stack(stack_path)
        # video_stack = video_stack.astype(float) / 255
        mean_img = np.mean(video_stack, axis=0)
        print(video_stack.max())
        print(mean_img.max())
        print("Calculating metric..")
        ssim_scores = [
            ssim(img, mean_img, data_range=255, full=False) for img in video_stack
        ]
        print(f"Mean SSIM scores: {np.mean(ssim_scores)}")
        return np.mean(ssim_scores)


class MutualInformation(Metric):
    def _load_video_stack(self, stack_path: str) -> np.ndarray:
        return np.load(stack_path)

    def _mutual_information(
        self, img1: np.ndarray, img2: np.ndarray, bins: int = 256
    ) -> float:
        """
        Computes mutual information between two images using histogram estimation.

        Args:
            img1, img2: 2D numpy arrays (grayscale images)
            bins: number of bins for joint histogram

        Returns:
            Normalized mutual information score (float)
        """
        hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
        pxy = hgram / np.sum(hgram)
        px = np.sum(pxy, axis=1)  # marginal for x
        py = np.sum(pxy, axis=0)  # marginal for y

        px_py = px[:, None] * py[None, :]
        nz = pxy > 0
        mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))
        return mi

    def calculate_metric(self, stack_path: str, normalise: bool) -> float:
        video_stack = self._load_video_stack(stack_path)
        mean_img = np.max(video_stack, axis=0)

        mi_scores = [self._mutual_information(img, mean_img) for img in video_stack]

        print(f"Mean MI scores: {np.mean(mi_scores)}")
        return np.mean(mi_scores)
