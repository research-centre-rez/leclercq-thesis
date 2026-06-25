import numpy as np
import cv2


def build_circular_mask(width, height, center, radius):
    mask = np.zeros((height, width))
    XX, YY = np.meshgrid(np.arange(width), np.arange(height))
    mask[np.sqrt((XX - center[0]) ** 2 + (YY - center[1]) ** 2) <= radius] = 1
    return mask


def otsu_based_thresholding(img, mask, otsu_factor=2.7, kernel=(3, 3)):
    threshold, _ = cv2.threshold(img[mask == 1], 0, 255, cv2.THRESH_OTSU)
    thresholded = cv2.morphologyEx(
        (img > threshold * otsu_factor).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    )
    thresholded = cv2.morphologyEx(
        (thresholded * mask).astype(np.uint8),
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    )
    return thresholded, threshold * otsu_factor

def light_balance_mask(img, kernel=(131, 131), eps=1e-6):
    lightness = cv2.GaussianBlur(img, kernel, 25).astype(float) + eps
    return lightness


def light_balance(img, kernel=(131, 131), eps=1e-6):
    lightness = cv2.GaussianBlur(img, kernel, 25).astype(float) + eps
    im2 = img.astype(float)
    im2 = im2 / lightness
    return ((im2 - np.min(im2)) / (np.max(im2) - np.min(im2)) * 255).astype(np.uint8)
