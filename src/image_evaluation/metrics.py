import numpy as np
import cv2 as cv


def normalised_grey_level_variance(image_path: str):
    """
    Computes the NGLV metric for an image.
    """

    image_grayscale = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    img_mean = np.mean(image_grayscale)
    img_var = np.var(image_grayscale)

    H, W = image_grayscale.shape
    return img_var / (img_mean)


def brenner_method(image: str):
    """
    Calculate Brenner's focus metric using the formula:
    ϕ = Σ_i Σ_j max(|I(i,j) - I(i+2,j)|, |I(i,j) - I(i,j+2)|)^2

    Args:
        image: 2D numpy array representing a grayscale image

    Returns:
        phi: Brenner's focus metric (float)
    """
    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    # Convert to float for calculations
    img = image
    H, W = img.shape

    # Initialize metric at 0.0
    phi = 0.0

    # Handle images too small for the calculation
    if H < 3 or W < 3:
        return phi

    # Create shifted versions of the image
    down_shifted = np.zeros_like(img)
    right_shifted = np.zeros_like(img)

    # Shift image down by 2 rows (top rows become zero)
    down_shifted[:-2, :] = img[2:, :]

    # Shift image right by 2 columns (left columns become zero)
    right_shifted[:, :-2] = img[:, 2:]

    # Calculate absolute differences
    diff_down = np.abs(img - down_shifted)
    diff_right = np.abs(img - right_shifted)

    # Take element-wise maximum of the two differences
    max_diff = np.maximum(diff_down, diff_right)

    # Square the max differences and sum (only valid regions)
    # Valid region: rows 0 to H-3 and columns 0 to W-3
    valid_region = max_diff[: H - 2, : W - 2]  # Exclude last 2 rows and columns
    phi = np.sum(valid_region**2)

    return phi
