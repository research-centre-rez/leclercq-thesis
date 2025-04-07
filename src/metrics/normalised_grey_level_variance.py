import numpy as np

def NGLV(image:np.ndarray):
    '''
    Computes Normalised Grey Level Variance (NGLV) for an image.
    Args:
        image (np.ndarray): Grayscale image
    Returns:
        The computed NGLV for the grayscale image
    '''
    mean_intensity = np.mean(image)
    variance       = np.var(image)
    if mean_intensity != 0:
        return variance / (mean_intensity ** 2)

    return 0
