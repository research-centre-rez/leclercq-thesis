import numpy as np

def brenner_sharpness(image:np.ndarray):
    '''
    Calculates Brenner's image sharpness in both the `x` and `y` directions for a greyscale image. 
    Args:
        image (np.ndarray): Grayscale image
    Returns:
        Sum of Brenner's sharpness in the x and y directions
    '''
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)

    sharpness_x = np.sum(dx**2)
    sharpness_y = np.sum(dy**2)

    return sharpness_x + sharpness_y
