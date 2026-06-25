import numpy as np

def create_checkerboard(width, height, square_size, color1=0, color2=255):
    """
    width, height: image size in pixels
    square_size: size of one square in pixels
    color1, color2: BGR colors (OpenCV convention)

    returns: uint8 image (H, W, 3)
    """
    img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which square we are in
            sx = x // square_size
            sy = y // square_size

            if (sx + sy) % 2 == 0:
                img[y, x] = color1
            else:
                img[y, x] = color2

    return img