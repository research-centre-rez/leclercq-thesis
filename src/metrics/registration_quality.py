import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as mi

def compute_structural_similarity(img_stack:np.ndarray):
    ref_image = img_stack[15]
    ssim_values = []

    # Determine if the images are multichannel (e.g., RGB)
    multichannel = (ref_image.ndim == 3 and ref_image.shape[2] in [3, 4])
    
    # Loop over the rest of the images and compute SSIM
    for idx, img in enumerate(img_stack[15:], start=1):
        score = mi(ref_image, img, bins=600)
        ssim_values.append(score)
        print(f"MI between reference and image {idx}: {score:.4f}")
    
    average_ssim = np.mean(ssim_values)
    return average_ssim, ssim_values

def main():
    img_stack = np.load('./npy_files/temp_3A-part0_rotated_short_registered.npy')
    avg_ssim, _ = compute_structural_similarity(img_stack)
    print(f"Average_ssim: {avg_ssim}")
if __name__ == "__main__":
    main()
