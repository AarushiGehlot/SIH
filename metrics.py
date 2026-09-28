import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate_image(ground_truth: np.ndarray, generated: np.ndarray) -> tuple:
    """
    Computes PSNR and SSIM between the original high-res image and the upscaled output.
    Images should be numpy arrays of the same shape.
    """
    # Ensure images are in the same data range (0 to 255)
    # data_range=255 means pixel values span from 0 to 255
    psnr_val = compute_psnr(ground_truth, generated, data_range=255)
    
    # For SSIM, if it's a color image, set channel_axis=2
    if len(generated.shape) == 3:
        ssim_val = compute_ssim(ground_truth, generated, data_range=255, channel_axis=2)
    else:
        ssim_val = compute_ssim(ground_truth, generated, data_range=255)
        
    return round(psnr_val, 2), round(ssim_val, 4)