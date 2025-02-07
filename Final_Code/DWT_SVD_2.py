import time
import numpy as np
import pywt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import cv2

def dwt_svd_watermark(image, watermark, alpha=0.01):
    
    # Start time measurement
    start_time = time.time()
    
    # Apply DWT to the image
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Apply SVD to the LL subband
    U, S, V = np.linalg.svd(LL, full_matrices=False)
    
    # Embed watermark
    S_w = S.flatten()
    S_w[:256] = S_w[:256] + alpha * watermark
    S_w = S_w.reshape(S.shape)
    
    # Reconstruct watermarked LL subband
    LL_w = np.dot(U, np.dot(np.diag(S_w), V))
    
    # Reconstruct the watermarked image
    watermarked_coeffs = LL_w, (LH, HL, HH)
    watermarked_image = pywt.idwt2(watermarked_coeffs, 'haar')

    # End time measurement
    end_time = time.time()
    embedding_time = end_time - start_time
    
    # Print the embedding time
    print(f"Time taken for embedding watermark: {embedding_time:.4f} seconds")
    
    return watermarked_image, S

def extract_watermark(watermarked_image, original_S, alpha=0.01):
    # Apply DWT to the watermarked image
    coeffs = pywt.dwt2(watermarked_image, 'haar')
    LL_w, _ = coeffs
    
    # Apply SVD to the watermarked LL subband
    _, S_w, _ = np.linalg.svd(LL_w, full_matrices=False)
    
    # Extract watermark
    extracted_watermark = (S_w.flatten()[:256] - original_S.flatten()[:256]) / alpha
    
    return extracted_watermark


def calculate_psnr(original, watermarked):
    return peak_signal_noise_ratio(original, watermarked)

def calculate_ssim(original, watermarked):
    ssim_value, _ = structural_similarity(original, watermarked, full=True)
    return ssim_value

# --------------------------------------------
def correlation_coefficient(x, y):
    return pearsonr(x, y)[0]

def normalized_cross_correlation(x, y):
    return np.correlate(x, y)[0] / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))

def cosine_similarity(x, y):
    return 1 - cosine(x, y)

def watermark_psnr(original, extracted):
    mse = np.mean((original - extracted) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = max(np.max(original), np.max(extracted))
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def watermark_ssim(original, extracted):
    ssim_value, _ = structural_similarity(original, extracted, full=True)
    return ssim_value
    
#------------------------------------------
# Load the medical X-ray image
medical_image = np.array(Image.open('B3.bmp').convert('L'))

# Generate a random watermark of size 256
watermark = np.random.rand(256)

# Embed watermark
watermarked_image, original_S = dwt_svd_watermark(medical_image, watermark)

# Extract watermark
extracted_watermark = extract_watermark(watermarked_image, original_S)

# Save results
Image.fromarray(watermarked_image.astype(np.uint8)).save('watermarked_xray.jpeg')


# Load the images (assuming grayscale images)
host_image_path = 'B3.bmp'
watermarked_image_path = 'watermarked_xray.bmp'
host_image = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
watermarked_images = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)

# Calculate PSNR and SSIM
# Calculate image quality metrics
psnr_image = calculate_psnr(medical_image, watermarked_image)
ssim_image = calculate_ssim(host_image, watermarked_images)

# Calculate watermark similarity metrics
corr_coef = correlation_coefficient(watermark, extracted_watermark)
norm_cross_corr = normalized_cross_correlation(watermark, extracted_watermark)
cos_sim = cosine_similarity(watermark, extracted_watermark)
psnr_watermark = watermark_psnr(watermark, extracted_watermark)


#ssim_watermark = watermark_ssim(watermark, extracted_watermark)

# Print original and extracted watermarks for comparison
print("Original watermark:", watermark)
print("Extracted watermark:", extracted_watermark)

# Calculate and print the mean squared error between original and extracted watermarks
mse = np.mean((watermark - extracted_watermark)**2)
print("Mean Squared Error:", mse)

# Print results
print("Image Quality Metrics:")
print(f"PSNR (Image): {psnr_image:.2f} dB")
print(f"SSIM (Image): {ssim_image:.4f}")


print("\nWatermark Similarity Metrics:")
print(f"Mean Squared Error: {np.mean((watermark - extracted_watermark)**2):.6f}")
print(f"Correlation Coefficient: {corr_coef:.4f}")
print(f"Normalized Cross-Correlation: {norm_cross_corr:.4f}")
print(f"Cosine Similarity: {cos_sim:.4f}")
print(f"PSNR (Watermark): {psnr_watermark:.2f} dB")
#print(f"SSIM (Watermark): {ssim_watermark:.4f}")