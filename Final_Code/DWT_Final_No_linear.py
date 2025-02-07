import time
import numpy as np
import pywt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import cv2

def dwt_svd_watermark(image, watermark, alpha=0.001):
    start_time = time.time()

    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    U, S, V = np.linalg.svd(LL, full_matrices=False)

    S_w = S.flatten()
    S_w[:32] = S_w[:32] * (1 + alpha * watermark[:32])  # Non-linear embedding
    S_w = S_w.reshape(S.shape)

    LL_w = np.dot(U, np.dot(np.diag(S_w), V))

    watermarked_coeffs = LL_w, (LH, HL, HH)
    watermarked_image = pywt.idwt2(watermarked_coeffs, 'haar')

    end_time = time.time()
    embedding_time = end_time - start_time

    print(f"Time taken for embedding watermark: {embedding_time:.4f} seconds")

    return watermarked_image, S

def extract_watermark(watermarked_image, original_S, alpha=0.001):
    coeffs = pywt.dwt2(watermarked_image, 'haar')
    LL_w, _ = coeffs

    _, S_w, _ = np.linalg.svd(LL_w, full_matrices=False)

    S_w = S_w.flatten()[:32]
    S = original_S.flatten()[:32]

    extracted_watermark = np.log(S_w / S) / alpha  # Non-linear extraction

    return extracted_watermark



def calculate_psnr(original, watermarked):
    return peak_signal_noise_ratio(original, watermarked)

def calculate_ssim(host_image, watermarked_image):
    # Open the images
    host_img = Image.open(host_image)
    watermarked_img = Image.open(watermarked_image)

    # Convert images to grayscale
    host_gray = host_img.convert('L')
    watermarked_gray = watermarked_img.convert('L')

    # Convert to numpy arrays
    host_array = np.array(host_gray)
    watermarked_array = np.array(watermarked_gray)

    # Calculate SSIM
    ssim_value = ssim(host_array, watermarked_array)

    return ssim_value

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

# Function to calculate Bit Error Rate (BER)
def bit_error_rate(original_watermark, extracted_watermark, threshold=1e-6):
    # Convert to binary using a threshold
    original_binary = (original_watermark > 0.5).astype(int)
    extracted_binary = (extracted_watermark > 0.5).astype(int)
    
    # Calculate BER
    errors = np.sum(original_binary != extracted_binary)
    total_bits = len(original_watermark)
    
    return errors / total_bits
# Load the medical X-ray image
medical_image = np.array(Image.open('Dataset/airplane.png').convert('L'))

# Generate a random watermark of size 32
watermark = np.random.rand(32)  # Reduced to 32

# Save the original watermark
np.save('original_watermark.npy', watermark)
print('-------------------------------------------')
print(watermark)
# Embed watermark
watermarked_image, original_S = dwt_svd_watermark(medical_image, watermark)

# Save the original S matrix
np.save('original_S.npy', original_S)
print('-------------------------------------------')
print(original_S)
# Extract watermark
extracted_watermark = extract_watermark(watermarked_image, original_S)

# Save results
Image.fromarray(watermarked_image.astype(np.uint8)).save('watermarked_airplane.png')

# Load the images
host_image = cv2.imread('Dataset/airplane.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = cv2.imread('watermarked_airplane.png', cv2.IMREAD_GRAYSCALE)

# Calculate image quality metrics
psnr_image = calculate_psnr(medical_image, watermarked_image)
ssim_image = calculate_ssim('Dataset/airplane.png', 'watermarked_airplane.png')

# Calculate watermark similarity metrics
corr_coef = correlation_coefficient(watermark, extracted_watermark)
norm_cross_corr = normalized_cross_correlation(watermark, extracted_watermark)
cos_sim = cosine_similarity(watermark, extracted_watermark)
psnr_watermark = watermark_psnr(watermark, extracted_watermark)
ber_value = bit_error_rate(watermark, extracted_watermark)
# Print results
print("Image Quality Metrics:")
print(f"PSNR (Image): {psnr_image:.2f} dB")
print(f"SSIM (Image): {ssim_image:.4f}")

print("\nWatermark Similarity Metrics:")
print(f"Mean Squared Error: {np.mean((watermark - extracted_watermark)**2):.6f}")
print(f"Correlation Coefficient: {corr_coef:.4f}")
print(f"Normalized Cross-Correlation: {norm_cross_corr:.4f}")
print(f"Cosine Similarity: {cos_sim:.4f}")
print(f'Bit Error Rate (BER): {ber_value:.6f}')
print(f"PSNR (Watermark): {psnr_watermark:.2f} dB")