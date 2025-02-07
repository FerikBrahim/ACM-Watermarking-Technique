import numpy as np
import cv2
from scipy.ndimage import rotate
from skimage.transform import AffineTransform, warp
import pywt
import time
from sklearn.metrics import mean_squared_error
from math import log10, sqrt

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def dwt_svd_watermark(image, watermark, alpha=0.1):
    start_time = time.time()
    
    # Apply DWT
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Apply SVD to all subbands
    U_LL, S_LL, V_LL = np.linalg.svd(LL, full_matrices=False)
    U_LH, S_LH, V_LH = np.linalg.svd(LH, full_matrices=False)
    U_HL, S_HL, V_HL = np.linalg.svd(HL, full_matrices=False)
    U_HH, S_HH, V_HH = np.linalg.svd(HH, full_matrices=False)
    
    # Reshape watermark to match the total length of singular values
    total_length = len(S_LL) + len(S_LH) + len(S_HL) + len(S_HH)
    watermark_reshaped = np.resize(watermark, total_length)
    
    # Embed watermark in all subbands
    start = 0
    S_LL_w = S_LL + alpha * watermark_reshaped[start:start+len(S_LL)]
    start += len(S_LL)
    S_LH_w = S_LH + alpha * watermark_reshaped[start:start+len(S_LH)]
    start += len(S_LH)
    S_HL_w = S_HL + alpha * watermark_reshaped[start:start+len(S_HL)]
    start += len(S_HL)
    S_HH_w = S_HH + alpha * watermark_reshaped[start:start+len(S_HH)]
    
    # Reconstruct watermarked subbands
    LL_w = np.dot(U_LL, np.dot(np.diag(S_LL_w), V_LL))
    LH_w = np.dot(U_LH, np.dot(np.diag(S_LH_w), V_LH))
    HL_w = np.dot(U_HL, np.dot(np.diag(S_HL_w), V_HL))
    HH_w = np.dot(U_HH, np.dot(np.diag(S_HH_w), V_HH))
    
    # Apply inverse DWT
    watermarked_coeffs = LL_w, (LH_w, HL_w, HH_w)
    watermarked_image = pywt.idwt2(watermarked_coeffs, 'haar')
    
    end_time = time.time()
    embedding_time = end_time - start_time
    
    print(f"Time taken for embedding watermark: {embedding_time:.4f} seconds")
    
    return watermarked_image, (S_LL, S_LH, S_HL, S_HH)

def extract_watermark(watermarked_image, original_S, alpha=0.1):
    # Apply DWT
    coeffs = pywt.dwt2(watermarked_image, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs
    
    # Apply SVD to all subbands
    _, S_LL_w, _ = np.linalg.svd(LL_w, full_matrices=False)
    _, S_LH_w, _ = np.linalg.svd(LH_w, full_matrices=False)
    _, S_HL_w, _ = np.linalg.svd(HL_w, full_matrices=False)
    _, S_HH_w, _ = np.linalg.svd(HH_w, full_matrices=False)
    
    # Extract watermark from all subbands
    S_LL, S_LH, S_HL, S_HH = original_S
    
    # Ensure the shapes match before subtraction
    w_LL = (S_LL_w[:len(S_LL)] - S_LL) / alpha
    w_LH = (S_LH_w[:len(S_LH)] - S_LH) / alpha
    w_HL = (S_HL_w[:len(S_HL)] - S_HL) / alpha
    w_HH = (S_HH_w[:len(S_HH)] - S_HH) / alpha
    
    extracted_watermark = np.concatenate((w_LL, w_LH, w_HL, w_HH))
    
    return extracted_watermark

# Modify the watermark creation part in the main code:
image = cv2.imread('B3.bmp', 0)  # Load as grayscale
coeffs = pywt.dwt2(image, 'haar')
LL, (LH, HL, HH) = coeffs
watermark_length = LL.size + LH.size + HL.size + HH.size
watermark = np.random.rand(watermark_length)

# Embed watermark
watermarked_image, original_S = dwt_svd_watermark(image, watermark)

# Define attacks (same as before)
def jpeg_compression(image, quality=50):
    _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded_image, 1)

def rescaling(image, scale=0.5):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized = cv2.resize(image, (new_width, new_height))
    return cv2.resize(resized, (width, height))

def rotation(image, angle=30):
    return rotate(image, angle, reshape=False)

def additive_noise(image, std=0.01):
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def affine_transformation(image):
    rows, cols = image.shape[:2]
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    dst_points = np.float32([[cols*0.1,rows*0.1], [cols*0.9,rows*0.1], [cols*0.1,rows*0.9]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    return cv2.warpAffine(image, affine_matrix, (cols,rows))

# Apply attacks and extract watermarks
attacks = [
    ('JPEG Compression', jpeg_compression),
    ('Rescaling', rescaling),
    ('Rotation', rotation),
    ('Additive Noise', additive_noise),
    ('Affine Transformation', affine_transformation)
]

# Modify the attack application part in the main code:
for attack_name, attack_func in attacks:
    print(f"\nApplying {attack_name}...")
    
    # Apply attack
    attacked_image = attack_func(watermarked_image)
    
    # Ensure the attacked image is 2D (grayscale)
    if len(attacked_image.shape) > 2:
        attacked_image = cv2.cvtColor(attacked_image, cv2.COLOR_BGR2GRAY)
    
    # Extract watermark
    extracted_watermark = extract_watermark(attacked_image, original_S)
    
    # Compute metrics
    mse = mean_squared_error(watermark[:len(extracted_watermark)], extracted_watermark)
    psnr_value = psnr(watermark[:len(extracted_watermark)], extracted_watermark)
    correlation = np.corrcoef(watermark[:len(extracted_watermark)], extracted_watermark)[0, 1]
    
    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr_value:.4f} dB")
    print(f"Correlation: {correlation:.4f}")