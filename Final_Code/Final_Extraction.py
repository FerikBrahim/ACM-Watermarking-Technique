import numpy as np
import pywt
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import cv2

def extract_watermark(watermarked_image, original_S, alpha=0.001):
    coeffs = pywt.dwt2(watermarked_image, 'haar')
    LL_w, _ = coeffs
    _, S_w, _ = np.linalg.svd(LL_w, full_matrices=False)
    S_w = S_w.flatten()[:32]
    S = original_S.flatten()[:32]
    extracted_watermark = np.log(S_w / S) / alpha
    return extracted_watermark

def correlation_coefficient(x, y):
    return pearsonr(x, y)[0]

def normalized_cross_correlation(x, y):
    return np.correlate(x, y)[0] / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)))

def cosine_similarity(x, y):
    return 1 - cosine(x, y)

def mean_squared_error(x, y):
    return np.mean((x - y)**2)

def peak_signal_to_noise_ratio(original, extracted):
    mse = mean_squared_error(original, extracted)
    if mse == 0:
        return float('inf')
    max_pixel = max(np.max(original), np.max(extracted))
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def bit_error_rate(original, extracted, threshold=0.5):
    original_binary = (original > threshold).astype(int)
    extracted_binary = (extracted > threshold).astype(int)
    errors = np.sum(original_binary != extracted_binary)
    return errors / len(original)

def compare_watermarks(original_watermark, extracted_watermark):
    metrics = {
        "Correlation Coefficient": correlation_coefficient(original_watermark, extracted_watermark),
        "Normalized Cross-Correlation": normalized_cross_correlation(original_watermark, extracted_watermark),
        "Cosine Similarity": cosine_similarity(original_watermark, extracted_watermark),
        "Mean Squared Error": mean_squared_error(original_watermark, extracted_watermark),
        "PSNR": peak_signal_to_noise_ratio(original_watermark, extracted_watermark),
        "Bit Error Rate": bit_error_rate(original_watermark, extracted_watermark)
    }
    return metrics

# Example usage:
watermarked_image = cv2.imread('watermarked_Cyst_512.png', cv2.IMREAD_GRAYSCALE)

# Load the original S matrix
original_S = np.load('original_S.npy')

# Load the original watermark
original_watermark = np.load('original_watermark.npy')

extracted_watermark = extract_watermark(watermarked_image, original_S)
metrics = compare_watermarks(original_watermark, extracted_watermark)
for metric, value in metrics.items():
   print(f"{metric}: {value}")