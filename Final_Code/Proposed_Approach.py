import cv2
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pywt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = cv2.equalizeHist(img)
    return img

def extract_lbp_features(image, n_points=40, radius=3, method='uniform'):
    lbp = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist, lbp

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=True)
    return fd, hog_image

def lbp_hog_fusion(image):
    lbp_features, lbp_image = extract_lbp_features(image)
    hog_features, hog_image = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    return combined_features, lbp_image, hog_image

def visualize_results(original, lbp, hog):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(lbp, cmap='gray')
    ax2.set_title('LBP')
    ax2.axis('off')
    
    ax3.imshow(hog, cmap='gray')
    ax3.set_title('HOG')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_image(image, filename):
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(filename, image_normalized)

def reduce_features(features, n_components=256):
    # اختيار أعلى n من الميزات بناءً على تباينها
    variances = np.var(features, axis=0)  # حساب التباين لكل ميزة
    top_indices = np.argsort(variances)[-n_components:]
    reduced_features = features[:, top_indices]
    
    return reduced_features, top_indices
# -----------------------------------

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
    
# التنفيذ الرئيسي
if __name__ == "__main__":
    
    image_path = "s1.bmp"
    
    output_dir = "palmprint_output"
    os.makedirs(output_dir, exist_ok=True)
    
    palmprint_image = load_and_preprocess(image_path)
    
    combined_features, lbp_image, hog_image = lbp_hog_fusion(palmprint_image)
    
    # watermark = reduced_features
    reduced_features, top_indices = reduce_features(combined_features.reshape(1, -1), n_components=256)
    
    visualize_results(palmprint_image, lbp_image, hog_image)
    
    cv2.imwrite(os.path.join(output_dir, "original_palmprint.jpg"), palmprint_image)
    save_image(lbp_image, os.path.join(output_dir, "lbp_palmprint.jpg"))
    save_image(hog_image, os.path.join(output_dir, "hog_palmprint.jpg"))
    
    print("Original feature vector shape:", combined_features.shape)
    print("Reduced feature vector shape:", reduced_features.shape)
    print("Number of features retained:", reduced_features.shape[1])
    print("Percentage of features retained:", reduced_features.shape[1] / combined_features.shape[0] * 100, "%")
    print("First few elements of the reduced feature vector:", reduced_features[0, :5])
    print(f"Images saved in the '{output_dir}' directory.")

    # Load the medical X-ray image
    medical_image = np.array(Image.open('B3.jpeg').convert('L'))
    
    watermark = reduced_features
    
    # Embed watermark
    watermarked_image, original_S = dwt_svd_watermark(medical_image, watermark)

    # Extract watermark
    extracted_watermark = extract_watermark(watermarked_image, original_S)

    # Save results
    Image.fromarray(watermarked_image.astype(np.uint8)).save('watermarked_xray.jpeg')


    # Load the images (assuming grayscale images)
    host_image_path = 'B3.jpeg'
    watermarked_image_path = 'watermarked_xray.jpeg'
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


    ssim_watermark = watermark_ssim(watermark, extracted_watermark)

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
    print(f"SSIM (Watermark): {ssim_watermark:.4f}")