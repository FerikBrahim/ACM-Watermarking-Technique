import time
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt

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
    
    LL_scrambled = arnold_cat_map(LL)

    # Apply SVD to the LL subband
    U, S, V = np.linalg.svd(LL_scrambled, full_matrices=False)
    
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

# Load the host image
host_image = cv2.imread('b3.jpeg', cv2.IMREAD_GRAYSCALE)

# Transform the host image using DWT
coeffs = pywt.dwt2(host_image, 'haar')
LL, (LH, HL, HH) = coeffs

# Apply Arnold Cat Map to the LL sub-band
def arnold_cat_map(image):
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = image[(i + j) % rows, (i + 2*j) % cols]
    return result

LL_scrambled = arnold_cat_map(LL)

# Insert the watermark feature vector into the LL sub-band using SVD
u, s, vh = np.linalg.svd(LL_scrambled, full_matrices=False)

# Ensure the watermark_features length matches the singular values' length
n_singular_values = len(s)
if watermark_features.shape[0] > n_singular_values:
    watermark_features = watermark_features[:n_singular_values]

# Embed watermark into singular values
S_watermarked = s + np.mean(s) * watermark_features.flatten()

# Reconstruct LL with the watermarked singular values
LL_watermarked = np.dot(u * S_watermarked, vh)

# Reconstruct the watermarked image using inverse DWT
watermarked_image = pywt.idwt2((LL_watermarked, (LH, HL, HH)), 'haar')
watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

# Save the watermarked host image
cv2.imwrite('watermarked_host_image.jpg', watermarked_image)

# Extraction process
# Load the watermarked host image
watermarked_host_image = cv2.imread('watermarked_host_image.jpg', cv2.IMREAD_GRAYSCALE)

# Transform the watermarked host image using DWT
coeffs = pywt.dwt2(watermarked_host_image, 'haar')
LL, _ = coeffs

# Apply Arnold Cat Map to the LL sub-band
LL_scrambled = arnold_cat_map(LL)

# Extract the watermark feature vector using SVD
u, s, vh = np.linalg.svd(LL_scrambled, full_matrices=False)
extracted_watermark_features = s - np.mean(s)


# Calculate the similarity between the original and extracted watermark feature vectors
original_watermark_features = watermark_features
extracted_watermark_features = extracted_watermark_features.flatten()

similarity = np.dot(original_watermark_features.flatten(), extracted_watermark_features) / (np.linalg.norm(original_watermark_features) * np.linalg.norm(extracted_watermark_features))
print("Similarity:", similarity)

# Visualize the original and extracted watermark feature vectors
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_watermark_features.reshape(16, 16), cmap='gray')
plt.title('Original Watermark Feature Vector')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(extracted_watermark_features.reshape(16, 16), cmap='gray')
plt.title('Extracted Watermark Feature Vector')
plt.axis('off')

plt.show()
