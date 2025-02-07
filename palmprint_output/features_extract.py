import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
import os

def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = cv2.equalizeHist(img)
    return img

def extract_lbp_features(image, n_points=8, radius=1, method='uniform'):
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
    print(len(lbp_features))
    hog_features, hog_image = extract_hog_features(image)
    print(len(hog_features))
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
    # Normalize the image to 0-255 range
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(filename, image_normalized)

# Main execution
if __name__ == "__main__":
    # Replace with the path to your palmprint image
    image_path = "s1.bmp"
    
    # Create output directory if it doesn't exist
    output_dir = "palmprint_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess the image
    palmprint_image = load_and_preprocess(image_path)
    
    # Apply LBP-HOG fusion
    combined_features, lbp_image, hog_image = lbp_hog_fusion(palmprint_image)
    
    # Visualize results
    visualize_results(palmprint_image, lbp_image, hog_image)
    
    # Save images
    cv2.imwrite(os.path.join(output_dir, "original_palmprint.jpg"), palmprint_image)
    save_image(lbp_image, os.path.join(output_dir, "lbp_palmprint.jpg"))
    save_image(hog_image, os.path.join(output_dir, "hog_palmprint.jpg"))
    
    print("Combined feature vector shape:", combined_features.shape)
    print("First few elements of the combined feature vector:", combined_features[:10])
    print(f"Images saved in the '{output_dir}' directory.")

    # Calculate the minimum and maximum values
min_value = np.min(combined_features)
max_value = np.max(combined_features)

print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")