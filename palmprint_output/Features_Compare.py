import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os
from skimage.feature import local_binary_pattern, hog

# Feature Extraction Functions
def extract_lbp_features(image, n_points=8, radius=1, method='uniform'):
    lbp = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    fd, _ = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, visualize=True)
    return fd

def lbp_hog_fusion(image):
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    return combined_features

# Calculate similarity scores
def calculate_similarity(feature_vec1, feature_vec2):
    return cosine_similarity([feature_vec1], [feature_vec2])[0][0]

# Calculate FAR and FRR
def calculate_far_frr(genuine_scores, impostor_scores):
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    far = fpr  # False Acceptance Rate
    frr = 1 - tpr  # False Rejection Rate
    
    return far, frr, thresholds

# Main execution
if __name__ == "__main__":
    # Placeholder for image paths - replace with actual dataset paths
    image_paths = ["s1.bmp", "s2.bmp", "s3.bmp", "s4.bmp", "s5.bmp", "s6.bmp", "s7.bmp", "s8.bmp"]  # Add more images for a complete dataset
    # Replace with actual labels (same label for genuine pairs, different for impostors)
    labels = [1, 1, 2, 2, 3, 3, 4, 4]  # Example: 8 images, with pairs of genuine (same) and impostor (different) labels
    
    lbp_scores_genuine, lbp_scores_impostor = [], []
    hog_scores_genuine, hog_scores_impostor = [], []
    fusion_scores_genuine, fusion_scores_impostor = [], []
    
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (150, 150))  # Resizing to 150x150 pixels
        
        lbp_features = extract_lbp_features(img)
        hog_features = extract_hog_features(img)
        fusion_features = lbp_hog_fusion(img)
        
        for j, compare_path in enumerate(image_paths):
            if i != j:  # Avoid comparing the same image
                compare_img = cv2.imread(compare_path, cv2.IMREAD_GRAYSCALE)
                compare_img = cv2.resize(compare_img, (150, 150))  # Resizing to 150x150 pixels
                
                compare_lbp = extract_lbp_features(compare_img)
                compare_hog = extract_hog_features(compare_img)
                compare_fusion = lbp_hog_fusion(compare_img)
                
                lbp_score = calculate_similarity(lbp_features, compare_lbp)
                hog_score = calculate_similarity(hog_features, compare_hog)
                fusion_score = calculate_similarity(fusion_features, compare_fusion)
                
                if labels[i] == labels[j]:  # Genuine pairs
                    lbp_scores_genuine.append(lbp_score)
                    hog_scores_genuine.append(hog_score)
                    fusion_scores_genuine.append(fusion_score)
                else:  # Impostor pairs
                    lbp_scores_impostor.append(lbp_score)
                    hog_scores_impostor.append(hog_score)
                    fusion_scores_impostor.append(fusion_score)
    
    # Calculate FAR and FRR for LBP, HOG, and Fusion
    lbp_far, lbp_frr, lbp_thresholds = calculate_far_frr(lbp_scores_genuine, lbp_scores_impostor)
    hog_far, hog_frr, hog_thresholds = calculate_far_frr(hog_scores_genuine, hog_scores_impostor)
    fusion_far, fusion_frr, fusion_thresholds = calculate_far_frr(fusion_scores_genuine, fusion_scores_impostor)
    
    # Plot FAR vs FRR for comparison
    plt.plot(lbp_far, lbp_frr, label='LBP')
    plt.plot(hog_far, hog_frr, label='HOG')
    plt.plot(fusion_far, fusion_frr, label='Fusion')
    
    plt.xlabel('FAR (False Acceptance Rate)')
    plt.ylabel('FRR (False Rejection Rate)')
    plt.title('FAR vs FRR for LBP, HOG, and Fusion')
    plt.legend()
    plt.show()

    # Find and print EER for each method (where FAR and FRR are closest)
    lbp_eer = lbp_far[np.nanargmin(np.abs(lbp_far - lbp_frr))]
    hog_eer = hog_far[np.nanargmin(np.abs(hog_far - hog_frr))]
    fusion_eer = fusion_far[np.nanargmin(np.abs(fusion_far - fusion_frr))]

    print(f"LBP EER: {lbp_eer:.4f}")
    print(f"HOG EER: {hog_eer:.4f}")
    print(f"Fusion EER: {fusion_eer:.4f}")
