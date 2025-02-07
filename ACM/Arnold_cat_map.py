import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim

def arnold_cat_map(image, iterations):
    height, width = image.shape[:2]
    result = np.zeros_like(image)
    
    for _ in range(iterations):
        for y in range(height):
            for x in range(width):
                new_x = (2*x + y) % width
                new_y = (x + y) % height
                result[new_y, new_x] = image[y, x]
        image = result.copy()
    
    return result

def image_similarity(img1, img2):
    min_side = min(img1.shape[0], img1.shape[1])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    
    return ssim(img1, img2, data_range=img1.max() - img1.min(), 
                channel_axis=-1 if img1.ndim == 3 else None, 
                win_size=win_size)

# Load the image
image = io.imread('s1-1.jpg')  # Replace with your image path
if image.ndim == 3:
    image = np.mean(image, axis=2).astype(np.uint8)  # Convert to grayscale
original_image = image.copy()

# Parameters
max_iterations = 1000
similarity_threshold = 0.99  # Threshold to consider images as similar

# Apply random number of iterations
random_iterations = np.random.randint(1, 100)
image = arnold_cat_map(image, random_iterations)
print(f"Applied {random_iterations} random iterations")

plt.figure(figsize=(20, 10))
plt.subplot(141)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

iterations = 0
while iterations < max_iterations:
    iterations += 1
    image = arnold_cat_map(image, 1)
    
    similarity = image_similarity(original_image, image)
    
    if iterations % 10 == 0 or similarity > similarity_threshold:
        plt.subplot(142)
        plt.imshow(image, cmap='gray')
        plt.title(f'After {iterations} iterations')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(np.abs(original_image.astype(int) - image.astype(int)), cmap='gray')
        plt.title('Difference')
        plt.axis('off')
        
        plt.subplot(144)
        plt.text(0.5, 0.5, f'Similarity: {similarity:.4f}', ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.pause(0.1)
    
    if similarity > similarity_threshold:
        print(f"Returned close to original after {iterations} iterations")
        break

if iterations == max_iterations:
    print("Max iterations reached without returning to original")

plt.show()