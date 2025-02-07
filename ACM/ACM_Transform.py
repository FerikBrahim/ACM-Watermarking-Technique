import numpy as np
import cv2
import matplotlib.pyplot as plt

def arnold_cat_map(image, iterations):
    height, width = image.shape
    n = height  # assuming square image
    
    # Create a copy of the image
    transformed_image = np.copy(image)
    
    for iteration in range(iterations):
        for y in range(height):
            for x in range(width):
                # Apply Arnold's Cat Map
                new_x = (x + y) % n
                new_y = (x + 2*y) % n
                transformed_image[new_y, new_x] = image[y, x]
        
        # Display the transformed image
        plt.subplot(2, 3, iteration+1)
        plt.imshow(transformed_image, cmap='gray')
        plt.title(f'Iteration {iteration+1}')
        plt.axis('off')
        
        # Save the transformed image
        cv2.imwrite(f'arnold_cat_map_iteration_{iteration+1}.png', transformed_image)
        
        # Update the image for the next iteration
        image = np.copy(transformed_image)
    
    plt.tight_layout()
    plt.show()
    
    return transformed_image

# Load the image
image_path = 'B3.jpeg'  # Replace with your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is square, if not, crop it to make it square
height, width = original_image.shape
min_dim = min(height, width)
original_image = original_image[:min_dim, :min_dim]

# Apply Arnold's Cat Map
iterations = 5  # You can adjust this number
final_transformed_image = arnold_cat_map(original_image, iterations)

# Display the original and final transformed image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(final_transformed_image, cmap='gray')
plt.title(f'Final Transformed Image (After {iterations} iterations)')
plt.axis('off')

plt.tight_layout()
plt.show()