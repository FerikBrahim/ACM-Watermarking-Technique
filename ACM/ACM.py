import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def arnold_cat_map(image, num_iterations):
    n = image.shape[0]
    transformed_images = []
    
    for _ in range(num_iterations):
        new_image = np.zeros_like(image)
        for x in range(n):
            for y in range(n):
                new_x = (x + y) % n
                new_y = (x + 2 * y) % n
                new_image[new_x, new_y] = image[x, y]
        image = new_image
        transformed_images.append(image)
    
    return transformed_images

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, img in enumerate(images):
        output_path = os.path.join(output_folder, f'iteration_{i + 1}.png')
        img_pil = Image.fromarray(img)
        img_pil.save(output_path)

def plot_images(images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f'Iteration {i + 1}')
        axes[i].axis('off')
    
    plt.show()

# Load the image
input_image_path = 'LL_level1.jpg'  # Replace with your image path
image = Image.open(input_image_path).convert('RGB')
image_np = np.array(image)

# Apply Arnold Cat Map
num_iterations = 300
transformed_images = arnold_cat_map(image_np, num_iterations)

# Save and plot the images
output_folder = 'arnold_cat_map_iterations'
save_images(transformed_images, output_folder)
#plot_images(transformed_images)
