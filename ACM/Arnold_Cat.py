import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Arnold Cat Map
def arnold_cat_map(image_array, iterations=1):
    h, w = image_array.shape
    result = image_array.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for i in range(h):
            for j in range(w):
                new_i = (i + j) % h
                new_j = (i + 2 * j) % w
                temp[new_i, new_j] = result[i, j]
        result = temp
    return result

def inverse_arnold_cat_map(image_array, iterations=1):
    h, w = image_array.shape
    result = image_array.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for i in range(h):
            for j in range(w):
                new_i = (2 * i - j) % h
                new_j = (-i + j) % w
                temp[new_i, new_j] = result[i, j]
        result = temp
    return result

# Tribonacci Cat Map
def tribonacci_cat_map(image_array, iterations=1):
    h, w = image_array.shape
    result = image_array.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for i in range(h):
            for j in range(w):
                new_i = (i + j) % h
                new_j = (i + 2 * j) % w
                temp[new_i, new_j] = result[i, j]
        result = temp
    return result

def inverse_tribonacci_cat_map(image_array, iterations=1):
    h, w = image_array.shape
    result = image_array.copy()
    for _ in range(iterations):
        temp = np.zeros_like(result)
        for i in range(h):
            for j in range(w):
                new_i = (2 * i - j) % h
                new_j = (-i + j) % w
                temp[new_i, new_j] = result[i, j]
        result = temp
    return result

# Add watermark to an image array
def add_watermark(image_array, watermark):
    h, w = image_array.shape
    wh, ww = watermark.shape
    x_offset = (h - wh) // 2
    y_offset = (w - ww) // 2
    watermarked_image = image_array.copy()
    watermarked_image[x_offset:x_offset+wh, y_offset:y_offset+ww] = watermark
    return watermarked_image

# Load image and convert to grayscale
image = Image.open('lighthouse.bmp').convert('L')
image_array = np.array(image)

# Create a simple watermark
watermark = np.zeros((50, 50), dtype=np.uint8)
watermark[10:40, 10:40] = 255  # A white square as a watermark

# Apply Arnold Cat Map, add watermark, and then inverse
iterations = 5  # Set the number of iterations
arnold_transformed = arnold_cat_map(image_array, iterations=iterations)
arnold_transformed_watermarked = add_watermark(arnold_transformed, watermark)
inverse_arnold_result = inverse_arnold_cat_map(arnold_transformed_watermarked, iterations=iterations)

# Apply Tribonacci Cat Map, add watermark, and then inverse
tribonacci_transformed = tribonacci_cat_map(image_array, iterations=iterations)
tribonacci_transformed_watermarked = add_watermark(tribonacci_transformed, watermark)
inverse_tribonacci_result = inverse_tribonacci_cat_map(tribonacci_transformed_watermarked, iterations=iterations)

# Calculate PSNR and SSIM for the final reconstructions
psnr_arnold = peak_signal_noise_ratio(image_array, inverse_arnold_result)
ssim_arnold = structural_similarity(image_array, inverse_arnold_result)

psnr_tribonacci = peak_signal_noise_ratio(image_array, inverse_tribonacci_result)
ssim_tribonacci = structural_similarity(image_array, inverse_tribonacci_result)

# Display the results in a table
import pandas as pd

data = {
    'Metric': ['PSNR', 'SSIM'],
    'Arnold Cat Map': [psnr_arnold, ssim_arnold],
    'Tribonacci Cat Map': [psnr_tribonacci, ssim_tribonacci]
}

df = pd.DataFrame(data)
print(df)

# Display the original, watermarked, and reconstructed images
fig, ax = plt.subplots(3, 3, figsize=(18, 18))
ax[0, 0].imshow(image_array, cmap='gray')
ax[0, 0].set_title('Original Image')
ax[0, 1].imshow(arnold_transformed, cmap='gray')
ax[0, 1].set_title('Arnold Transformed Image')
ax[0, 2].imshow(inverse_arnold_result, cmap='gray')
ax[0, 2].set_title('Arnold Reconstructed Image')

ax[1, 0].imshow(watermark, cmap='gray')
ax[1, 0].set_title('Watermark')
ax[1, 1].imshow(arnold_transformed_watermarked, cmap='gray')
ax[1, 1].set_title('Arnold Watermarked Image')
ax[1, 2].imshow(inverse_arnold_result, cmap='gray')
ax[1, 2].set_title('Arnold Reconstructed Image with Watermark')

ax[2, 0].imshow(image_array, cmap='gray')
ax[2, 0].set_title('Original Image')
ax[2, 1].imshow(tribonacci_transformed, cmap='gray')
ax[2, 1].set_title('Tribonacci Transformed Image')
ax[2, 2].imshow(inverse_tribonacci_result, cmap='gray')
ax[2, 2].set_title('Tribonacci Reconstructed Image with Watermark')

plt.show()
