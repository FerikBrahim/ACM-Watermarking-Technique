import cv2
import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from scipy.ndimage.interpolation import affine_transform
from skimage.util import random_noise
from PIL import Image, ImageFilter
import os

# Load the watermarked image
image_path = 'watermarked_BACT-03.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create a directory to save the attacked images
output_dir = 'attacked_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. JPEG Compression
def apply_jpeg_compression(image, quality=30):
    output_path = os.path.join(output_dir, 'jpeg_compressed.jpg')
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

# 2. Cropping
def apply_cropping(image, crop_percent=0.2):
    h, w = image.shape
    crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
    cropped_image = image[crop_h:h-crop_h, crop_w:w-crop_w]
    output_path = os.path.join(output_dir, 'cropped.jpg')
    cv2.imwrite(output_path, cropped_image)

# 3. Rescaling
def apply_rescaling(image, scale=0.5):
    rescaled_image = cv2.resize(image, None, fx=scale, fy=scale)
    output_path = os.path.join(output_dir, 'rescaled.jpg')
    cv2.imwrite(output_path, rescaled_image)

# 4. Rotation
def apply_rotation(image, angle=30):
    rotated_image = rotate(image, angle, reshape=False)
    output_path = os.path.join(output_dir, 'rotated.jpg')
    cv2.imwrite(output_path, rotated_image)

# 5. Additive Noise
def apply_additive_noise(image, mode='gaussian'):
    noisy_image = random_noise(image, mode=mode)
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    output_path = os.path.join(output_dir, 'noisy.jpg')
    cv2.imwrite(output_path, noisy_image)

# 6. Affine Transformation
def apply_affine_transformation(image):
    h, w = image.shape
    matrix = np.array([[1, 0.2, -30], [0.2, 1, -30], [0, 0, 1]])
    affine_image = affine_transform(image, matrix)
    output_path = os.path.join(output_dir, 'affine_transformed.jpg')
    cv2.imwrite(output_path, affine_image)

# 7. Random Bending Attack
def apply_bending_attack(image):
    bending_image = np.copy(image)
    bending_image = np.float32(bending_image)
    for i in range(image.shape[0]):
        bending_image[i] = np.roll(bending_image[i], int(np.sin(np.pi * i / 15) * 10))
    output_path = os.path.join(output_dir, 'bending_attack.jpg')
    cv2.imwrite(output_path, bending_image)

# 8. Blurring
def apply_blurring(image, radius=5):
    pil_image = Image.fromarray(image)
    blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius))
    blurred_image = np.array(blurred_image)
    output_path = os.path.join(output_dir, 'blurred.jpg')
    cv2.imwrite(output_path, blurred_image)

# 9. JPEG2000 Compression
def apply_jpeg2000_compression(image):
    output_path = os.path.join(output_dir, 'jpeg2000_compressed.jp2')
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 500])

# Apply all attacks and save the results
apply_jpeg_compression(image)
apply_cropping(image)
apply_rescaling(image)
apply_rotation(image)
apply_additive_noise(image)
apply_affine_transformation(image)
apply_bending_attack(image)
apply_blurring(image)
apply_jpeg2000_compression(image)

print("All attacks applied and images saved in the 'attacked_images' directory.")
