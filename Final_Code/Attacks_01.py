import cv2
import numpy as np
from scipy import ndimage
from skimage.util import random_noise

def apply_attacks(image_path):
    # Read the watermarked image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. JPEG Compression
    cv2.imwrite('attacked_jpeg.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # 2. Gaussian Noise
    noisy = random_noise(img, mode='gaussian', var=0.01)
    cv2.imwrite('attacked_gaussian_noise.png', (noisy*255).astype(np.uint8))

    # 3. Salt and Pepper Noise
    noisy = random_noise(img, mode='s&p', amount=0.01)
    cv2.imwrite('attacked_salt_pepper.png', (noisy*255).astype(np.uint8))

    # 4. Median Filtering
    median = cv2.medianBlur(img, 3)
    cv2.imwrite('attacked_median.png', median)

    # 5. Rotation
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite('attacked_rotation.png', rotated)

    # 6. Scaling
    scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    scaled = cv2.resize(scaled, (cols, rows), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('attacked_scaling.png', scaled)

    # 7. Cropping
    cropped = img[20:-20, 20:-20]
    cv2.imwrite('attacked_cropping.png', cropped)

    # 8. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('attacked_sharpening.png', sharpened)

    # 9. Gaussian Blurring
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('attacked_gaussian_blur.png', blurred)

    print("All attacks have been applied and saved.")

# Use the function
apply_attacks('watermarked_BACT-03.bmp')