import cv2
import numpy as np

def psnr(img1, img2):
  """
  Calculates the PSNR between two images.

  Args:
      img1: The first image as a NumPy array.
      img2: The second image as a NumPy array.

  Returns:
      The PSNR value in decibels (dB).
  """
  mse = np.mean((img1 / 255 - img2 / 255) ** 2)
  if mse == 0:  # Avoid division by zero
    return float('inf')
  max_pixel = 1.0
  psnr = 10 * np.log10(max_pixel**2 / mse)
  return psnr

# Read the images
img1 = cv2.imread("B3.jpeg")
img2 = cv2.imread("watermarked_xray.jpeg")

# Check if images are loaded successfully
if img1 is None or img2 is None:
  print("Error: Could not read images!")
  exit(1)

# Convert images to grayscale if they are color
if len(img1.shape) == 3:
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate PSNR
psnr_value = psnr(img1, img2)

# Print the PSNR value
print(f"PSNR between images: {psnr_value:.2f} dB")
