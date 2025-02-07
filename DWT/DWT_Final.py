import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image

# Load your image
image_path = 'B3.jpeg'  # Replace with your image file path
img = Image.open(image_path).convert('L')  # Convert image to grayscale
data = np.array(img)

# Apply one-level discrete wavelet decomposition
coeffs = pywt.dwt2(data, 'haar')
cA, (cH, cV, cD) = coeffs

# Normalize and scale the coefficients for better visualization
def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

# Normalize the coefficients
cA_norm = normalize(cA)
cH_norm = normalize(cH)
cV_norm = normalize(cV)
cD_norm = normalize(cD)

# Save each component as a separate image
plt.imsave('LL_level1.jpg', cA_norm, cmap='gray')
plt.imsave('HL_level1.jpg', cH_norm, cmap='gray')
plt.imsave('LH_level1.jpg', cV_norm, cmap='gray')
plt.imsave('HH_level1.jpg', cD_norm, cmap='gray')
