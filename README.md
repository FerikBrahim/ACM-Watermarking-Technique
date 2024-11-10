
## Project Overview

This repository contains the implementation of an adaptive ACM watermarking technique for secure medical image processing. The method embeds hidden information within images using a robust, palmprint-based approach that enhances patient security and data integrity. The project leverages:

- **Feature Extraction**: Combining Local Binary Patterns (LBP) and Histograms of Oriented Gradients (HOG) for detailed image analysis.
- **Image Decomposition**: Utilizing Discrete Wavelet Transform (DWT) for multi-level decomposition.
- **Chaotic Mapping**: Employing Arnold Cat Map (ACM) for enhanced security through non-linear transformations.
- **Watermark Embedding**: Applying Singular Value Decomposition (SVD) for stability and resilience.

Key outcomes include a Peak Signal-to-Noise Ratio (PSNR) of 63.52 and a Structural Similarity Index (SSIM) of 1.00, demonstrating excellent image quality retention. The method achieves an Equal Error Rate (EER) of 0.035, validating its robustness and reliability against image processing attacks, making it suitable for secure medical applications that prioritize authenticity and privacy.

