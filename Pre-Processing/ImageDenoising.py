import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import download_images

# Function to load image using OpenCV and convert it to grayscale
def load_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert to grayscale (optional, but improves OCR clarity)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Apply GaussianBlur to remove noise
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    # Apply Gaussian Blur
    denoised_image = cv2.GaussianBlur(image, kernel_size, 0)
    return denoised_image

# Apply MedianBlur to further reduce noise
def apply_median_blur(image, kernel_size=5):
    # Apply Median Blur
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image

# Sharpening the image using a kernel
def sharpen_image(image):
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the sharpening kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Display images using matplotlib for comparison
def display_images(original, denoised, sharpened):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(denoised, cmap='gray')
    plt.title('Denoised Image')

    plt.subplot(1, 3, 3)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened Image')

    plt.show()

# Main function to process image
def process_image(image_path):
    # Step 1: Load Image
    image = load_image(image_path)

    # Step 2: Denoise the image (GaussianBlur + MedianBlur)
    denoised_image = apply_gaussian_blur(image)
    denoised_image = apply_median_blur(denoised_image)

    # Step 3: Sharpen the image
    sharpened_image = sharpen_image(denoised_image)

    # Step 4: Display original, denoised, and sharpened images
    display_images(image, denoised_image, sharpened_image)

    # Save sharpened image for further use
    cv2.imwrite('sharpened_image.png', sharpened_image)

# Example usage
image_path = '61I9XdN6OFL.jpg'
process_image(image_path)