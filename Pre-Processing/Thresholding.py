import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte


# Load image
img = cv2.imread('61I9XdN6OFL.jpg', 0)  # Load in grayscale

# Display the original image
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Otsu's Binarization
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(6,6))
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu Thresholding')
plt.axis('off')
plt.show()

# Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(6,6))
plt.imshow(adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')
plt.show()

# Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(6,6))
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')
plt.show()

# Simple Global Thresholding
_, simple_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6,6))
plt.imshow(simple_thresh, cmap='gray')
plt.title('Simple Global Thresholding')
plt.axis('off')
plt.show()

# Compare all methods side by side
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Otsu's Thresholding
plt.subplot(2, 2, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu\'s Thresholding')
plt.axis('off')

# Adaptive Mean Thresholding
plt.subplot(2, 2, 3)
plt.imshow(adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')

# Adaptive Gaussian Thresholding
plt.subplot(2, 2, 4)
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')

plt.tight_layout()
plt.show()

# Function to calculate edge sharpness using Sobel operator
def edge_sharpness(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    sobel_magnitude = np.hypot(sobelx, sobely)  # Edge magnitude
    return np.mean(sobel_magnitude)

# Function to calculate contrast
def contrast(image):
    return image.max() - image.min()

# Function to calculate entropy
def image_entropy(image):
    entropy_value = entropy(image, disk(5))  # Disk size is arbitrary
    return np.mean(entropy_value)

# Function to evaluate an image
def evaluate_image(image, title=""):
    sharpness = edge_sharpness(image)
    img_contrast = contrast(image)
    img_entropy = image_entropy(img_as_ubyte(image))  # Convert to unsigned byte
    
    print(f"Evaluation Metrics for {title}:")
    print(f"Sharpness: {sharpness:.2f}")
    print(f"Contrast: {img_contrast:.2f}")
    print(f"Entropy: {img_entropy:.2f}\n")
    
    return sharpness, img_contrast, img_entropy

# Otsu's Binarization
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
evaluate_image(otsu_thresh, title="Otsu Thresholding")

# Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
evaluate_image(adaptive_mean, title="Adaptive Mean Thresholding")

# Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
evaluate_image(adaptive_gaussian, title="Adaptive Gaussian Thresholding")

# Simple Global Thresholding
_, simple_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
evaluate_image(simple_thresh, title="Simple Global Thresholding")

# Compare all methods side by side
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu\'s Thresholding')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')

plt.tight_layout()
plt.show()

def evaluate_threshold(output):
    """Evaluate the thresholding result based on the number of white pixels."""
    return np.sum(output == 255)  # Count white pixels

def save_best_output(best_output, method_name):
    """Save the best thresholded image to a file."""
    cv2.imwrite(f'best_threshold_output_{method_name}.png', best_output)


# Display the original image
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Method 1: Otsu's Binarization
_, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Method 2: Adaptive Mean Thresholding
adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)

# Method 3: Adaptive Gaussian Thresholding
adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)

# Method 4: Simple Global Thresholding
_, simple_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Evaluate the results
evaluations = {
    'Otsu': evaluate_threshold(otsu_thresh),
    'Adaptive Mean': evaluate_threshold(adaptive_mean),
    'Adaptive Gaussian': evaluate_threshold(adaptive_gaussian),
    'Simple': evaluate_threshold(simple_thresh)
}

# Find the best method
best_method = max(evaluations, key=evaluations.get)
best_output = None

if best_method == 'Otsu':
    best_output = otsu_thresh
elif best_method == 'Adaptive Mean':
    best_output = adaptive_mean
elif best_method == 'Adaptive Gaussian':
    best_output = adaptive_gaussian
elif best_method == 'Simple':
    best_output = simple_thresh

# Save the best threshold output
save_best_output(best_output, best_method)

# Compare all methods side by side
plt.figure(figsize=(12, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Otsu's Thresholding
plt.subplot(2, 2, 2)
plt.imshow(otsu_thresh, cmap='gray')
plt.title('Otsu\'s Thresholding')
plt.axis('off')

# Adaptive Mean Thresholding
plt.subplot(2, 2, 3)
plt.imshow(adaptive_mean, cmap='gray')
plt.title('Adaptive Mean Thresholding')
plt.axis('off')

# Adaptive Gaussian Thresholding
plt.subplot(2, 2, 4)
plt.imshow(adaptive_gaussian, cmap='gray')
plt.title('Adaptive Gaussian Thresholding')
plt.axis('off')

plt.tight_layout()
plt.show()

# Print evaluation results
print("Thresholding Evaluations:")
for method, count in evaluations.items():
    print(f"{method}: {count} white pixels")

print(f"\nBest method: {best_method} with {evaluations[best_method]} white pixels.")
