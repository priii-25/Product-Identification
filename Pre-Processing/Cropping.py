import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from ultralytics import YOLO

# Load YOLOv5 Model
def detect_with_yolo(image_path):
    model = YOLO('yolov5s.pt')  # You can also try 'yolov5x.pt' for better accuracy
    results = model(image_path)
    
    # Parse YOLOv5 results and crop detected text regions
    crops = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():  # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
    
    return crops

# Load Faster R-CNN Model
def detect_with_faster_rcnn(image):
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Convert the image to tensor
    image_tensor = F.to_tensor(image).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        prediction = model(image_tensor)

    # Parse Faster R-CNN results and crop detected text regions
    crops = []
    for box in prediction[0]['boxes'].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        crops.append(crop)
    
    return crops

# Load image and perform detection
def process_image(image_path, use_yolo=True):
    # Read image
    image = np.array(Image.open(image_path))

    # Choose the detection model
    if use_yolo:
        print("Using YOLO for text detection")
        crops = detect_with_yolo(image_path)
    else:
        print("Using Faster R-CNN for text detection")
        crops = detect_with_faster_rcnn(image)
    
    # Display or save the cropped regions
    for i, crop in enumerate(crops):
        cv2.imwrite(f'cropped_region_{i}.png', crop)  # Save cropped regions as images
        print(f'Cropped region {i} saved.')

    return crops

# Evaluate the model
def evaluate_model(image_path, use_yolo=True):
    # Perform detection and cropping
    crops = process_image(image_path, use_yolo)
    
    # Evaluation metrics (here we focus on number of crops for simplicity)
    print(f'Total regions detected: {len(crops)}')
    
    return crops

# Example Usage
if __name__ == "__main__":
    image_path = '/Users/kartik/Desktop/vs/Product-Identification/61I9XdN6OFL.jpg'  # Path to your image

    # Run YOLO
    yolo_crops = evaluate_model(image_path, use_yolo=True)
    
    # Run Faster R-CNN
    faster_rcnn_crops = evaluate_model(image_path, use_yolo=False)
