import cv2
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define entity-specific unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Define allowed units including synonyms
allowed_units = {
    'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton',
    'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd',
    'kilovolt', 'kV', 'millivolt', 'mV', 'volt', 'V',
    'kilowatt', 'kW', 'watt', 'W',
    'centilitre', 'cl', 'cubic foot', 'cu ft', 'cubic inch', 'cu in', 'cup', 'decilitre', 'dl', 'fluid ounce', 'fl oz', 'gallon', 'imperial gallon', 'imp gal', 'litre', 'l', 'microlitre', 'µl', 'millilitre', 'ml', 'pint', 'pt', 'quart', 'qt'
}

def preprocess_image(image_path):
    """Preprocess the image to improve OCR accuracy"""
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise the image
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Apply thresholding (Otsu's Binarization)
    _, thresholded = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return thresholded

def perform_ocr(image_path):
    """Perform OCR on the preprocessed image"""
    preprocessed_image = preprocess_image(image_path)
    
    custom_config = r'--oem 3 --psm 6' 
    text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    
    print("OCR Output:", text) 
    return text


def normalize_units(unit):
    """Normalize units to match allowed units"""
    unit_map = {
        'cm': 'centimetre',
        'kg': 'kilogram',
        'g': 'gram',
        'ml': 'millilitre',
        'l': 'litre',
        'oz': 'ounce',
        'ft': 'foot',
        'in': 'inch',
        'm': 'metre',
        'yd': 'yard',
    }
    return unit_map.get(unit.lower(), unit.lower())


def extract_values(ocr_text, entity_type):
    """Extract valid numeric values with units from OCR text based on entity type"""
    allowed_units_for_entity = entity_unit_map.get(entity_type, set())
    pattern = r'\d+(\.\d+)?\s?(' + '|'.join(re.escape(unit) for unit in allowed_units) + r')\b'
    matches = re.findall(pattern, ocr_text, re.IGNORECASE)
    
    print("Regex Pattern:", pattern)  
    print("OCR Text:", ocr_text) 
    
    if not matches:
        print("No valid values found in OCR output.")
        return []
    
    valid_values = [f'{num} {normalize_units(unit)}' for num, unit in matches]
    return valid_values


def ocr_pipeline(image_path, entity_type):
    """Complete OCR pipeline: preprocessing, OCR, and post-processing based on entity type"""
    text = perform_ocr(image_path)
    extracted_values = extract_values(text, entity_type)
    
    if extracted_values:
        print("Extracted Values:", extracted_values)
    else:
        print("No valid values found.")
    
    return extracted_values

image_path = r'C:\Users\VICTUS\Product-Identification\Pre-Processing\best_threshold_output_Simple.png'
entity_type = 'item_weight' 
extracted_values = ocr_pipeline(image_path, entity_type)
