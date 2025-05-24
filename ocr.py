import easyocr
import cv2
import numpy as np
import os
import glob
from pathlib import Path
import pandas as pd
import re
from dpi_check import is_ocr_ready
import io
from google.cloud import vision

STATE_NAMES = [
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO", 
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO", 
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA", 
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", 
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA", 
    "NEWHAMPSHIRE", "NEWJERSEY", "NEWMEXICO", "NEWYORK", "NORTHCAROLINA", 
    "NORTHDAKOTA", "OHIO", "OKLAHOMA", "OREGON", "PENNSYLVANIA", 
    "RHODEISLAND", "SOUTHCAROLINA", "SOUTHDAKOTA", "TENNESSEE", "TEXAS", 
    "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON", "WESTVIRGINIA", 
    "WISCONSIN", "WYOMING", "DISTRICTOFCOLUMBIA"
]

def filter_state_names(text):
    """Remove state names and common license plate text from OCR results"""
    text = text.upper()
    text = text.replace(" ", "")
    if text in STATE_NAMES:
        return ""
    for state in STATE_NAMES:
        if text.startswith(state):
            text = text[len(state):]
        elif text.endswith(state):
            text = text[:-len(state)]
    common_texts = ["LICENSEPLATENUMBER", "LICENSEPLATE", "PLATE", "LICENSE"]
    for common in common_texts:
        text = text.replace(common, "")
    
    return text

def read_confidence_and_dpi(img_path):
    """Read confidence score and DPI from the corresponding text file"""
    img_stem = Path(img_path).stem
    match = re.match(r'(.+)_plate_(\d+)', img_stem)
    if not match:
        return "N/A", 0
    base_name = match.group(1)
    plate_idx = int(match.group(2))
    conf_file = Path("cropped_plates") / f"{base_name}_confidence.txt"
    if not conf_file.exists():
        return "N/A", 0
    with open(conf_file, 'r') as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                idx = int(parts[0])
                conf = float(parts[1])
                dpi = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
                if idx == plate_idx:
                    return f"{conf:.4f}", dpi
            except (ValueError, IndexError):
                continue
    
    return "N/A", 0

def preprocess_for_ocr(image):
    """Enhance image for better OCR performance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, searchWindowSize=21, templateWindowSize=7)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    amount = 1.2
    sharpened = cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 19, 8)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cleaned

def recognize_with_google_vision(img_path, debug=True):
    """Recognize license plate text using Google Cloud Vision API"""
    try:
        client = vision.ImageAnnotatorClient()
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        if response.error.message:
            print(f"Google Vision API error: {response.error.message}")
            return "", 0.0
        texts = response.text_annotations
        if not texts:
            return "", 0.0
        full_text = texts[0].description if texts else ""
        plate_text = re.sub(r'[^A-Z0-9]', '', full_text.upper())
        plate_text = filter_state_names(plate_text)
        google_confidence = 0.85
        if debug:
            img = cv2.imread(img_path)
            if img is not None:
                debug_img = img.copy()
                if texts and len(texts) > 0:
                    vertices = [(vertex.x, vertex.y) 
                               for vertex in texts[0].bounding_poly.vertices]
                    pts = np.array(vertices, dtype=np.int32)
                    cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
                    cv2.putText(debug_img, f"Google: {plate_text}", 
                               (pts[0][0], pts[0][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                debug_dir = "google_ocr_debug"
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"{Path(img_path).stem}_debug.jpg")
                cv2.imwrite(debug_path, debug_img)
        return plate_text, google_confidence
        
    except Exception as e:
        print(f"Google Vision API error: {e}")
        return "", 0.0

def recognize_plate(reader, img_path, debug=True):
    """Recognize license plate text using EasyOCR"""
    try:
        original = cv2.imread(img_path)
        if original is None:
            print(f"Error: Could not read image {img_path}")
            return "", "N/A", 0, False
        filename = os.path.basename(img_path)
        dpi_match = re.match(r'^(\d+(?:\.\d+)?)dpi_', filename)
        if dpi_match:
            filename_dpi = float(dpi_match.group(1))
        else:
            _, filename_dpi = is_ocr_ready(img_path)
        image_suitable, _ = is_ocr_ready(img_path)
        preprocessed = preprocess_for_ocr(original)
        results = reader.readtext(img_path, 
                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                                 detail=1,
                                 paragraph=False,
                                 decoder='greedy',
                                 beamWidth=5,
                                 contrast_ths=0.1,
                                 adjust_contrast=0.5,
                                 text_threshold=0.6)
        
        # Parse the results
        plate_text = ""
        debug_img = original.copy() if debug else None
        confidence_sum = 0.0
        confidence_count = 0
        
        if results:
            for i, (bbox, text, prob) in enumerate(results):
                # Add to plate text
                plate_text += text
                confidence_sum += prob
                confidence_count += 1
                
                # Draw bounding boxes and text for debugging
                if debug:
                    # Convert points to integer
                    bbox = np.array(bbox, dtype=np.int32)
                    
                    # Draw the bounding box
                    cv2.polylines(debug_img, [bbox], True, (0, 255, 0), 2)
                    
                    # Add text and probability
                    cv2.putText(debug_img, f"{text} ({prob:.2f})", 
                                (bbox[0][0], bbox[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Clean the text
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        # Filter out state names
        plate_text = filter_state_names(plate_text)
        
        # Calculate average confidence
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        confidence_str = f"{avg_confidence:.4f}"
        
        # Get detection confidence from file
        detect_confidence, _ = read_confidence_and_dpi(img_path)
        
        # Save debug image if requested
        if debug and debug_img is not None:
            debug_dir = "easyocr_debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create two-panel image with original and preprocessed
            h, w = preprocessed.shape
            debug_img_resized = cv2.resize(debug_img, (w, h))
            combined = np.hstack((debug_img_resized, cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)))
            
            # Add DPI info to debug image
            dpi_text = f"DPI: {filename_dpi} ({('OCR-Ready' if image_suitable else 'Low-Res')})"
            cv2.putText(combined, dpi_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if image_suitable else (0, 0, 255), 2)
            cv2.putText(combined, f"OCR: {plate_text}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            debug_path = os.path.join(debug_dir, f"{Path(img_path).stem}_debug.jpg")
            cv2.imwrite(debug_path, combined)
        
        return plate_text, confidence_str, filename_dpi, image_suitable
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return "", "N/A", 0, False

def ensemble_recognition(reader, img_path, debug=True):
    """Combine results from EasyOCR and Google Cloud Vision for better accuracy"""
    image_suitable, image_dpi = is_ocr_ready(img_path)
    if image_dpi < 14:
        print(f"Skipping OCR for {img_path} - DPI too low ({image_dpi:.1f})")
        return "LOW_DPI", "N/A", image_dpi, False, "None"
    easyocr_text, confidence, dpi, ocr_ready = recognize_plate(reader, img_path, debug)
    conf_value = float(confidence) if confidence != "N/A" else 0.0
    if conf_value < 0.93:
        google_text, google_conf = recognize_with_google_vision(img_path, debug)
        candidates = [
            (easyocr_text, conf_value, "EasyOCR"),
            (google_text, google_conf, "GoogleVision")
        ]
        valid_candidates = [(text, conf, model) for text, conf, model in candidates if len(text) >= 4]
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            final_text, final_conf, model_used = valid_candidates[0]
        else:
            candidates.sort(key=lambda x: len(x[0]), reverse=True)
            final_text, final_conf, model_used = candidates[0]
        if model_used == "GoogleVision":
            confidence_str = "GoogleVision"
        else:
            confidence_str = str(final_conf)
    else:
        final_text = easyocr_text
        final_conf = conf_value
        model_used = "EasyOCR"
        confidence_str = str(final_conf)
    print(f"Final result: '{final_text}' (Confidence: {confidence_str}, Model: {model_used})")
    return final_text, confidence_str, dpi, ocr_ready, model_used

def process_all_plates(debug=True):
    """Process all license plates with EasyOCR and Google Cloud Vision as fallback"""
    reader = easyocr.Reader(['en'], gpu=True)
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
    output_dir = "ocr_results"
    os.makedirs(output_dir, exist_ok=True)
    plate_images = glob.glob("cropped_plates/*_plate_*.jpg")
    total_images = len(plate_images)
    print(f"Found {total_images} license plates to process")
    results = []
    ocr_ready_count = 0
    low_dpi_count = 0
    model_used_counts = {"EasyOCR": 0, "GoogleVision": 0, "None": 0}
    for i, img_path in enumerate(plate_images, 1):
        print(f"Processing plate {i}/{total_images}: {img_path}")
        plate_text, confidence, dpi, ocr_ready, model_used = ensemble_recognition(reader, img_path, debug=debug)
        if ocr_ready:
            ocr_ready_count += 1
        elif plate_text == "LOW_DPI":
            low_dpi_count += 1
        if model_used in model_used_counts:
            model_used_counts[model_used] += 1
        plate_name = Path(img_path).stem
        if ocr_ready:
            output_file = os.path.join(output_dir, f"{plate_name}_ocr.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"OCR Result: {plate_text}\n")
                f.write(f"Detection Confidence: {confidence}\n")
                f.write(f"Image DPI: {dpi}\n")
                f.write(f"OCR-Ready: {ocr_ready}\n")
                f.write(f"Model Used: {model_used}\n")
        result_data = {
            'plate_name': plate_name,
            'ocr_text': plate_text,
            'confidence': confidence,
            'dpi': dpi,
            'ocr_ready': ocr_ready,
            'model_used': model_used
        }
        if plate_text != "LOW_DPI":
            result_data['is_correct'] = False
        results.append(result_data)
        status = "✅" if ocr_ready else "❌"
        print(f"{status} Plate {plate_name}: '{plate_text}' (Conf: {confidence}, DPI: {dpi}, Model: {model_used})")
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "ocr_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved all OCR results to {csv_path}")
        print(f"Note: Added 'is_correct' column for OCR results that need manual verification")
    print(f"\nOCR Processing Complete:")
    print(f"Total plates processed: {total_images}")
    print(f"OCR-ready plates: {ocr_ready_count} ({ocr_ready_count/total_images*100:.1f}% of total)")
    print(f"Low DPI plates skipped: {low_dpi_count} ({low_dpi_count/total_images*100:.1f}% of total)")
    print("\nModel usage:")
    for model, count in model_used_counts.items():
        if count > 0:
            print(f"  {model}: {count} ({count/total_images*100:.1f}%)")
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    process_all_plates() 
