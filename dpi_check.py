from PIL import Image
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math
import re

def calculate_dpi_from_size(img_width, img_height, physical_width_inches=12, physical_height_inches=6):
    """
    Calculate DPI based on image dimensions and assumed physical size.
    
    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        physical_width_inches: Expected physical width in inches
        physical_height_inches: Expected physical height in inches
        
    Returns:
        tuple: (x_dpi, y_dpi)
    """
    x_dpi = img_width / physical_width_inches
    y_dpi = img_height / physical_height_inches
    return (round(x_dpi), round(y_dpi))

def get_license_plate_size_inches(img_width, img_height):
    """
    Estimate physical size of license plate based on aspect ratio.
    """
    aspect_ratio = img_width / img_height
    if 1.8 <= aspect_ratio <= 2.2:
        return (12, 6)
    elif 4.0 <= aspect_ratio <= 5.0:
        return (20.5, 4.5)
    elif aspect_ratio > 2.2:
        return (14, 6)
    else:
        return (12, 7)

def get_image_dpi(image_path):
    """
    Get the DPI information from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (x_dpi, y_dpi) based on metadata or estimation
    """
    try:
        with Image.open(image_path) as img:
            dpi = img.info.get('dpi')
            if dpi and dpi[0] > 0 and dpi[1] > 0:
                return dpi
            width, height = img.size
            try:
                x_mm = img.info.get('aspect_x', 0)
                y_mm = img.info.get('aspect_y', 0)
                if x_mm > 0 and y_mm > 0:
                    x_dpi = round(width / (x_mm / 25.4))
                    y_dpi = round(height / (y_mm / 25.4))
                    return (x_dpi, y_dpi)
            except Exception:
                pass
            if "plate" in str(image_path).lower():
                plate_width, plate_height = get_license_plate_size_inches(width, height)
                return calculate_dpi_from_size(width, height, plate_width, plate_height)
            else:
                if width > 1000 or height > 1000:
                    return calculate_dpi_from_size(width, height, 8.27, 11.69)
                else:
                    return calculate_dpi_from_size(width, height, width/100, height/100)
    except Exception as e:
        print(f"Error getting DPI from {image_path}: {e}")
        return (72, 72)

def estimate_dpi_from_image_size(image_path):
    """
    Estimate DPI based on image dimensions and assumed physical size.
    """
    try:
        with Image.open(image_path) as img:
            width_px, height_px = img.size
            if "plate" in str(image_path).lower():
                plate_width, plate_height = get_license_plate_size_inches(width_px, height_px)
                x_dpi = width_px / plate_width
                y_dpi = height_px / plate_height
            else:
                aspect_ratio = width_px / height_px
                if 0.7 <= aspect_ratio <= 0.75:
                    x_dpi = width_px / 8.27
                    y_dpi = height_px / 11.69
                else:
                    if max(width_px, height_px) > 1000:
                        physical_size = max(width_px, height_px) / 150
                    else:
                        physical_size = max(width_px, height_px) / 300
                    x_dpi = width_px / (physical_size * aspect_ratio)
                    y_dpi = height_px / physical_size
            return (round(x_dpi), round(y_dpi))
    except Exception as e:
        print(f"Error estimating DPI for {image_path}: {e}")
        return (72, 72)

def is_dpi_sufficient(dpi, threshold=300):
    """Check if the DPI is sufficient for OCR (300 DPI is recommended)"""
    return min(dpi[0], dpi[1]) >= threshold

def is_ocr_ready(image_path, min_dpi=14):
    """
    Check if an image's DPI is in the acceptable range for OCR.
    """
    metadata_dpi = get_image_dpi(image_path)
    estimated_dpi = estimate_dpi_from_image_size(image_path)
    min_metadata_dpi = min(metadata_dpi)
    min_estimated_dpi = min(estimated_dpi)
    effective_dpi = max(min_metadata_dpi, min_estimated_dpi)
    if effective_dpi < 10:
        with Image.open(image_path) as img:
            width, height = img.size
            small_object_dpi = width / 5
            effective_dpi = max(effective_dpi, small_object_dpi)
    is_ready = effective_dpi >= min_dpi
    
    return is_ready, effective_dpi

def analyze_image_dpi(image_path):
    """
    Analyze an image's DPI and print results.
    """
    dpi = get_image_dpi(image_path)
    estimated_dpi = estimate_dpi_from_image_size(image_path)
    min_metadata_dpi = min(dpi)
    min_estimated_dpi = min(estimated_dpi)
    effective_dpi = max(min_metadata_dpi, min_estimated_dpi)
    with Image.open(image_path) as img:
        dimensions = img.size
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Dimensions: {dimensions[0]}x{dimensions[1]} pixels")
    print(f"Metadata DPI: {dpi[0]}x{dpi[1]}")
    print(f"Estimated DPI: {estimated_dpi[0]}x{estimated_dpi[1]}")
    print(f"Effective DPI: {effective_dpi}")
    is_ready, _ = is_ocr_ready(image_path)
    if is_ready:
        print("✅ Image appears suitable for OCR based on DPI")
    else:
        print("⚠️ Image may not be suitable for OCR based on DPI")
    
    return dpi, estimated_dpi, dimensions, effective_dpi

def batch_analyze_dpi(directory, file_pattern="*.jpg"):
    """
    Analyze DPI for all images matching pattern in directory.
    """
    path = Path(directory)
    image_files = list(path.glob(file_pattern))
    
    if not image_files:
        print(f"No images matching {file_pattern} found in {directory}")
        return {}
    
    print(f"Analyzing {len(image_files)} images in {directory}...")
    metadata_dpis = []
    estimated_dpis = []
    effective_dpis = []
    filenames = []
    results = {}
    
    for img_path in image_files:
        dpi, estimated_dpi, _, effective_dpi = analyze_image_dpi(img_path)
        metadata_dpis.append(min(dpi))
        estimated_dpis.append(min(estimated_dpi))
        effective_dpis.append(effective_dpi)
        filename = os.path.basename(str(img_path))
        filenames.append(filename)
        results[filename] = {
            'metadata_dpi': dpi,
            'estimated_dpi': estimated_dpi,
            'effective_dpi': effective_dpi,
            'is_ocr_ready': is_ocr_ready(img_path)[0]
        }
    plot_dpi_results(filenames, metadata_dpis, estimated_dpis, effective_dpis)
    return results

def plot_dpi_results(filenames, metadata_dpis, estimated_dpis, effective_dpis=None):
    """Create a bar chart comparing DPI values"""
    plt.figure(figsize=(12, 6))
    x = np.arange(len(filenames))
    width = 0.3 if effective_dpis else 0.35
    plt.bar(x - width, metadata_dpis, width, label='Metadata DPI')
    plt.bar(x, estimated_dpis, width, label='Estimated DPI')
    if effective_dpis:
        plt.bar(x + width, effective_dpis, width, label='Effective DPI')
    plt.axhline(y=300, color='r', linestyle='--', label='300 DPI Recommendation')
    plt.axhline(y=200, color='orange', linestyle='--', label='200 DPI Minimum')
    plt.xlabel('Images')
    plt.ylabel('DPI (minimum of x and y)')
    plt.title('DPI Analysis for OCR')
    plt.xticks(x, filenames, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('dpi_analysis.png')
    print("\nDPI analysis chart saved as 'dpi_analysis.png'")
    plt.close()

def add_dpi_to_confidence_file(image_path, confidence_dir="cropped_plates"):
    """
    Add DPI information to the confidence file associated with an image.
    """
    try:
        img_path = Path(image_path)
        img_stem = img_path.stem
        match = re.match(r'(.+)_plate_(\d+)', img_stem)
        if not match:
            print(f"Could not parse image name: {img_stem}")
            return False
        base_name = match.group(1)
        plate_idx = int(match.group(2))
        conf_file = Path(confidence_dir) / f"{base_name}_confidence.txt"
        if not conf_file.exists():
            print(f"Confidence file not found: {conf_file}")
            return False
        _, _, _, effective_dpi = analyze_image_dpi(image_path)
        with open(conf_file, 'r') as f:
            lines = f.readlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.strip() and line.strip()[0].isdigit():
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        if idx == plate_idx:
                            if len(parts) >= 3:
                                parts[2] = str(effective_dpi)
                            else:
                                parts.append(str(effective_dpi))
                            new_lines.append(' '.join(parts) + '\n')
                            updated = True
                        else:
                            new_lines.append(line)
                    except (ValueError, IndexError):
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        if not updated:
            if not lines:
                new_lines.append("# Plate confidence scores\n")
                new_lines.append("# Index Confidence DPI\n")
            new_lines.append(f"{plate_idx} 0.0 {effective_dpi}\n")
        with open(conf_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Updated confidence file with DPI info: {conf_file}")
        return True
        
    except Exception as e:
        print(f"Error adding DPI to confidence file: {e}")
        return False

def update_all_confidence_files(image_dir, confidence_dir="cropped_plates"):
    """
    Update all confidence files with DPI information for the corresponding images.
    """
    image_path = Path(image_dir)
    plate_images = list(image_path.glob("*_plate_*.jpg"))
    print(f"Found {len(plate_images)} plate images to process")
    
    for img_path in plate_images:
        add_dpi_to_confidence_file(img_path, confidence_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isdir(path):
            batch_analyze_dpi(path)
        elif os.path.isfile(path):
            is_ready, dpi = is_ocr_ready(path)
            print(f"OCR readiness: {'Yes' if is_ready else 'No'} (DPI: {dpi})")
        else:
            print(f"Invalid path: {path}")
    else:
        print("Usage:")
        print("  python dpi_check.py <image_path> - Check DPI and OCR readiness of a single image")
        print("  python dpi_check.py <directory_path> - Check DPI of all images in directory")
