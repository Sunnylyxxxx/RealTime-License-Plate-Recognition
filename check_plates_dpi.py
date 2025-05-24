from PIL import Image
import os
from pathlib import Path
import re
from dpi_check import is_ocr_ready, add_dpi_to_confidence_file, batch_analyze_dpi

def check_plates_dpi(input_dir="cropped_plates", confidence_dir=None, min_dpi=150):
    """
    Analyze DPI of all license plate images in a directory and update confidence files.
    
    Args:
        input_dir: Directory containing plate images
        confidence_dir: Directory containing confidence files (defaults to input_dir if None)
        min_dpi: Minimum DPI for considering a plate suitable for OCR
    
    Returns:
        tuple: (total_plates, ocr_ready_count, dpi_results)
    """
    if confidence_dir is None:
        confidence_dir = input_dir
    
    input_path = Path(input_dir)
    plate_images = list(input_path.glob("*_plate_*.jpg"))
    
    print(f"Found {len(plate_images)} plate images to analyze")
    total_plates = len(plate_images)
    ocr_ready_count = 0
    dpi_results = {}
    for img_path in plate_images:
        try:
            is_suitable, effective_dpi = is_ocr_ready(img_path)
            add_dpi_to_confidence_file(img_path, confidence_dir)
            filename = os.path.basename(str(img_path))
            dpi_results[filename] = {
                'dpi': effective_dpi,
                'ocr_ready': is_suitable
            }
            if is_suitable:
                ocr_ready_count += 1
                status = "✅"
            else:
                status = "❌"
            print(f"{status} {filename} - DPI: {effective_dpi}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print(f"\nDPI Analysis Summary:")
    print(f"Total plates analyzed: {total_plates}")
    print(f"OCR-ready plates: {ocr_ready_count} ({ocr_ready_count/total_plates*100:.1f}% of total)")
    print(f"Minimum DPI threshold: {min_dpi}")
    return total_plates, ocr_ready_count, dpi_results

def create_dpi_report(results, output_file="dpi_report.txt"):
    """
    Create a report file summarizing DPI analysis results.
    """
    sorted_results = sorted(results.items(), key=lambda x: x[1]['dpi'], reverse=True)
    
    with open(output_file, 'w') as f:
        f.write("# License Plate DPI Analysis Report\n\n")
        f.write("Filename\tDPI\tOCR Ready\n")
        f.write("-" * 50 + "\n")
        for filename, data in sorted_results:
            f.write(f"{filename}\t{data['dpi']}\t{'Yes' if data['ocr_ready'] else 'No'}\n")
    print(f"DPI report saved to {output_file}")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Check DPI of license plate images")
    parser.add_argument("--input", default="cropped_plates", help="Directory containing plate images")
    parser.add_argument("--confidence", default=None, help="Directory containing confidence files")
    parser.add_argument("--min-dpi", type=int, default=150, help="Minimum DPI threshold")
    args = parser.parse_args()
    
    total, ready, results = check_plates_dpi(
        input_dir=args.input,
        confidence_dir=args.confidence,
        min_dpi=args.min_dpi
    )
    create_dpi_report(results)
    if total > 0:
        print("\nGenerating DPI analysis chart...")
        batch_analyze_dpi(args.input) 