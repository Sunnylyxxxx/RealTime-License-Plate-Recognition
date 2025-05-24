import os
import cv2
import numpy as np
from pathlib import Path

def read_yolo_label(label_path, img_width, img_height):
    """
    Read YOLO format labels and convert to pixel coordinates
    """
    bounding_boxes = []
    
    if not os.path.exists(label_path):
        return bounding_boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center = float(data[1]) * img_width
        y_center = float(data[2]) * img_height
        width = float(data[3]) * img_width
        height = float(data[4]) * img_height
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        bounding_boxes.append([class_id, x1, y1, int(width), int(height)])
    
    return bounding_boxes

def crop_and_save_plates(image_path, label_path, output_dir):
    """
    Crop license plates from images based on bounding boxes and save them
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    img_height, img_width = image.shape[:2]
    bounding_boxes = read_yolo_label(label_path, img_width, img_height)
    image_name = Path(image_path).stem
    
    for i, bbox in enumerate(bounding_boxes):
        class_id, x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        plate_img = image[y:y+h, x:x+w]
        
        if plate_img.size == 0:
            print(f"Skipping empty crop from {image_path}")
            continue
        output_path = os.path.join(output_dir, f"{image_name}_plate_{i}.jpg")
        cv2.imwrite(output_path, plate_img)
        print(f"Saved: {output_path}")

def main():
    prediction_dir = "runs\detect\predict14\labels"
    image_dir = "runs\detect\predict14"
    output_dir = "cropped_plates4"
    predictions = os.listdir(prediction_dir)
    for pred_file in predictions:
        if not pred_file.endswith('.txt'):
            continue
        image_name = Path(pred_file).stem
        image_path = os.path.join(image_dir, f"{image_name}.png")
        if not os.path.exists(image_path):
            for ext in ['.jpg', '.jpeg']:
                alt_path = os.path.join(image_dir, f"{image_name}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
            
            if not os.path.exists(image_path):
                print(f"Corresponding image not found for {pred_file}")
                continue
        
        label_path = os.path.join(prediction_dir, pred_file)
        crop_and_save_plates(image_path, label_path, output_dir)
    
    print(f"Processing complete. Cropped plates saved to {output_dir}/")

if __name__ == "__main__":
    main() 