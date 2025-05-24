import os
import glob
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

output_dir = "yolo_plate_detection_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def download_base_model():
    """Download a pre-trained YOLO model for training."""
    try:
        model = YOLO('yolov8n.pt')
        print("Base model downloaded successfully")
        return model
    except Exception as e:
        print(f"Error downloading base model: {e}")
        return None

def train_model(model, data_yaml_path, epochs=50):
    """Train the YOLO model on license plate data."""
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=640,
            batch=16,
            name="yolo_license_plate_detector"
        )
        print(f"Training completed: {results}")
        best_model_path = os.path.join('runs', 'detect', 'yolo_license_plate_detector', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print(f"Best model saved at: {best_model_path}")
            best_model = YOLO(best_model_path)
            return best_model
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def create_data_yaml():
    """Create a YAML file for YOLO training with the dataset structure."""
    # Get absolute paths
    current_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(current_dir, "dataset")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    train_images_dir = os.path.join(images_dir, "train")
    train_labels_dir = os.path.join(labels_dir, "train")
    if not os.path.exists(train_images_dir) or not os.path.exists(train_labels_dir):
        print(f"Make sure you have: dataset/images/train and dataset/labels/train directories.")
        return None
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(train_images_dir, ext)))
    
    if not image_files:
        print(f"No images found in {train_images_dir}")
        print("Please add images before training.")
        return None
    
    print(f"Found {len(image_files)} images in {train_images_dir}")
    
    yaml_content = f"""
# License Plate Detection Dataset
path: {current_dir}  # dataset root dir (absolute path)
train: {os.path.join('dataset', 'images', 'train')}  # train images relative to path
val: {os.path.join('dataset', 'images', 'train')}  # val images relative to path (using train for now)

# Classes
names:
  0: license_plate
"""
    
    yaml_path = 'license_plate_data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created data YAML file: {yaml_path}")
    print(f"Using root path: {current_dir}")
    return yaml_path

def detect_with_trained_model(model):
    """Detect license plates using trained YOLO model."""
    dataset_dir = "dataset"
    images_dir = os.path.join(dataset_dir, "images")
    train_images_dir = os.path.join(images_dir, "train")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(train_images_dir, ext)))
    total_images = len(image_files)
    detected_plates = []

    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{total_images}: {image_path}")
        results = model(image_path, conf=0.25)
        result_img = cv2.imread(image_path)
        plate_found = False
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                plate_found = True
                detected_plates.append(image_path)
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_img, f"PLATE {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_detection.png")
        cv2.imwrite(output_path, result_img)
        print(f"Plate detected in {image_path}: {plate_found}")
    print(f"\nDetection complete! Found license plates in {len(detected_plates)} of {total_images} images")
    with open(os.path.join(output_dir, "detected_plates.txt"), "w") as f:
        for img_path in detected_plates:
            f.write(f"{img_path}\n")

def verify_dataset_structure():
    """Verify and create the proper dataset structure if it doesn't exist."""
    current_dir = os.path.abspath(os.getcwd())
    dataset_dir = os.path.join(current_dir, "dataset")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    train_images_dir = os.path.join(images_dir, "train")
    train_labels_dir = os.path.join(labels_dir, "train")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    print(f"Dataset structure verified and created if needed.")
    print(f"Image path: {train_images_dir}")
    print(f"Labels path: {train_labels_dir}")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(train_images_dir, ext)))
    annotation_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
    print(f"Found {len(image_files)} images and {len(annotation_files)} annotation files.")
    
    return len(image_files) > 0 and len(annotation_files) > 0

def check_annotations():
    """Check if annotations exist and are in the correct format."""
    dataset_dir = "dataset"
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    train_images_dir = os.path.join(images_dir, "train")
    train_labels_dir = os.path.join(labels_dir, "train")
    
    if not os.path.exists(train_labels_dir):
        print(f"No labels directory found at: {train_labels_dir}")
        print("Please make sure you've run the annotation tool.")
        return False
    
    annotation_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
    if len(annotation_files) == 0:
        print(f"No annotation files found in {train_labels_dir}.")
        print("Please make sure you have annotation files in YOLO format.")
        return False
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(train_images_dir, ext)))
    
    if len(image_files) == 0:
        print(f"No image files found in {train_images_dir}.")
        return False
    
    for ann_file in annotation_files[:3]:
        try:
            with open(ann_file, 'r') as f:
                line = f.readline().strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Annotation file {ann_file} doesn't seem to be in YOLO format.")
                    print("Each line should have: <class_id> <center_x> <center_y> <width> <height>")
                    print("You might need to convert your annotations with the annotation tool.")
        except Exception as e:
            print(f"Error reading annotation file {ann_file}: {e}")
    
    print(f"Found {len(annotation_files)} annotation files and {len(image_files)} images.")
    annotated_count = len(annotation_files)
    image_count = len(image_files)
    if annotated_count < image_count:
        print(f"Warning: Only {annotated_count} of {image_count} images have annotations.")
        print("You might want to annotate all images for better training results.")
    
    return True

def main():
    """Main function to train the YOLO license plate detector."""
    has_data = verify_dataset_structure()
    if not has_data:
        print("\nWARNING: No images or annotations found.")
        print("Please add images to dataset/images/train/ and run the annotation tool first.")
        print("To annotate images, run: python annotation_tool.py")
        return
    print("\nChecking annotations...")
    if not check_annotations():
        print("Please prepare your annotations before training.")
        return
    
    print("\nCreating dataset configuration...")
    data_yaml = create_data_yaml()
    if not data_yaml:
        print("Failed to create data configuration.")
        return
    model = download_base_model()
    if not model:
        return
    try:
        epochs = int(input("\nEnter number of training epochs (default: 50): ") or "50")
    except ValueError:
        print("Invalid input, using default 50 epochs.")
        epochs = 50
    print(f"\nStarting training for {epochs} epochs...")
    trained_model = train_model(model, data_yaml, epochs)
    if not trained_model:
        return
    print("\nTraining complete! Running detection with trained model...")
    detect_with_trained_model(trained_model)
    
if __name__ == "__main__":
    main() 