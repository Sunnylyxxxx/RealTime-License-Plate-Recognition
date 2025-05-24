import os
import cv2
import glob
import numpy as np
import shutil

class LicensePlateAnnotator:
    def __init__(self):
        self.dataset_dir = "dataset"
        self.images_dir = os.path.join(self.dataset_dir, "images")
        self.labels_dir = os.path.join(self.dataset_dir, "labels")
        self.train_images_dir = os.path.join(self.images_dir, "test")
        self.train_labels_dir = os.path.join(self.labels_dir, "test")
        self.current_image = None
        self.current_filename = None
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.img_height = 0
        self.img_width = 0
        self.boxes = []
        self.create_directory_structure()
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.train_images_dir, ext)))
        if not self.image_files:
            print(f"No image files found in {self.train_images_dir}.")
            print(f"Please add images to {self.train_images_dir} directory.")
            exit(1)
        self.current_image_index = 0
    
    def create_directory_structure(self):
        """Create the proper directory structure for YOLO dataset."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.train_images_dir, exist_ok=True)
        os.makedirs(self.train_labels_dir, exist_ok=True)
        val_images_dir = os.path.join(self.images_dir, "val")
        val_labels_dir = os.path.join(self.labels_dir, "val")
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        old_images_dir = "images"
        old_annotations_dir = "annotations"
        if os.path.exists(old_images_dir) and os.listdir(old_images_dir) and not self.train_images_dir:
            print(f"Found images in the old structure. Migrating to new directory structure...")
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_file in glob.glob(os.path.join(old_images_dir, ext)):
                    shutil.copy2(img_file, self.train_images_dir)
                    print(f"Copied {img_file} to {self.train_images_dir}")
            if os.path.exists(old_annotations_dir):
                for ann_file in glob.glob(os.path.join(old_annotations_dir, "*.txt")):
                    shutil.copy2(ann_file, self.train_labels_dir)
                    print(f"Copied {ann_file} to {self.train_labels_dir}")
        
    def load_existing_annotations(self, image_file):
        """Load existing annotations for the current image if they exist."""
        self.boxes = []
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(self.train_labels_dir, f"{base_name}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * self.img_width
                        y_center = float(parts[2]) * self.img_height
                        width = float(parts[3]) * self.img_width
                        height = float(parts[4]) * self.img_height
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        self.boxes.append(((x1, y1), (x2, y2)))
            print(f"Loaded {len(self.boxes)} existing annotations from {label_file}")
        
    def save_annotations(self):
        """Save annotations for the current image."""
        if self.current_filename is None:
            return
        base_name = os.path.splitext(os.path.basename(self.current_filename))[0]
        label_file = os.path.join(self.train_labels_dir, f"{base_name}.txt")
        with open(label_file, 'w') as f:
            for box in self.boxes:
                x1, y1 = box[0]
                x2, y2 = box[1]
                x1 = max(0, min(x1, self.img_width - 1))
                y1 = max(0, min(y1, self.img_height - 1))
                x2 = max(0, min(x2, self.img_width - 1))
                y2 = max(0, min(y2, self.img_height - 1))
                x_center = (x1 + x2) / 2 / self.img_width
                y_center = (y1 + y2) / 2 / self.img_height
                width = abs(x2 - x1) / self.img_width
                height = abs(y2 - y1) / self.img_height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"Saved {len(self.boxes)} annotations to {label_file}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        img_copy = self.current_image.copy()
        for box in self.boxes:
            cv2.rectangle(img_copy, box[0], box[1], (0, 255, 0), 2)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point != self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                self.boxes.append(((x1, y1), (x2, y2)))
                for box in self.boxes:
                    cv2.rectangle(img_copy, box[0], box[1], (0, 255, 0), 2)
        cv2.imshow("License Plate Annotator", img_copy)
        
    def start(self):
        """Start the annotation process."""
        cv2.namedWindow("License Plate Annotator")
        cv2.setMouseCallback("License Plate Annotator", self.mouse_callback)
        print("\nLicense Plate Annotation Tool")
        print("Controls:")
        print("- Left click and drag to create a bounding box")
        print("- 'd' to delete the last box")
        print("- 'c' to clear all boxes")
        print("- 's' to save annotations")
        print("- 'n' to go to next image")
        print("- 'p' to go to previous image")
        print("- 'q' to quit\n")
        
        self.navigate_to_image(0)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('d'):
                if self.boxes:
                    self.boxes.pop()
                    img_copy = self.current_image.copy()
                    for box in self.boxes:
                        cv2.rectangle(img_copy, box[0], box[1], (0, 255, 0), 2)
                    cv2.imshow("License Plate Annotator", img_copy)
            elif key == ord('c'):
                self.boxes = []
                cv2.imshow("License Plate Annotator", self.current_image.copy())
            elif key == ord('n'):
                self.save_annotations()
                self.navigate_to_image(self.current_image_index + 1)
            elif key == ord('p'):
                self.save_annotations()
                self.navigate_to_image(self.current_image_index - 1)
        cv2.destroyAllWindows()

    def navigate_to_image(self, index):
        """Navigate to a specific image."""
        if index < 0:
            index = 0
        elif index >= len(self.image_files):
            index = len(self.image_files) - 1
        self.current_image_index = index
        self.current_filename = self.image_files[index]
        self.current_image = cv2.imread(self.current_filename)
        self.img_height, self.img_width = self.current_image.shape[:2]
        self.load_existing_annotations(self.current_filename)
        img_copy = self.current_image.copy()
        for box in self.boxes:
            cv2.rectangle(img_copy, box[0], box[1], (0, 255, 0), 2)
        cv2.imshow("License Plate Annotator", img_copy)
        print(f"\nImage {self.current_image_index + 1}/{len(self.image_files)}: {self.current_filename}")
if __name__ == "__main__":
    annotator = LicensePlateAnnotator()
    annotator.start() 