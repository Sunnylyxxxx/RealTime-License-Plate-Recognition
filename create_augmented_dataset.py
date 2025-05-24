import os
import cv2
import numpy as np
from PIL import Image
import shutil
import glob

def create_directory_structure():
    """Create directories for augmented datasets."""
    base_dirs = ['dataset_augmented_colorshift', 'dataset_augmented_rotation']
    for base_dir in base_dirs:
        os.makedirs(os.path.join(base_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'labels', 'train'), exist_ok=True)

def apply_color_shift(image):
    """Apply random color shifts to the image."""
    img = image.astype(np.float32)
    shift = np.random.uniform(-30, 30, 3)
    for i in range(3):
        img[:, :, i] = np.clip(img[:, :, i] + shift[i], 0, 255)
    return img.astype(np.uint8)

def apply_rotation(image, angle, label):
    """Apply rotation to image and update label coordinates."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    if label is not None:
        x_center, y_center = label[1] * width, label[2] * height
        w, h = label[3] * width, label[4] * height
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix)
        x_coords = rotated_points[:, 0, 0]
        y_coords = rotated_points[:, 0, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        new_x_center = (x_min + x_max) / (2 * width)
        new_y_center = (y_min + y_max) / (2 * height)
        new_w = (x_max - x_min) / width
        new_h = (y_max - y_min) / height
        new_x_center = np.clip(new_x_center, 0, 1)
        new_y_center = np.clip(new_y_center, 0, 1)
        new_w = np.clip(new_w, 0, 1)
        new_h = np.clip(new_h, 0, 1)
        return rotated_image, [label[0], new_x_center, new_y_center, new_w, new_h]
    return rotated_image, None

def process_dataset():
    """Process the dataset to create augmented versions."""
    create_directory_structure()
    image_files = glob.glob(os.path.join('dataset', 'images', 'train', '*.png'))
    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
        label_path = os.path.join('dataset', 'labels', 'train', 
                                 os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        label = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_line = f.readline().strip()
                if label_line:
                    label = [float(x) for x in label_line.split()]
        for i in range(3):
            shifted_image = apply_color_shift(image)
            new_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_colorshift_{i}.png"
            cv2.imwrite(os.path.join('dataset_augmented_colorshift', 'images', 'train', new_img_name), 
                       shifted_image)
            if label is not None:
                new_label_path = os.path.join('dataset_augmented_colorshift', 'labels', 'train',
                                            os.path.splitext(new_img_name)[0] + '.txt')
                with open(new_label_path, 'w') as f:
                    f.write(' '.join(map(str, label)))
        for angle in [15, -15, 30, -30]:
            rotated_image, rotated_label = apply_rotation(image, angle, label)
            new_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_rot_{angle}.png"
            cv2.imwrite(os.path.join('dataset_augmented_rotation', 'images', 'train', new_img_name),
                       rotated_image)
            if rotated_label is not None:
                new_label_path = os.path.join('dataset_augmented_rotation', 'labels', 'train',
                                            os.path.splitext(new_img_name)[0] + '.txt')
                with open(new_label_path, 'w') as f:
                    f.write(' '.join(map(str, rotated_label)))
if __name__ == "__main__":
    print("Creating augmented datasets...")
    process_dataset()
    print("Done! Augmented datasets created in:")
    print("- dataset_augmented_colorshift")
    print("- dataset_augmented_rotation") 