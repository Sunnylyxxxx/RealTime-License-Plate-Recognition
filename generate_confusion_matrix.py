import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
from pathlib import Path

def calculate(box1, box2):
    """Calculate intersection over union between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    box1_x1 = x1 - w1/2
    box1_y1 = y1 - h1/2
    box1_x2 = x1 + w1/2
    box1_y2 = y1 + h1/2
    box2_x1 = x2 - w2/2
    box2_y1 = y2 - h2/2
    box2_x2 = x2 + w2/2
    box2_y2 = y2 + h2/2
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    if xi2 < xi1 or yi2 < yi1:
        return 0.0
    intersection = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection
    return intersection / union

def read_yolo_label(label_path):
    """Read YOLO format label file."""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

def evaluate_predictions(pred_dir, gt_dir, threshold=0.5):
    """Evaluate predictions against ground truth."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pred_files = list(pred_dir.glob('*.txt'))
    gt_files = list(gt_dir.glob('*.txt'))
    true_positives = []
    scores = []
    num_gt = 0

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        gt_boxes = read_yolo_label(str(gt_file))
        pred_boxes = read_yolo_label(str(pred_file)) if pred_file.exists() else []
        num_gt += len(gt_boxes)
        if not pred_boxes:
            continue
        matched = np.zeros(len(gt_boxes))
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if matched[i]:
                    continue
                iou = calculate(pred_box[1:], gt_box[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            if best_iou >= threshold:
                matched[best_gt_idx] = 1
                true_positives.append(1)
            else:
                true_positives.append(0)
            scores.append(pred_box[0])
    return np.array(true_positives), np.array(scores), num_gt

def plot_confusion_matrix(tp, scores, num_gt, save_dir='results'):
    """Plot confusion matrix and precision-recall curve."""
    os.makedirs(save_dir, exist_ok=True)
    sorted_indices = np.argsort(-scores)
    tp = tp[sorted_indices]
    scores = scores[sorted_indices]
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(1 - tp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / num_gt
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()
    ap = average_precision_score(tp, scores)
    final_precision = precision[-1]
    final_recall = recall[-1]
    f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall)
    TP = np.sum(tp)
    FP = np.sum(1 - tp)
    FN = num_gt - TP
    plt.figure(figsize=(8, 6))
    cm = np.array([[TP, FP], [FN, 0]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Positive', 'Negative'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f'Final Precision: {final_precision:.4f}\n')
        f.write(f'Final Recall: {final_recall:.4f}\n')
        f.write(f'F1 Score: {f1_score:.4f}\n')
        f.write(f'Total Ground Truth: {num_gt}\n')
        f.write(f'Total Predictions: {len(scores)}\n')
        f.write(f'True Positives: {TP}\n')
        f.write(f'False Positives: {FP}\n')
        f.write(f'False Negatives: {FN}\n')
        f.write('\nConfusion Matrix:\n')
        f.write('            Predicted\n')
        f.write('             Pos  Neg\n')
        f.write(f'Actual Pos   {TP:3d}  {FP:3d}\n')
        f.write(f'      Neg   {FN:3d}    0\n')

def main():
    pred_dir = Path('runs/detect/predict9/labels')
    gt_dir = Path('dataset/labels/test')
    print("Evaluating predictions...")
    tp, scores, num_gt = evaluate_predictions(pred_dir, gt_dir)
    print("Generating plots and metrics...")
    plot_confusion_matrix(tp, scores, num_gt)
    print("Done! Results saved in 'results' directory")

if __name__ == "__main__":
    main() 