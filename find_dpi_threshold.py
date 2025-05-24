import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_data_from_files(ocr_results_dir="easyocr_results"):
    """
    Extract DPI and OCR-Ready values from all result files
    Returns a list of (dpi, is_ocr_ready) tuples
    """
    results = []
    if not os.path.isabs(ocr_results_dir):
        ocr_results_dir = os.path.join(os.getcwd(), ocr_results_dir)
    
    print(f"Looking for OCR results in: {ocr_results_dir}")
    if not os.path.exists(ocr_results_dir):
        print(f"Error: Directory {ocr_results_dir} does not exist")
        return results
    result_files = []
    for root, dirs, files in os.walk(ocr_results_dir):
        for file in files:
            if file.endswith('_ocr.txt'):
                full_path = os.path.join(root, file)
                result_files.append((file, full_path))
    
    print(f"Found {len(result_files)} OCR result files in {ocr_results_dir} and subdirectories.")
    abs_specified_path = r"D:\mlproject\easyocr_results"
    if ocr_results_dir != abs_specified_path and os.path.exists(abs_specified_path):
        try:
            alt_result_files = []
            for root, dirs, files in os.walk(abs_specified_path):
                for file in files:
                    if file.endswith('_ocr.txt'):
                        full_path = os.path.join(root, file)
                        alt_result_files.append((file, full_path))
            
            print(f"Alternative path {abs_specified_path} contains {len(alt_result_files)} OCR result files.")
            if len(alt_result_files) > len(result_files):
                print(f"Using alternative path with more files")
                result_files = alt_result_files
        except Exception as e:
            print(f"Error checking alternative path: {e}")
    
    valid_results = 0
    failed_files = 0
    
    for filename, file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract DPI value
                dpi_match = re.search(r'Image DPI: ([0-9.]+)', content)
                if dpi_match:
                    dpi = float(dpi_match.group(1))
                else:
                    print(f"No DPI found in {filename}")
                    failed_files += 1
                    continue
                ocr_match = re.search(r'OCR-Ready: (TRUE|FALSE|True|False)', content, re.IGNORECASE)
                if ocr_match:
                    is_ready = ocr_match.group(1).upper() == 'TRUE'
                else:
                    print(f"No OCR-Ready status found in {filename}")
                    failed_files += 1
                    continue
                
                results.append((dpi, is_ready))
                valid_results += 1
                if valid_results <= 5:
                    print(f"Sample {valid_results}: File={filename}, DPI={dpi}, OCR-Ready={is_ready}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_files += 1
    
    print(f"Successfully processed {valid_results} files, {failed_files} files failed.")
    print(f"Found {len([r for r in results if r[1]])} OCR-Ready=True and {len([r for r in results if not r[1]])} OCR-Ready=False files.")
    if valid_results < 300:
        print("Trying alternative extraction method - looking for DPI in filenames...")
        
        alt_results = []
        pattern = re.compile(r'^(\d+(?:\.\d+)?)dpi_(.+)_ocr\.txt$')
        
        for filename, file_path in result_files:
            match = pattern.match(filename)
            if match:
                dpi = float(match.group(1))
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        ocr_match = re.search(r'OCR-Ready: (TRUE|FALSE|True|False)', content, re.IGNORECASE)
                        is_ready = ocr_match.group(1).upper() == 'TRUE' if ocr_match else False
                        
                        alt_results.append((dpi, is_ready))
                except:
                    alt_results.append((dpi, False))
        
        if len(alt_results) > len(results):
            print(f"Alternative method found {len(alt_results)} results, using these instead.")
            results = alt_results
    
    return results

def evaluate_threshold(data, threshold):
    """
    Evaluate a threshold value by counting true positives, false positives, etc.
    Returns accuracy, precision, recall, and f1 score
    """
    true_positives = sum(1 for dpi, is_ready in data if dpi >= threshold and is_ready)
    false_positives = sum(1 for dpi, is_ready in data if dpi >= threshold and not is_ready)
    true_negatives = sum(1 for dpi, is_ready in data if dpi < threshold and not is_ready)
    false_negatives = sum(1 for dpi, is_ready in data if dpi < threshold and is_ready)
    total = len(data)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total': total
    }

def find_optimal_threshold(data):
    """
    Find optimal DPI threshold by testing a range of values
    Focus on maximizing true positives while minimizing false positives
    """
    dpi_values = sorted(set(dpi for dpi, _ in data))
    if not dpi_values:
        return None, []
    min_dpi = min(dpi_values)
    max_dpi = max(dpi_values)
    test_thresholds = sorted(set(dpi_values + np.linspace(min_dpi, max_dpi, 100).tolist()))
    evaluations = [evaluate_threshold(data, threshold) for threshold in test_thresholds]
    def custom_score(eval_dict):
        beta = 0.5
        if eval_dict['precision'] == 0 and eval_dict['recall'] == 0:
            return 0
        return (1 + beta**2) * (eval_dict['precision'] * eval_dict['recall']) / \
               ((beta**2 * eval_dict['precision']) + eval_dict['recall']) if \
               ((beta**2 * eval_dict['precision']) + eval_dict['recall']) > 0 else 0
    best_eval = max(evaluations, key=custom_score)
    optimal_threshold = best_eval['threshold']
    max_accuracy_eval = max(evaluations, key=lambda x: x['accuracy'])
    max_accuracy_threshold = max_accuracy_eval['threshold']
    print(f"F-beta Score Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Max Accuracy Threshold: {max_accuracy_threshold:.2f}")
    for eval_dict in evaluations:
        eval_dict['custom_score'] = custom_score(eval_dict)
    
    return optimal_threshold, evaluations

def plot_results(evaluations, optimal_threshold, output_dir="dpi_threshold_analysis"):
    """
    Create visualizations of the threshold analysis
    """
    if not evaluations:
        print("No data to plot")
        return
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(evaluations)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'DPI Threshold Analysis (Optimal: {optimal_threshold:.2f})', fontsize=16)
    axs[0, 0].plot(df['threshold'], df['accuracy'], label='Accuracy')
    axs[0, 0].plot(df['threshold'], df['precision'], label='Precision')
    axs[0, 0].plot(df['threshold'], df['recall'], label='Recall')
    axs[0, 0].plot(df['threshold'], df['f1'], label='F1 Score')
    axs[0, 0].plot(df['threshold'], df['custom_score'], label='Custom Score (F-beta)', linewidth=2)
    axs[0, 0].axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    axs[0, 0].set_xlabel('DPI Threshold')
    axs[0, 0].set_ylabel('Score')
    axs[0, 0].set_title('Performance Metrics')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    dpi_values = [dpi for dpi, _ in data]
    axs[0, 1].hist(dpi_values, bins=30, alpha=0.7)
    axs[0, 1].axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    axs[0, 1].set_xlabel('DPI Value')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_title('Distribution of DPI Values')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[1, 0].plot(df['threshold'], df['true_positives'], label='True Positives')
    axs[1, 0].plot(df['threshold'], df['false_positives'], label='False Positives')
    axs[1, 0].plot(df['threshold'], df['true_negatives'], label='True Negatives')
    axs[1, 0].plot(df['threshold'], df['false_negatives'], label='False Negatives')
    axs[1, 0].axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    axs[1, 0].set_xlabel('DPI Threshold')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title('Confusion Matrix Counts')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    tp_ratio = df['true_positives'] / (df['true_positives'] + df['false_negatives'])
    fp_ratio = df['false_positives'] / (df['false_positives'] + df['true_negatives'])
    
    for i, thresh in enumerate(df['threshold']):
        if i % 5 == 0:
            axs[1, 1].annotate(f"{thresh:.1f}", (fp_ratio[i], tp_ratio[i]), fontsize=8)
    
    axs[1, 1].scatter(fp_ratio, tp_ratio, alpha=0.5)
    axs[1, 1].plot(fp_ratio, tp_ratio)
    opt_idx = df[df['threshold'] == optimal_threshold].index[0]
    axs[1, 1].scatter([fp_ratio[opt_idx]], [tp_ratio[opt_idx]], color='red', s=100, marker='*', 
                     label=f'Optimal ({optimal_threshold:.2f})')
    
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_title('ROC-like Trade-off Curve')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dpi_threshold_analysis.png'), dpi=300)
    plt.figure(figsize=(12, 8))
    plt.hist([dpi for dpi, ready in data if ready], bins=30, alpha=0.5, label='OCR-Ready=TRUE', color='green')
    plt.hist([dpi for dpi, ready in data if not ready], bins=30, alpha=0.5, label='OCR-Ready=FALSE', color='red')
    plt.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2, label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.title('DPI Distribution by OCR-Ready Status')
    plt.xlabel('DPI Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dpi_distribution_by_status.png'), dpi=300)
    plt.figure(figsize=(12, 8))
    plt.plot(df['threshold'], df['accuracy'], label='Accuracy', linewidth=2)
    plt.plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
    plt.plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
    plt.plot(df['threshold'], df['f1'], label='F1 Score', linewidth=2)
    plt.plot(df['threshold'], df['custom_score'], label='Custom Score (F-beta)', linewidth=3, color='magenta')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.2f}')
    plt.title('Performance Metrics by DPI Threshold')
    plt.xlabel('DPI Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
    plt.show()
    best_eval = next((e for e in evaluations if e['threshold'] == optimal_threshold), None)
    if best_eval:
        print(f"\nOptimal DPI Threshold: {optimal_threshold:.2f}")
        print(f"Accuracy: {best_eval['accuracy']:.4f}")
        print(f"Precision: {best_eval['precision']:.4f}")
        print(f"Recall: {best_eval['recall']:.4f}")
        print(f"F1 Score: {best_eval['f1']:.4f}")
        print(f"Custom Score (F-beta): {best_eval['custom_score']:.4f}")
        
        print(f"\nAt this threshold:")
        print(f"- {best_eval['true_positives']} plates above threshold are correctly identified as OCR-ready")
        print(f"- {best_eval['false_positives']} plates above threshold are incorrectly identified as OCR-ready")
        print(f"- {best_eval['true_negatives']} plates below threshold are correctly identified as not OCR-ready")
        print(f"- {best_eval['false_negatives']} plates below threshold are incorrectly identified as not OCR-ready")
        print(f"Total Images: {best_eval['total']}")

def save_threshold_results(evaluations, optimal_threshold, data, output_dir="dpi_threshold_analysis"):
    """
    Save comprehensive threshold analysis results to a text file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "threshold_results.txt")
    
    with open(filename, "w") as f:
        f.write("DPI THRESHOLD ANALYSIS RESULTS\n")
        f.write("============================\n\n")
        f.write(f"Total images analyzed: {len(data)}\n")
        f.write(f"OCR-Ready=TRUE images: {sum(1 for _, is_ready in data if is_ready)}\n")
        f.write(f"OCR-Ready=FALSE images: {sum(1 for _, is_ready in data if not is_ready)}\n\n")
        best_eval = next((e for e in evaluations if e['threshold'] == optimal_threshold), None)
        if best_eval:
            f.write(f"OPTIMAL DPI THRESHOLD: {optimal_threshold:.2f}\n")
            f.write(f"Accuracy: {best_eval['accuracy']:.4f}\n")
            f.write(f"Precision: {best_eval['precision']:.4f}\n")
            f.write(f"Recall: {best_eval['recall']:.4f}\n")
            f.write(f"F1 Score: {best_eval['f1']:.4f}\n")
            f.write(f"Custom Score (F-beta): {best_eval['custom_score']:.4f}\n\n")
            f.write("CONFUSION MATRIX AT OPTIMAL THRESHOLD:\n")
            f.write("------------------------------------\n")
            f.write(f"True Positives (TP): {best_eval['true_positives']} - Plates above threshold correctly identified as OCR-ready\n")
            f.write(f"False Positives (FP): {best_eval['false_positives']} - Plates above threshold incorrectly identified as OCR-ready\n")
            f.write(f"True Negatives (TN): {best_eval['true_negatives']} - Plates below threshold correctly identified as not OCR-ready\n")
            f.write(f"False Negatives (FN): {best_eval['false_negatives']} - Plates below threshold incorrectly identified as OCR-ready\n\n")
        
        f.write("ALL THRESHOLD EVALUATIONS:\n")
        f.write("-------------------------\n")
        f.write("Threshold\tAccuracy\tPrecision\tRecall\tF1 Score\tTP\tFP\tTN\tFN\tCustom Score\n")
        
        for eval_dict in sorted(evaluations, key=lambda e: e['threshold']):
            f.write(f"{eval_dict['threshold']:.2f}\t")
            f.write(f"{eval_dict['accuracy']:.4f}\t")
            f.write(f"{eval_dict['precision']:.4f}\t")
            f.write(f"{eval_dict['recall']:.4f}\t")
            f.write(f"{eval_dict['f1']:.4f}\t")
            f.write(f"{eval_dict['true_positives']}\t")
            f.write(f"{eval_dict['false_positives']}\t")
            f.write(f"{eval_dict['true_negatives']}\t")
            f.write(f"{eval_dict['false_negatives']}\t")
            f.write(f"{eval_dict['custom_score']:.4f}\n")

    csv_filename = os.path.join(output_dir, "dpi_data.csv")
    with open(csv_filename, "w") as f:
        f.write("DPI,OCR_Ready\n")
        for dpi, is_ready in data:
            f.write(f"{dpi},{1 if is_ready else 0}\n")
    eval_csv = os.path.join(output_dir, "threshold_evaluations.csv")
    pd.DataFrame(evaluations).to_csv(eval_csv, index=False)
    
    print(f"\nComplete threshold analysis saved to {output_dir} folder")

if __name__ == "__main__":
    data = extract_data_from_files()
    if len(data) < 300:
        print("\nSearching entire project directory for OCR result files...")
        project_dir = r"D:\mlproject"
        if os.path.exists(project_dir):
            all_results = []
            dpi_pattern = re.compile(r'^(\d+(?:\.\d+)?)dpi_')
            for root, dirs, files in os.walk(project_dir):
                for file in files:
                    if file.endswith('_ocr.txt'):
                        full_path = os.path.join(root, file)
                        match = dpi_pattern.match(file)
                        if match:
                            dpi = float(match.group(1))
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    ocr_match = re.search(r'OCR-Ready: (TRUE|FALSE|True|False)', content, re.IGNORECASE)
                                    is_ready = ocr_match.group(1).upper() == 'TRUE' if ocr_match else False
                                    all_results.append((dpi, is_ready))
                            except:
                                all_results.append((dpi, False))
            
            if len(all_results) > len(data):
                print(f"Found {len(all_results)} OCR results in project directory.")
                data = all_results
                
    if not data:
        print("No data found in OCR result files")
    else:
        print(f"Found {len(data)} OCR results to analyze")
        optimal_threshold, evaluations = find_optimal_threshold(data)
        plot_results(evaluations, optimal_threshold)
        save_threshold_results(evaluations, optimal_threshold, data)
