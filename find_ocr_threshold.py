import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def analyze_ocr_thresholds(csv_path="ocr_results/ocr_results.csv", output_dir="threshold_analysis", beta=0.5):
    """
    Analyze OCR results to find the optimal confidence threshold for EasyOCR.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading OCR results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        df = df[df['ocr_text'] != "LOW_DPI"]
        df = df[~df['is_correct'].isna()]
        df['is_correct'] = df['is_correct'].astype(bool)
        numeric_conf = []
        for conf in df['confidence']:
            try:
                numeric_conf.append(float(conf))
            except ValueError:
                numeric_conf.append(np.nan)
        
        df['numeric_confidence'] = numeric_conf
        easyocr_df = df.dropna(subset=['numeric_confidence'])
        if len(easyocr_df) == 0:
            print("No manually verified EasyOCR results found.")
            return
            
        print(f"Found {len(easyocr_df)} manually verified EasyOCR results.")
        total_correct = easyocr_df['is_correct'].sum()
        total_incorrect = len(easyocr_df) - total_correct
        print(f"Number of correct results: {total_correct} ({total_correct/len(easyocr_df)*100:.1f}%)")
        print(f"Number of incorrect results: {total_incorrect} ({total_incorrect/len(easyocr_df)*100:.1f}%)")
        thresholds = np.arange(0.0, 1.01, 0.05)
        results = []
        for t in np.arange(0.5, 0.95, 0.01):
            if t not in thresholds:
                thresholds = np.append(thresholds, t)
        thresholds.sort()
        
        for threshold in thresholds:
            above_threshold = easyocr_df[easyocr_df['numeric_confidence'] >= threshold]
            above_threshold_count = len(above_threshold)
            above_threshold_correct = above_threshold['is_correct'].sum()
            above_threshold_incorrect = above_threshold_count - above_threshold_correct
            precision = above_threshold_correct / above_threshold_count if above_threshold_count > 0 else 0
            recall = above_threshold_correct / total_correct if total_correct > 0 else 0
            fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
            retained_pct = above_threshold_count / len(easyocr_df) if len(easyocr_df) > 0 else 0
            results.append({
                'threshold': threshold,
                'above_threshold_count': above_threshold_count,
                'above_threshold_correct': above_threshold_correct,
                'above_threshold_incorrect': above_threshold_incorrect,
                'precision': precision,
                'recall': recall,
                'fbeta_score': fbeta,
                'retained_pct': retained_pct
            })
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(output_dir, "threshold_analysis.csv")
        results_df.to_csv(results_csv_path, index=False)
        try:
            optimal_idx = results_df['fbeta_score'].idxmax()
            optimal_row = results_df.loc[optimal_idx]
            optimal_threshold = optimal_row['threshold']
            print(f"Optimal threshold (maximizing F{beta}-score): {optimal_threshold:.2f}")
            print(f"At this threshold:")
            print(f"  - Precision: {optimal_row['precision']:.2%} (minimizing incorrect predictions)")
            print(f"  - Recall: {optimal_row['recall']:.2%} (keeping correct predictions)")
            print(f"  - F{beta}-score: {optimal_row['fbeta_score']:.2%}")
            print(f"  - Predictions kept: {optimal_row['above_threshold_count']} ({optimal_row['retained_pct']*100:.1f}%)")
            print(f"  - Correct predictions kept: {optimal_row['above_threshold_correct']} of {total_correct} ({optimal_row['above_threshold_correct']/total_correct*100:.1f}%)")
            print(f"  - Incorrect predictions kept: {optimal_row['above_threshold_incorrect']} of {total_incorrect} ({optimal_row['above_threshold_incorrect']/total_incorrect*100:.1f}% of incorrect)")
        except:
            print("Could not determine optimal threshold - may not have enough data")
            optimal_threshold = None
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2, label='Precision')
        plt.plot(results_df['threshold'], results_df['recall'], 'g-', linewidth=2, label='Recall')
        plt.plot(results_df['threshold'], results_df['fbeta_score'], 'r-', linewidth=2, label=f'F{beta}-score')
        if optimal_threshold is not None:
            plt.axvline(x=optimal_threshold, color='k', linestyle='--', 
                        label=f'Optimal Threshold ({optimal_threshold:.2f})')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.title(f'EasyOCR Performance Metrics by Confidence Threshold (β={beta})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "easyocr_metrics.png"))
        plt.figure(figsize=(10, 6))
        correct_points = easyocr_df[easyocr_df['is_correct'] == True]
        incorrect_points = easyocr_df[easyocr_df['is_correct'] == False]
        plt.scatter(incorrect_points.index, incorrect_points['numeric_confidence'], 
                   color='red', alpha=0.7, label='Incorrect', marker='x')
        plt.scatter(correct_points.index, correct_points['numeric_confidence'], 
                   color='green', alpha=0.7, label='Correct', marker='o')
        if optimal_threshold is not None:
            plt.axhline(y=optimal_threshold, color='blue', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.2f})')
        plt.xlabel('Sample Index')
        plt.ylabel('Confidence Score')
        plt.title('EasyOCR Predictions by Confidence Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "easyocr_predictions.png"))
        plt.figure(figsize=(10, 6))
        bins = np.linspace(0, 1, 21)
        plt.hist([incorrect_points['numeric_confidence'], correct_points['numeric_confidence']], 
                bins=bins, stacked=True, color=['red', 'green'], 
                alpha=0.7, label=['Incorrect', 'Correct'])
        if optimal_threshold is not None:
            plt.axvline(x=optimal_threshold, color='blue', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.2f})')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Predictions')
        plt.title('Distribution of EasyOCR Predictions by Confidence Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "easyocr_histogram.png"))
        return optimal_threshold
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if os.path.exists("ocr_results/ocr_results.csv"):
        print("Starting threshold analysis...")
        optimal_threshold = analyze_ocr_thresholds(beta=0.5)
        
        if optimal_threshold is not None:
            print(f"\nRecommendation: ")
            print(f"Based on analysis of manually verified results, use a confidence threshold of {optimal_threshold:.2f}")
            print(f"for EasyOCR predictions. This threshold minimizes incorrect predictions")
            print(f"while keeping as many correct predictions as possible.")
    else:
        print("OCR results file not found. Please run ocr.py first to generate results.")
        print("Then manually verify the results by editing the 'is_correct' column in the CSV file.")
        print("After verification, run this script to find the optimal threshold.") 