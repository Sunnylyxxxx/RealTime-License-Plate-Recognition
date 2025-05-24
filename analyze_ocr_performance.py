import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_ocr_performance(csv_path="ocr_results/ocr_results.csv", output_dir="ocr_analysis"):
    """
    Analyze OCR performance from the results CSV file.
    Focus on correct and incorrect results by model and overall.
    
    Expected CSV columns:
    plate_name, ocr_text, confidence, model_used, is_correct
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading OCR results from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("CSV file is empty.")
            return
        print(f"Loaded {len(df)} OCR results.")
        verified_df = df[~df['is_correct'].isna()]
        print(f"Manually verified entries: {len(verified_df)} ({len(verified_df)/len(df)*100:.1f}%)")
        if len(verified_df) == 0:
            print("No manually verified entries found. Please verify OCR results first.")
            return
        verified_df['is_correct'] = verified_df['is_correct'].astype(bool)
        print("\n--- Overall OCR Performance ---")
        correct_count = verified_df['is_correct'].sum()
        incorrect_count = len(verified_df) - correct_count
        accuracy = correct_count / len(verified_df)
        print(f"Total OCR results: {len(verified_df)}")
        print(f"Correct results: {correct_count} ({correct_count/len(verified_df)*100:.1f}%)")
        print(f"Incorrect results: {incorrect_count} ({incorrect_count/len(verified_df)*100:.1f}%)")
        print(f"Overall accuracy: {accuracy:.2%}")
        plt.figure(figsize=(8, 8))
        plt.pie([correct_count, incorrect_count], 
                labels=['Correct', 'Incorrect'], 
                colors=['green', 'red'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.05, 0))
        plt.title('Overall OCR Results')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "overall_results.png"))
        if 'model_used' in verified_df.columns:
            print("\n--- OCR Performance by Model ---")
            model_results = verified_df.groupby('model_used')['is_correct'].agg(['count', 'sum'])
            model_results.columns = ['Total', 'Correct']
            model_results['Incorrect'] = model_results['Total'] - model_results['Correct']
            model_results['Accuracy'] = model_results['Correct'] / model_results['Total']
            model_results = model_results.sort_values('Total', ascending=False)
            for model, row in model_results.iterrows():
                model_name = model if pd.notnull(model) and model != "" else "Unspecified"
                print(f"\nModel: {model_name}")
                print(f"  Total results: {row['Total']}")
                print(f"  Correct: {row['Correct']} ({row['Correct']/row['Total']*100:.1f}%)")
                print(f"  Incorrect: {row['Incorrect']} ({row['Incorrect']/row['Total']*100:.1f}%)")
                print(f"  Accuracy: {row['Accuracy']:.2%}")
            plt.figure(figsize=(12, 8))
            valid_models = model_results.index[pd.notnull(model_results.index) & (model_results.index != "")]
            if len(valid_models) > 0:
                valid_results = model_results.loc[valid_models]
                valid_results = valid_results.sort_values('Accuracy', ascending=False)
                x = np.arange(len(valid_results))
                width = 0.8
                plt.bar(x, valid_results['Correct'], width, label='Correct', color='green')
                plt.bar(x, valid_results['Incorrect'], width, bottom=valid_results['Correct'], label='Incorrect', color='red')
                for i, acc in enumerate(valid_results['Accuracy']):
                    plt.text(i, valid_results['Total'][i] + 1, f"{acc:.1%}", 
                            ha='center', va='bottom', fontweight='bold')
                plt.xlabel('Model')
                plt.ylabel('Number of Results')
                plt.title('OCR Results by Model')
                plt.xticks(x, valid_results.index, rotation=45, ha='right')
                plt.legend(loc='upper right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "results_by_model.png"))
                if len(valid_results) > 3:
                    plt.figure(figsize=(12, max(6, len(valid_results) * 0.6)))
                    y = np.arange(len(valid_results))
                    plt.barh(y, valid_results['Correct'], width, label='Correct', color='green')
                    plt.barh(y, valid_results['Incorrect'], width, left=valid_results['Correct'], label='Incorrect', color='red')
                    for i, acc in enumerate(valid_results['Accuracy']):
                        plt.text(valid_results['Total'][i] + 1, i, f"{acc:.1%}", 
                                va='center', fontweight='bold')
                    plt.ylabel('Model')
                    plt.xlabel('Number of Results')
                    plt.title('OCR Results by Model')
                    plt.yticks(y, valid_results.index)
                    plt.legend(loc='lower right')
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "results_by_model_horizontal.png"))
        print(f"\nAnalysis complete. Results saved to {output_dir}/")
        return verified_df
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if os.path.exists("ocr_results/ocr_results.csv"):
        print("Starting OCR performance analysis...")
        analyze_ocr_performance()
    else:
        print("OCR results file not found. Please run ocr.py first to generate results,")
        print("and manually verify the results by editing the 'is_correct' column in the CSV file.") 