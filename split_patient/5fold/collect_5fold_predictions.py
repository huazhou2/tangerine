#!/usr/bin/env python3
"""
Collect and aggregate predictions from all 5 folds into a single dataset.

This script:
1. Finds test_predictions.csv from each fold
2. Combines them into a single file
3. Verifies patient coverage (should have ~20% per fold)
4. Generates summary statistics
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def main():
    print("="*80)
    print("COLLECTING 5-FOLD CROSS-VALIDATION PREDICTIONS")
    print("="*80)
    print("")

    # Find all fold outputs
    output_dir = Path("outputs")
    if not output_dir.exists():
        print(f"✗ ERROR: {output_dir} directory not found")
        print("No training outputs found. Make sure to run training first!")
        return False

    fold_predictions = {}
    fold_results = {}

    for fold_idx in range(5):
        # Find the latest run for this fold
        fold_outputs = list(output_dir.glob(f"fold{fold_idx}_*"))
        if not fold_outputs:
            print(f"✗ FOLD {fold_idx}: No output directory found")
            continue

        # Get the most recent one
        latest_output = max(fold_outputs, key=lambda p: p.stat().st_mtime)

        pred_file = latest_output / "test_predictions.csv"
        results_file = latest_output / "test_results.json"

        if not pred_file.exists():
            print(f"✗ FOLD {fold_idx}: test_predictions.csv not found in {latest_output}")
            continue

        # Read predictions
        df_pred = pd.read_csv(pred_file)
        fold_predictions[fold_idx] = df_pred

        # Read results
        if results_file.exists():
            with open(results_file) as f:
                fold_results[fold_idx] = json.load(f)

        print(f"✓ FOLD {fold_idx}: Loaded {len(df_pred)} test samples from {latest_output.name}")

    if not fold_predictions:
        print("✗ ERROR: No fold predictions found!")
        return False

    print(f"\n✓ Loaded predictions from {len(fold_predictions)} folds")
    print("")

    # Combine predictions
    all_predictions = pd.concat(fold_predictions.values(), ignore_index=False)
    all_predictions = all_predictions.sort_values('patient_id').reset_index(drop=True)

    print("="*80)
    print("5-FOLD COMBINED PREDICTIONS")
    print("="*80)
    print(f"\nTotal test samples: {len(all_predictions)}")
    print(f"Unique patients: {all_predictions['patient_id'].nunique()}")
    print(f"\nCancer status distribution:")
    print(all_predictions['cancer'].value_counts().to_string())

    # Compute per-fold statistics
    print(f"\n{'Fold':<5} {'Samples':<10} {'Patients':<10} {'Cancer+':<10}")
    print("-" * 40)
    for fold_idx, df_fold in fold_predictions.items():
        n_samples = len(df_fold)
        n_patients = df_fold['patient_id'].nunique()
        n_cancer = int(df_fold['cancer'].sum())
        print(f"{fold_idx:<5} {n_samples:<10} {n_patients:<10} {n_cancer:<10}")

    # Compute average metrics across folds
    print(f"\n{'='*80}")
    print("AGGREGATED METRICS ACROSS FOLDS")
    print(f"{'='*80}")

    if fold_results:
        print(f"\n{'Fold':<5} {'Best Epoch':<12} {'Val AUC':<12} {'Test AUC (raw)':<15} {'Test AUC (cal)':<15}")
        print("-" * 60)

        best_epochs = []
        val_aucs = []
        test_aucs_raw = []
        test_aucs_cal = []

        for fold_idx in range(5):
            if fold_idx not in fold_results:
                continue

            results = fold_results[fold_idx]
            best_epoch = results.get('best_epoch', 'N/A')
            val_auc = results.get('best_val_avg_auc', np.nan)
            test_auc_raw = results.get('test_avg_auc_raw', np.nan)
            test_auc_cal = results.get('test_avg_auc_cal', np.nan)

            print(f"{fold_idx:<5} {best_epoch:<12} {val_auc:<12.4f} {test_auc_raw:<15.4f} {test_auc_cal:<15.4f}")

            if not np.isnan(best_epoch):
                best_epochs.append(best_epoch)
            if not np.isnan(val_auc):
                val_aucs.append(val_auc)
            if not np.isnan(test_auc_raw):
                test_aucs_raw.append(test_auc_raw)
            if not np.isnan(test_auc_cal):
                test_aucs_cal.append(test_auc_cal)

        if val_aucs:
            print("-" * 60)
            print(f"{'MEAN':<5} {np.mean(best_epochs):<12.1f} {np.mean(val_aucs):<12.4f} {np.mean(test_aucs_raw):<15.4f} {np.mean(test_aucs_cal):<15.4f}")
            print(f"{'STDEV':<5} {np.std(best_epochs):<12.1f} {np.std(val_aucs):<12.4f} {np.std(test_aucs_raw):<15.4f} {np.std(test_aucs_cal):<15.4f}")

    # Save combined predictions
    output_csv = "5fold_combined_test_predictions.csv"
    all_predictions.to_csv(output_csv, index=False)
    print(f"\n✓ Saved combined predictions to: {output_csv}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(all_predictions),
        'unique_patients': int(all_predictions['patient_id'].nunique()),
        'cancer_positive': int(all_predictions['cancer'].sum()),
        'cancer_negative': int((all_predictions['cancer'] == 0).sum()),
        'fold_count': len(fold_predictions),
        'fold_info': {
            str(fold_idx): {
                'samples': len(df_fold),
                'patients': int(df_fold['patient_id'].nunique()),
                'cancer_positive': int(df_fold['cancer'].sum()),
            }
            for fold_idx, df_fold in fold_predictions.items()
        }
    }

    output_json = "5fold_summary.json"
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to: {output_json}")

    print(f"\n{'='*80}")
    print("✅ COLLECTION COMPLETE!")
    print(f"{'='*80}")
    print("")
    print("Output files:")
    print(f"  {output_csv}  — All test predictions from 5 folds (R-ready)")
    print(f"  {output_json}  — Summary statistics and fold breakdown")
    print("")
    print("Next steps:")
    print("  1. Load predictions in R for final analysis")
    print("  2. Compute cross-validation AUC across all folds")
    print("  3. Compare with single-split results")
    print("")

    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
