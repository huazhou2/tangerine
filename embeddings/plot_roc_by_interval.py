"""
ROC curves for fine-grained time intervals:
  0–6 months, 6–12 months, Year 2, Year 3, Year 4, Year 5, Year 6

For each interval:
  cancer+: cancer==1 AND t_low <= time_ct < t_high
  cancer-: cancer==0 AND time_ct >= t_high  (followed at least to end of interval)

Requires merging test_predictions.csv with the split CSV to get continuous time
(time_ct_to_last_event_or_followup), since predictions only store integer time_at_event.

Usage:
    python plot_roc_by_interval.py \
        --predictions  outputs/run_XXXX/test_predictions.csv \
        --split_csv    dataset_splits/test.csv \
        --output_dir   outputs/run_XXXX
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ── Interval definitions ──────────────────────────────────────────────────────
# (label, t_low, t_high, pred_col)
#   t_low  : inclusive lower bound in years
#   t_high : exclusive upper bound in years
#   pred_col: which model output to use as score
INTERVALS = [
    ('0–6 mo',   0.0,  0.5,  'pred_1'),
    ('6–12 mo',  0.5,  1.0,  'pred_1'),
    ('Year 2',   1.0,  2.0,  'pred_2'),
    ('Year 3',   2.0,  3.0,  'pred_3'),
    ('Year 4',   3.0,  4.0,  'pred_4'),
    ('Year 5',   4.0,  5.0,  'pred_5'),
    ('Year 6',   5.0,  6.0,  'pred_6'),
]

COLORS = ['#4575B4', '#91BFDB', '#74C476', '#FEE090', '#FDAE61', '#FC8D59', '#D73027']


# ── Bootstrap 95% CI ─────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng  = np.random.RandomState(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) < 10:
        return float('nan'), float('nan')
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Per-interval ROC ──────────────────────────────────────────────────────────

def compute_interval_rocs(df):
    """
    df must have: cancer, time_ct (continuous years), pred_1..pred_6
    Cumulative definition (mirrors compute_auc_at_year / plot_survival_results.py):
      cancer+: cancer==1 AND time_ct < t_high  (diagnosed before end of interval)
      cancer-: cancer==0 AND time_ct >= t_high (followed at least to end of interval)
    Returns list of dicts with keys: label, n, n_pos, n_neg, auc, ci_lo, ci_hi, fpr, tpr
    """
    results = []
    for label, t_low, t_high, pred_col in INTERVALS:
        pos_mask = (df['cancer'] == 1) & (df['time_ct'] < t_high)
        neg_mask = (df['cancer'] == 0) & (df['time_ct'] >= t_high)
        mask     = pos_mask | neg_mask

        sub      = df[mask]
        y_true   = pos_mask[mask].astype(int).values
        y_prob   = sub[pred_col].values

        n_pos = int(y_true.sum())
        n_neg = int((1 - y_true).sum())

        if n_pos == 0 or n_neg == 0:
            print(f"  {label}: skipped (pos={n_pos}, neg={n_neg})")
            results.append({'label': label, 'n': n_pos + n_neg,
                            'n_pos': n_pos, 'n_neg': n_neg,
                            'auc': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
                            'fpr': None, 'tpr': None})
            continue

        auc          = roc_auc_score(y_true, y_prob)
        ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob)
        fpr, tpr, _  = roc_curve(y_true, y_prob)

        print(f"  {label}: n={n_pos+n_neg}  pos={n_pos}  neg={n_neg}  AUC={auc:.4f} ({ci_lo:.3f}–{ci_hi:.3f})")
        results.append({'label': label, 'n': n_pos + n_neg,
                        'n_pos': n_pos, 'n_neg': n_neg,
                        'auc': auc, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                        'fpr': fpr, 'tpr': tpr})
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_interval_rocs(results, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for res, color in zip(results, COLORS):
        if res['fpr'] is None:
            continue
        auc_str = (f"{res['label']}  AUC={res['auc']:.3f} "
                   f"({res['ci_lo']:.3f}–{res['ci_hi']:.3f})  "
                   f"n={res['n_pos']}/{res['n']})")
        # Plot specificity on x-axis (pROC convention: 1-fpr reversed)
        ax.plot(1 - res['fpr'], res['tpr'], color=color, lw=1.5, label=auc_str)

    ax.plot([1, 0], [0, 1], 'r--', lw=0.8)
    ax.set_xlim(1, 0)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Specificity', fontsize=11)
    ax.set_ylabel('Sensitivity', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=7, loc='lower left', frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions and split CSV, merge on patient_id to get continuous time
    preds = pd.read_csv(args.predictions)
    split = pd.read_csv(args.split_csv)

    # Identify ID column in split CSV
    id_col = next((c for c in ['ct_id', 'AnonAcc', 'MRN'] if c in split.columns), None)
    if id_col is None:
        raise ValueError("No ID column found in split CSV (ct_id / AnonAcc / MRN)")

    split = split[[id_col, 'time_ct_to_last_event_or_followup']].rename(
        columns={id_col: 'patient_id', 'time_ct_to_last_event_or_followup': 'time_ct'}
    )
    split['patient_id'] = split['patient_id'].astype(str)
    preds['patient_id'] = preds['patient_id'].astype(str)

    df = preds.merge(split, on='patient_id', how='inner')
    print(f"\nMerged: {len(df)} patients  (cancer+={int(df['cancer'].sum())})")

    print("\nInterval ROC results:")
    results = compute_interval_rocs(df)

    plot_interval_rocs(results, 'ROC by Diagnosis Interval', out_dir / 'roc_by_interval.png')

    # Save summary JSON + CSV
    summary = [{k: v for k, v in r.items() if k not in ('fpr', 'tpr')} for r in results]
    with open(out_dir / 'roc_by_interval.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir / 'roc_by_interval.json'}")

    csv_path = out_dir / 'roc_by_interval.csv'
    pd.DataFrame(summary).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True,
                   help='test_predictions.csv from training run')
    p.add_argument('--split_csv',   required=True,
                   help='dataset_splits/test.csv (has continuous time_ct_to_last_event_or_followup)')
    p.add_argument('--output_dir',  required=True,
                   help='Directory to save plots and JSON')
    main(p.parse_args())
