"""
Post-training plots for TANGERINE 6-year survival model.
Style matches get_auc_bysex.R (Sybil analysis):
  - Colors  : rev(brewer.pal(6, "RdYlBu"))  — Year1=dark-blue, Year6=red
  - X-axis  : Specificity (1 → 0), Y-axis: Sensitivity (0 → 1)  [pROC convention]
  - Legend  : "Year N AUC = 0.XX (0.XX~0.XX)"  with bootstrap 95% CI
  - Style   : no gridlines, black border, red dashed diagonal, equal aspect

Generates:
  roc_6year_overall.png    — 6 ROC curves (years 1-6), overall cohort
  roc_6year_female.png     — same, Female subgroup
  roc_6year_male.png       — same, Male subgroup
  roc_6year_combined.png   — 1 row x 3 columns side-by-side
  confusion_matrix_yr1.png — confusion matrix at year 1, threshold=0.5
  plot_summary.json        — AUCs + confusion stats

Usage:
    python plot_survival_results.py \
        --predictions  outputs/run_XXXX/test_predictions.csv \
        --metadata     /path/to/lungct_with_mrn_anonacc.csv \
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
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

MAX_FOLLOWUP = 6

# rev(brewer.pal(6, "RdYlBu")) — Year 1=dark blue, Year 6=red
YEAR_COLORS = ['#4575B4', '#91BFDB', '#E0F3F8', '#FEE090', '#FC8D59', '#D73027']


# ── Bootstrap 95% CI for AUC ─────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    """Return (lower, upper) 95% CI via percentile bootstrap."""
    rng  = np.random.RandomState(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt  = y_true[idx]
        yp  = y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if len(aucs) < 10:
        return float('nan'), float('nan')
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Per-year Sybil mask ───────────────────────────────────────────────────────

def sybil_mask(cancer, time, year_idx):
    cancer = np.array(cancer)
    time   = np.array(time)
    mask   = ((cancer == 1) & (time <= year_idx)) | ((cancer == 0) & (time >= year_idx))
    labels = ((cancer == 1) & (time <= year_idx)).astype(int)
    return mask, labels


def compute_year_roc(df, year_idx):
    """Returns (specificity_arr, sensitivity_arr, auc, ci_lo, ci_hi, n_pos) or Nones."""
    pred_col = f'pred_{year_idx + 1}'
    mask, labels = sybil_mask(df['cancer'], df['time_at_event'], year_idx)
    y_true = labels[mask]
    y_prob = df[pred_col].values[mask]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None, None, None, None, None, None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc          = roc_auc_score(y_true, y_prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob)
    spec         = 1 - fpr   # Specificity for pROC-style x-axis

    return spec, tpr, auc, ci_lo, ci_hi, int(y_true.sum())


# ── Apply Sybil-style plot aesthetics ────────────────────────────────────────

def style_ax(ax):
    """No grid, black border, equal aspect — matches R theme."""
    ax.set_facecolor('white')
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    ax.set_aspect('equal')
    ax.set_xlim(1.0, 0.0)   # Specificity 1 → 0  (pROC convention)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.tick_params(labelsize=12)


# ── Single-cohort 6-year ROC plot ─────────────────────────────────────────────

def plot_6year_roc(df, ax, title, n_total, n_boot=1000):
    """
    Plot 6 ROC curves onto ax.
    Legend format: "Year N AUC = 0.XX (0.XX~0.XX)"  — matches R script.
    """
    style_ax(ax)

    # Red dashed diagonal reference line in (specificity, sensitivity) space
    ax.plot([1, 0], [0, 1], linestyle='--', color='red', lw=1.2, alpha=0.7)

    aucs = []
    for t in range(MAX_FOLLOWUP):
        spec, sens, auc, ci_lo, ci_hi, n_pos = compute_year_roc(df, t)
        if spec is None:
            continue

        if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
            label = (f'Year {t+1} AUC = {auc:.2f} ({ci_lo:.2f}~{ci_hi:.2f})')
        else:
            label = f'Year {t+1} AUC = {auc:.2f}'

        ax.plot(spec, sens, color=YEAR_COLORS[t], lw=2.0, label=label)
        aucs.append(auc)

    avg = np.mean(aucs) if aucs else float('nan')
    ax.set_title(f'{title}  (n={n_total})\navg AUC = {avg:.3f}', fontsize=13)

    # Legend in lower right (matching R legend.position = c(0.76, 0.2))
    ax.legend(loc='lower right', fontsize=9, framealpha=0.0,
              handlelength=1.5, borderpad=0.5)

    return aucs


# ── Confusion matrix at year 1 ────────────────────────────────────────────────

def plot_confusion(df, output_path, threshold=0.5):
    mask, labels = sybil_mask(df['cancer'], df['time_at_event'], year_idx=0)
    y_true = labels[mask]
    y_pred = (df['pred_1'].values[mask] >= threshold).astype(int)

    cm_vals = confusion_matrix(y_true, y_pred)
    disp    = ConfusionMatrixDisplay(confusion_matrix=cm_vals,
                                     display_labels=['No Cancer', 'Cancer'])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    tn, fp, fn, tp = cm_vals.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv  = tp / (tp + fp + 1e-8)
    npv  = tn / (tn + fn + 1e-8)

    ax.set_title(
        f'Year-1 Confusion Matrix  (threshold={threshold})\n'
        f'Sens={sens:.3f}  Spec={spec:.3f}  PPV={ppv:.3f}  NPV={npv:.3f}',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")
    return {'sensitivity': sens, 'specificity': spec, 'ppv': ppv, 'npv': npv,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(args.predictions)
    meta  = pd.read_csv(args.metadata)

    # Identify ID column in metadata
    id_col = next((c for c in ['ct_id', 'AnonAcc', 'MRN'] if c in meta.columns), None)
    if id_col is None:
        raise ValueError("No ID column found in metadata (ct_id / AnonAcc / MRN)")

    # Merge sex
    merged = preds.merge(
        meta[[id_col, 'sex']].drop_duplicates(subset=id_col),
        left_on='patient_id', right_on=id_col, how='left'
    )
    n_missing = merged['sex'].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} patients missing sex — excluded from sex plots")

    merged_valid = merged.dropna(subset=['sex'])
    female = merged_valid[merged_valid['sex'] == 'Female']
    male   = merged_valid[merged_valid['sex'] == 'Male']

    print(f"\nCohort sizes:")
    print(f"  Overall : {len(merged):5d}  (cancer+={int(merged['cancer'].sum())})")
    print(f"  Female  : {len(female):5d}  (cancer+={int(female['cancer'].sum())})")
    print(f"  Male    : {len(male):5d}  (cancer+={int(male['cancer'].sum())})")
    print(f"\nComputing bootstrap 95% CIs ({args.n_boot} samples per year × 3 cohorts) ...")

    # ── 1x3 combined plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    aucs_overall = plot_6year_roc(merged, axes[0], 'Overall', len(merged), args.n_boot)
    aucs_female  = plot_6year_roc(female, axes[1], 'Female',  len(female), args.n_boot)
    aucs_male    = plot_6year_roc(male,   axes[2], 'Male',    len(male),   args.n_boot)

    fig.suptitle('TANGERINE — 6-Year Lung Cancer Survival ROC Curves', fontsize=15)
    plt.tight_layout()
    plt.savefig(str(out / 'roc_6year_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: roc_6year_combined.png")

    # ── Individual plots ──────────────────────────────────────────────────────
    for subset_df, name, aucs in [
        (merged, 'overall', aucs_overall),
        (female, 'female',  aucs_female),
        (male,   'male',    aucs_male),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot_6year_roc(subset_df, ax, name.capitalize(), len(subset_df), args.n_boot)
        plt.tight_layout()
        plt.savefig(str(out / f'roc_6year_{name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: roc_6year_{name}.png  "
              f"aucs=[{', '.join(f'{a:.3f}' for a in aucs)}]")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_stats = plot_confusion(merged, out / 'confusion_matrix_yr1.png', args.threshold)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'overall': {f'year{t+1}_auc': float(aucs_overall[t]) for t in range(len(aucs_overall))},
        'female':  {f'year{t+1}_auc': float(aucs_female[t])  for t in range(len(aucs_female))},
        'male':    {f'year{t+1}_auc': float(aucs_male[t])    for t in range(len(aucs_male))},
        'overall_avg_auc': float(np.mean(aucs_overall)) if aucs_overall else None,
        'female_avg_auc':  float(np.mean(aucs_female))  if aucs_female  else None,
        'male_avg_auc':    float(np.mean(aucs_male))    if aucs_male    else None,
        'confusion_yr1':   cm_stats,
    }
    with open(out / 'plot_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAverage AUC (years 1-6):")
    print(f"  Overall : {np.mean(aucs_overall):.4f}")
    print(f"  Female  : {np.mean(aucs_female):.4f}")
    print(f"  Male    : {np.mean(aucs_male):.4f}")
    print(f"\nAll plots saved to: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True,
                   help='test_predictions.csv from finetune output dir')
    p.add_argument('--metadata',    required=True,
                   help='lungct_with_mrn_anonacc.csv')
    p.add_argument('--output_dir',  required=True,
                   help='Directory to save plots')
    p.add_argument('--threshold',   type=float, default=0.5,
                   help='Threshold for confusion matrix (default 0.5)')
    p.add_argument('--n_boot',      type=int,   default=1000,
                   help='Bootstrap samples for 95% CI (default 1000)')
    main(p.parse_args())
