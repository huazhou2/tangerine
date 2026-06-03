"""
plot_chro_split_analysis.py

Two-panel figure illustrating why chronological splitting breaks survival labels:
  Panel A: scatter of ct_date vs follow-up time, coloured by train/val/test split,
           cancer cases highlighted.
  Panel B: cancer event counts per year (Y1-Y6) for each split — shows NaN years.

Usage:
    python plot_chro_split_analysis.py \
        --metadata  /path/to/lungct_with_mrn_anonacc.csv \
        --output    chro_split_analysis.png \
        [--train_ratio 0.70] [--val_ratio 0.15]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SPLIT_COLORS = {'train': '#4878CF', 'val': '#6ACC65', 'test': '#D65F5F'}
YEAR_THRESHOLDS = [1, 2, 3, 4, 5, 6]


def assign_splits(df, train_ratio, val_ratio):
    df = df.copy()
    df['ct_date'] = pd.to_datetime(df['ct_date'], errors='coerce')
    df = df.dropna(subset=['ct_date']).sort_values('ct_date').reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    df['split'] = 'test'
    df.loc[:n_train - 1, 'split'] = 'train'
    df.loc[n_train:n_train + n_val - 1, 'split'] = 'val'
    return df


def informative_fraction(df):
    """For cancer-free cases, fraction with follow-up >= yr (i.e. informative negatives)."""
    rows = []
    for split in ['train', 'val', 'test']:
        sub = df[(df['split'] == split) & (df['cancer'] == 0)]
        row = {'split': split, 'n': len(sub)}
        for yr in YEAR_THRESHOLDS:
            row[f'Y{yr}'] = (sub['time_ct_to_last_event_or_followup'] >= yr).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows).set_index('split')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata',     default='/Volumes/hua_mac/research/aris/deeplearning/tangerine/lungct_with_mrn_anonacc.csv')
    parser.add_argument('--output',       default='/Volumes/hua_mac/research/aris/deeplearning/tangerine/tangerine_20260406/codes/chro_split_analysis.png')
    parser.add_argument('--train_ratio',  type=float, default=0.70)
    parser.add_argument('--val_ratio',    type=float, default=0.15)
    args = parser.parse_args()

    df_raw = pd.read_csv(args.metadata)
    df_raw = df_raw[df_raw['time_ct_to_last_event_or_followup'] >= 0].copy()
    df = assign_splits(df_raw, args.train_ratio, args.val_ratio)

    # cutoff dates
    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    date_cut1 = df['ct_date'].iloc[n_train]          # train/val boundary
    date_cut2 = df['ct_date'].iloc[n_train + n_val]  # val/test boundary

    counts = informative_fraction(df)

    base_dir = Path(args.metadata).parent / 'tangerine_20260406' / 'codes'
    # Use same folder as this script
    script_dir = Path(__file__).parent

    # ── Figure 1: ct_followup ───────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(8, 6))
    fig1.suptitle('CT Date vs. Follow-up Time',
                  fontsize=12, fontweight='bold')

    for split in ['train', 'val', 'test']:
        sub = df[(df['split'] == split) & (df['cancer'] == 0)]
        ax.scatter(sub['ct_date'], sub['time_ct_to_last_event_or_followup'],
                   c=SPLIT_COLORS[split], s=4, alpha=0.25, linewidths=0)

    cancer_all = df[df['cancer'] == 1]
    ax.scatter(cancer_all['ct_date'], cancer_all['time_ct_to_last_event_or_followup'],
               c='red', s=20, alpha=0.75, marker='o',
               edgecolors='darkred', linewidths=0.4, zorder=5)

    ax.axvline(date_cut1, color='gray', lw=1.5, ls='--')
    ax.axvline(date_cut2, color='gray', lw=1.5, ls=':')

    for yr in YEAR_THRESHOLDS:
        ax.axhline(yr, color='#aaaaaa', lw=0.6, ls='-', zorder=0)
        ax.text(df['ct_date'].max(), yr + 0.05, f'Y{yr}', fontsize=7,
                color='#888888', va='bottom', ha='right')

    ax.set_xlabel('CT Date', fontsize=11)
    ax.set_ylabel('Follow-up / Time-to-Event (years)', fontsize=11)
    ax.set_title('', fontsize=10)
    ax.tick_params(axis='x', rotation=30)

    handles = [
        mpatches.Patch(color=SPLIT_COLORS['train'], label='Train'),
        mpatches.Patch(color=SPLIT_COLORS['val'],   label='Val'),
        mpatches.Patch(color=SPLIT_COLORS['test'],  label='Test'),
        plt.Line2D([0],[0], color='gray', ls='--', lw=1.5, label='train/val cut'),
        plt.Line2D([0],[0], color='gray', ls=':',  lw=1.5, label='val/test cut'),
        plt.scatter([],[], marker='o', c='red', s=20, edgecolors='darkred', linewidths=0.4, label='cancer case'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right')

    fig1.tight_layout()
    out1 = script_dir / 'ct_followup.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f'Saved: {out1}')
    plt.close(fig1)

    # ── Figure 2: datasplit_followup ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.suptitle('Informative Cancer-free Cases per Year by Split',
                  fontsize=13, fontweight='bold')

    x     = np.arange(len(YEAR_THRESHOLDS))
    width = 0.25
    for i, split in enumerate(['train', 'val', 'test']):
        vals = [counts.loc[split, f'Y{yr}'] for yr in YEAR_THRESHOLDS]
        bars = ax2.bar(x + (i - 1) * width, vals, width,
                       label=split,
                       color=SPLIT_COLORS[split], alpha=0.85, edgecolor='k', linewidth=0.4)
        for bar, v in zip(bars, vals):
            label_txt = f'{v:.0f}%'
            color_txt = 'red' if v < 5 else 'black'
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, label_txt,
                     ha='center', va='bottom', fontsize=7, color=color_txt,
                     fontweight='bold' if v < 5 else 'normal')

    ax2.axhline(5, color='red', lw=1, ls='--', alpha=0.6, label='<5% NaN AUC risk')
    ax2.fill_between([-0.5, len(YEAR_THRESHOLDS) - 0.5], 0, 5,
                     color='red', alpha=0.07, zorder=0)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Y{yr} ≥{yr}yr' for yr in YEAR_THRESHOLDS])
    ax2.set_xlabel('Minimum follow-up required', fontsize=11)
    ax2.set_ylabel('% cancer-free cases with sufficient follow-up', fontsize=11)
    ax2.set_title('', fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)

    fig2.tight_layout()
    out2 = script_dir / 'datasplit_followup.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'Saved: {out2}')
    plt.close(fig2)

    # print summary table
    print('\n% cancer-free cases with follow-up >= N years (informative negatives):')
    print(counts.drop(columns='n').round(1).to_string())
    total = df.groupby('split')['cancer'].sum().rename('total_cancer')
    print('\nTotal cancer per split:')
    print(total)


if __name__ == '__main__':
    main()
