"""
Survival correlation with Lung-RADS score (1-4) treated as continuous.

- Cox PH regression: HR per 1-unit increase in LR score
- KM curves per category for visualisation
- Spearman correlation (LR vs cancer event, ignoring censoring as a sanity check)

Reads the embeddings meta CSV (lrads + cancer + time already merged).

Usage:
    python survival_lrads_analysis.py \
        --meta_csv outputs/run_20260518_123856/embeddings/embeddings_meta_layer6.csv \
        --output_dir outputs/run_20260518_123856/survival_lrads
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter


LRADS_COLORS = {1:'#4daf4a', 2:'#377eb8', 3:'#ff7f00', 4:'#e41a1c'}
LRADS_LABELS = {1:'LR-1', 2:'LR-2', 3:'LR-3', 4:'LR-4'}


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.meta_csv)
    df = df[df['lrads_category_base'].notna()].copy()
    df['lrads'] = df['lrads_category_base'].astype(int)
    df['event'] = df['cancer'].astype(int)
    df['time']  = df['time_at_event'].astype(float)
    df = df[(df['time'] > 0) & df['lrads'].isin([1, 2, 3, 4])]   # LR 1-4 only

    print(f"Patients (LR 1-4): {len(df)}  |  events: {df['event'].sum()}")
    for c in [1, 2, 3, 4]:
        sub = df[df['lrads'] == c]
        print(f"  LR-{c}: n={len(sub):4d}  events={int(sub['event'].sum()):3d} "
              f"({sub['event'].mean()*100:.1f}%)")

    # ── Cox PH (continuous LR score) ─────────────────────────────────────────
    cox_df = df[['time', 'event', 'lrads']].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time', event_col='event')

    print('\n── Cox PH (LR as continuous 1-4) ──')
    cph.print_summary()
    cox_summary = cph.summary
    cox_summary.to_csv(output_dir / 'cox_lrads_continuous.csv')
    print(f'Saved cox_lrads_continuous.csv')

    hr   = cox_summary.loc['lrads', 'exp(coef)']
    ci_l = cox_summary.loc['lrads', 'exp(coef) lower 95%']
    ci_u = cox_summary.loc['lrads', 'exp(coef) upper 95%']
    pval = cox_summary.loc['lrads', 'p']
    print(f'\nHR per 1-unit LR increase: {hr:.3f}  '
          f'95% CI [{ci_l:.3f}, {ci_u:.3f}]  p={pval:.3e}')

    # ── Spearman correlation (LR vs event, ignoring censoring) ───────────────
    rho, sp = spearmanr(df['lrads'], df['event'])
    print(f'\nSpearman ρ (LR vs cancer event): {rho:.3f}  p={sp:.3e}')

    # ── KM curves per category (visual reference) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for cat in [1, 2, 3, 4]:
        sub = df[df['lrads'] == cat]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['time'], event_observed=sub['event'],
                label=f'{LRADS_LABELS[cat]} (n={len(sub)})')
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                   color=LRADS_COLORS[cat], linewidth=2)

    ann = (f'Cox HR/unit = {hr:.2f} [{ci_l:.2f}–{ci_u:.2f}]  p={pval:.2e}\n'
           f'Spearman ρ = {rho:.3f}  p={sp:.2e}')
    ax.text(0.02, 0.05, ann, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Cancer-free probability', fontsize=12)
    ax.set_title('KM survival by Lung-RADS (LR 1–4 continuous Cox)', fontsize=13)
    ax.set_xlim(left=0); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'km_by_lrads.png', dpi=150)
    plt.close(fig)
    print(f'Saved km_by_lrads.png')
    print(f'\nAll outputs in: {output_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--meta_csv',   required=True,
                   help='embeddings_meta_layerN.csv (lrads + cancer + time columns)')
    p.add_argument('--output_dir', required=True)
    main(p.parse_args())
