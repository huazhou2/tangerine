"""
Linear probe analysis across embedding layers.

For each saved layer, fits a simple logistic regression on the 1024-dim CLS
embeddings and reports AUC for:
  - Cancer prediction
  - LR ≥ 2 vs LR-1       (any abnormal)
  - LR ≥ 3 vs LR ≤ 2     (probably benign threshold)
  - LR = 4 vs LR ≤ 3     (suspicious threshold)
  - CT era (2010-2015 vs 2015-2020 vs 2020-2025)  [macro OvR AUC]

The best layer for LR clustering is the one with highest LR AUC.
Works entirely from saved .npy + .csv files — no GPU needed.

Usage:
    python layer_probe_analysis.py \
        --embeddings_dir outputs/run_20260527_133357/embeddings \
        --output_dir     outputs/run_20260527_133357/layer_probe
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def probe_auc(X, y, binary=True, cv=5):
    """Cross-validated AUC using a linear probe (LogReg L2)."""
    valid = ~np.isnan(y.astype(float))
    if binary:
        valid &= np.isin(y, [0, 1])
    X_v, y_v = X[valid], y[valid]
    if len(np.unique(y_v)) < 2 or y_v.sum() < 10:
        return np.nan
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=0.01, max_iter=500, solver='lbfgs',
                                      multi_class='ovr', random_state=42)),
    ])
    if binary:
        probs = cross_val_predict(pipe, X_v, y_v, cv=cv, method='predict_proba')[:, 1]
        return roc_auc_score(y_v, probs)
    else:
        probs = cross_val_predict(pipe, X_v, y_v, cv=cv, method='predict_proba')
        return roc_auc_score(y_v, probs, multi_class='ovr', average='macro')


def analyse_layer(emb_file, meta_file):
    emb  = np.load(emb_file).astype(np.float32)
    meta = pd.read_csv(meta_file)

    results = {}

    # Cancer AUC
    results['cancer'] = probe_auc(emb, meta['cancer'].values.astype(float))

    # LR threshold AUCs
    lr = meta['lrads_category_base'].values.astype(float)
    lr_valid = ~np.isnan(lr)
    if lr_valid.sum() > 50:
        results['lr_ge2_vs_1'] = probe_auc(
            emb[lr_valid], (lr[lr_valid] >= 2).astype(float))
        results['lr_ge3_vs_le2'] = probe_auc(
            emb[lr_valid], (lr[lr_valid] >= 3).astype(float))
        results['lr_4_vs_le3'] = probe_auc(
            emb[lr_valid], (lr[lr_valid] == 4).astype(float))
    else:
        results['lr_ge2_vs_1'] = results['lr_ge3_vs_le2'] = results['lr_4_vs_le3'] = np.nan

    # CT era (multi-class OvR)
    if 'ct_era' in meta.columns:
        era = meta['ct_era'].values
        era_valid = np.array([v is not None and isinstance(v, str) for v in era])
        if era_valid.sum() > 50:
            results['ct_era'] = probe_auc(emb[era_valid], era[era_valid], binary=False)
        else:
            results['ct_era'] = np.nan
    else:
        results['ct_era'] = np.nan

    return results


def main(args):
    emb_dir = Path(args.embeddings_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover available layer files
    layer_files = sorted(emb_dir.glob('embeddings_layer*.npy'))
    if not layer_files:
        layer_files = sorted(emb_dir.glob('embeddings_*.npy'))

    layers, records = [], []
    for emb_file in layer_files:
        tag = emb_file.stem.replace('embeddings_', '')   # e.g. layer6, layer_final
        meta_file = emb_dir / f'embeddings_meta_{tag}.csv'
        if not meta_file.exists():
            print(f'  SKIP {tag} — meta CSV not found')
            continue

        # Extract numeric layer index for sorting
        if tag.startswith('layer') and tag[5:].isdigit():
            layer_idx = int(tag[5:])
        elif tag == 'layer_final':
            layer_idx = 999
        else:
            continue

        print(f'  Probing {tag}...')
        res = analyse_layer(emb_file, meta_file)
        res['layer'] = layer_idx
        res['tag']   = tag
        layers.append(layer_idx)
        records.append(res)
        print(f'    cancer={res["cancer"]:.3f}  '
              f'lr≥3={res["lr_ge3_vs_le2"]:.3f}  '
              f'lr=4={res["lr_4_vs_le3"]:.3f}  '
              f'era={res["ct_era"]:.3f}')

    if not records:
        print('No embedding files found.')
        return

    df = pd.DataFrame(records).sort_values('layer').reset_index(drop=True)
    df.to_csv(out_dir / 'layer_probe_aucs.csv', index=False)
    print(f'\nSaved layer_probe_aucs.csv')

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics_left  = [('cancer',        '#e41a1c', 'Cancer'),
                     ('lr_ge3_vs_le2', '#ff7f00', 'LR ≥ 3 vs ≤ 2'),
                     ('lr_4_vs_le3',   '#984ea3', 'LR = 4 vs ≤ 3'),
                     ('lr_ge2_vs_1',   '#377eb8', 'LR ≥ 2 vs 1')]
    metrics_right = [('ct_era',        '#4daf4a', 'CT era (OvR)')]

    xticks = df['layer'].values
    xlabels = [r['tag'].replace('layer', 'L') for _, r in df.iterrows()]

    for ax, metrics in [(axes[0], metrics_left), (axes[1], metrics_right)]:
        for key, color, label in metrics:
            if key in df.columns:
                ax.plot(xticks, df[key], marker='o', color=color, label=label, lw=2)
        ax.axhline(0.5, color='gray', lw=0.8, ls='--')
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_ylabel('Cross-validated AUC'); ax.set_xlabel('Layer')
        ax.set_ylim(0.45, 1.02); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    axes[0].set_title('Linear probe AUC — cancer & LR thresholds', fontsize=11)
    axes[1].set_title('Linear probe AUC — CT scan era', fontsize=11)

    # Annotate best LR layer
    best_lr_row = df.loc[df['lr_ge3_vs_le2'].idxmax()]
    axes[0].axvline(best_lr_row['layer'], color='#ff7f00', lw=1.5, ls=':',
                    label=f'Best LR layer: {best_lr_row["tag"]}')
    axes[0].legend(fontsize=9)

    fig.suptitle('TANGERINE — Linear probe AUC by encoder layer', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / 'layer_probe_aucs.png', dpi=150)
    plt.close(fig)
    print(f'Saved layer_probe_aucs.png')

    print(f'\n=== Best layers ===')
    for key, label in [('cancer', 'Cancer'), ('lr_ge3_vs_le2', 'LR ≥ 3'),
                        ('lr_4_vs_le3', 'LR = 4'), ('ct_era', 'CT era')]:
        if key in df.columns and df[key].notna().any():
            best = df.loc[df[key].idxmax()]
            print(f'  {label:20s}: {best["tag"]}  AUC={best[key]:.3f}')

    print(f'\nAll outputs in: {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings_dir', required=True,
                   help='Directory with embeddings_layerN.npy + embeddings_meta_layerN.csv')
    p.add_argument('--output_dir',     required=True)
    main(p.parse_args())
