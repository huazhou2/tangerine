"""
Regenerate all UMAP plots from already-computed embeddings + UMAP coords.
No GPU needed — reads .npy and .csv files, rewrites all PNGs.

Usage:
    python replot_embeddings.py \
        --output_dir outputs/run_20260518_123856/embeddings \
        --layers 12 18 23
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from extract_embeddings import (
    plot_lrads, plot_cancer, plot_pred1,
    plot_categorical, plot_age,
)

MAX_FOLLOWUP = 6


def replot_layer(output_dir, layer):
    layer_tag = f'layer{layer}' if layer != -1 else 'layer_final'
    emb_file   = output_dir / f'embeddings_{layer_tag}.npy'
    meta_file  = output_dir / f'embeddings_meta_{layer_tag}.csv'
    coord_file = output_dir / f'umap_coords_{layer_tag}.npy'

    if not emb_file.exists():
        print(f'  SKIP layer {layer} — embeddings file not found'); return
    if not coord_file.exists():
        print(f'  SKIP layer {layer} — umap_coords file not found'); return

    emb  = np.load(emb_file)
    xy   = np.load(coord_file)
    meta = pd.read_csv(meta_file)
    N    = len(meta)

    method_name = 'UMAP'
    lrads_cat  = meta['lrads_category_base']
    cancer     = meta['cancer'].tolist()
    pred1      = meta['pred_1'].values if 'pred_1' in meta.columns else None
    age_vals   = meta['age'].tolist()   if 'age'   in meta.columns else [np.nan]*N
    sex_vals   = meta['sex'].tolist()   if 'sex'   in meta.columns else [None]*N
    smoke_vals = meta['smoke'].tolist() if 'smoke' in meta.columns else [None]*N
    race_vals  = meta['race'].tolist()  if 'race'  in meta.columns else [None]*N

    print(f'  Layer {layer}: N={N}  '
          f'age_filled={sum(v is not None and not (isinstance(v,float) and np.isnan(v)) for v in age_vals)}  '
          f'smoke_cats={set(v for v in smoke_vals if v is not None and not (isinstance(v,float) and np.isnan(float(v) if isinstance(v,float) else 0)))}')

    # individual plots
    for fn, fname, kw in [
        (plot_lrads,  f'umap_lrads_{layer_tag}.png',  dict(lrads_cat=lrads_cat)),
        (plot_cancer, f'umap_cancer_{layer_tag}.png', dict(cancer=cancer)),
    ]:
        fig, ax = plt.subplots(figsize=(7,6))
        fn(ax, xy, method_name=method_name, **kw)
        fig.tight_layout(); fig.savefig(output_dir/fname, dpi=150); plt.close(fig)

    if pred1 is not None:
        fig, ax = plt.subplots(figsize=(7,6))
        plot_pred1(ax, xy, pred1=pred1, method_name=method_name)
        fig.tight_layout()
        fig.savefig(output_dir / f'umap_pred1_{layer_tag}.png', dpi=150)
        plt.close(fig)

    for title, vals, fname in [
        ('Sex',            sex_vals,   f'umap_sex_{layer_tag}.png'),
        ('Smoking status', smoke_vals, f'umap_smoke_{layer_tag}.png'),
        ('Race',           race_vals,  f'umap_race_{layer_tag}.png'),
    ]:
        fig, ax = plt.subplots(figsize=(7,6))
        plot_categorical(ax, xy, vals, title, method_name)
        fig.tight_layout(); fig.savefig(output_dir/fname, dpi=150); plt.close(fig)
        print(f'    saved umap_{fname.split("umap_")[1]}')

    fig, ax = plt.subplots(figsize=(7,6))
    plot_age(ax, xy, age_vals, method_name)
    fig.tight_layout()
    fig.savefig(output_dir / f'umap_age_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'    saved umap_age_{layer_tag}.png')

    # combined 2×3
    has_pred = pred1 is not None
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plot_lrads(      axes[0,0], xy, lrads_cat=lrads_cat, method_name=method_name)
    plot_cancer(     axes[0,1], xy, cancer=cancer,        method_name=method_name)
    if has_pred:
        plot_pred1(  axes[0,2], xy, pred1=pred1,          method_name=method_name)
    else:
        axes[0,2].axis('off')
    plot_categorical(axes[1,0], xy, sex_vals,   'Sex',            method_name)
    plot_categorical(axes[1,1], xy, smoke_vals, 'Smoking status', method_name)
    plot_age(        axes[1,2], xy, age_vals,               method_name)
    fig.suptitle(
        f'TANGERINE embeddings — {N} scans — {layer_tag} (dim={emb.shape[1]})', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f'umap_combined_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'    saved umap_combined_{layer_tag}.png')


def main(args):
    output_dir = Path(args.output_dir)
    for layer in args.layers:
        print(f'\n=== Layer {layer} ===')
        replot_layer(output_dir, layer)
    print('\nDone.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[6, 12, 18, 23])
    main(p.parse_args())
