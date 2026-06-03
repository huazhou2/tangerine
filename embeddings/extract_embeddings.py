"""
Extract CLS-token embeddings from a trained TANGERINE survival model and
visualise them with UMAP (or t-SNE as fallback), coloured by:
  - Lung-RADS category (from scan_master CSV, column lrads_category_base)
  - Cancer status (from dataset splits)
  - Predicted 1-year risk (pred_1)

Saves:
  embeddings.npy             — float32 [N, 1024]
  embeddings_meta.csv        — patient_id, split, cancer, time_at_event,
                               pred_1..pred_6, lrads_value, lrads_category_base
  umap_lrads.png             — UMAP coloured by Lung-RADS
  umap_cancer.png            — UMAP coloured by cancer status
  umap_pred1.png             — UMAP coloured by model year-1 risk
  umap_combined.png          — all three panels side-by-side

Usage:
    python extract_embeddings.py \
        --checkpoint  outputs/run_XXXX/best_model.pth \
        --dataset_dir dataset_splits \
        --images_dir  /path/to/images_3d_swine \
        --output_dir  outputs/run_XXXX/embeddings \
        --lrads_csv   scan_master_with_lrads_value_v3_with_base.csv \
        --split       all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from tangerine_survival_model import TANGERINESurvivalModel
from survival_dataset import LungCancerSurvivalDataset

MAX_FOLLOWUP = 6

# Race label shortening for better plot display
RACE_SHORTMAP = {
    'American Indian or Alaska Native': 'Am. Indian',
    'Native Hawaiian or Pacific Islander': 'Pac. Islander',
    'Not Reported': 'Not Reported',
    'Unknown': 'Unknown',
}


# ── Model loader (same pattern as extract_attention_maps.py) ──────────────────

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    import sys
    sys.path.insert(0, '/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/3D-MAE-MedImaging')
    import models_vit
    from cumulative_probability_layer import CumulativeProbabilityLayer
    import torch.nn as nn

    model = TANGERINESurvivalModel.__new__(TANGERINESurvivalModel)
    nn.Module.__init__(model)
    model.encoder = models_vit.vit_large_patch16_yo(
        num_classes=0, drop_path_rate=0.0, global_pool=False, img_size=256)
    model.num_layers  = len(model.encoder.blocks)
    model.head        = CumulativeProbabilityLayer(model.encoder.embed_dim, MAX_FOLLOWUP)

    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_embeddings(model, dataset, device, batch_size=4, layer=-1):
    """
    Extract CLS token embeddings from a specific transformer block.
    layer=-1 (default): final layer output of forward_features  [most task-specific]
    layer=N  (0..23):   CLS token after block N via forward hook [more visual/general]
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_emb, all_pids, all_cancer, all_time, all_probs = [], [], [], [], []

    num_layers = len(model.encoder.blocks)
    use_hook   = (layer != -1)
    layer_idx  = layer % num_layers if use_hook else None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Extracting embeddings (layer={layer})'):
            vols = batch['volume'].to(device)

            if use_hook:
                captured = {}
                def _hook(m, inp, out):
                    # ViT block output: [B, 1+num_patches, dim] — index 0 is CLS
                    captured['cls'] = out[:, 0, :].float()
                h = model.encoder.blocks[layer_idx].register_forward_hook(_hook)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                final_emb = model.encoder.forward_features(vols)
                logits    = model.head(final_emb)
                probs     = torch.sigmoid(logits)

            if use_hook:
                h.remove()
                emb = captured['cls']          # [B, 1024] from intermediate layer
            else:
                emb = final_emb                # [B, 1024] from final layer

            all_emb.append(emb.cpu().float().numpy())
            all_probs.append(probs.cpu().float().numpy())
            all_pids.extend(batch['patient_id'])
            all_cancer.extend(batch['cancer'].numpy().tolist())
            all_time.extend(batch['time_at_event'].numpy().tolist())

    emb_arr   = np.concatenate(all_emb,   axis=0)   # [N, 1024]
    probs_arr = np.concatenate(all_probs, axis=0)   # [N, 6]
    return emb_arr, probs_arr, all_pids, all_cancer, all_time


# ── Dimensionality reduction ───────────────────────────────────────────────────

def reduce_embeddings(emb, method='umap', labels=None):
    """
    method='umap'          : unsupervised UMAP (default)
    method='umap_sup'      : supervised UMAP using labels (Lung-RADS or cancer)
    method='umap_pca'      : PCA(50) → UMAP (best for noisy high-dim embeddings)
    method='tsne'          : t-SNE fallback
    labels                 : 1-D array passed to supervised UMAP; NaN rows get -1
    """
    from sklearn.decomposition import PCA
    import numpy as np

    if method in ('umap', 'umap_sup', 'umap_pca'):
        try:
            import umap as umap_lib

            if method == 'umap_pca':
                print('  Running PCA(50) before UMAP...')
                emb = PCA(n_components=50, random_state=42).fit_transform(emb)

            y = None
            label_str = ''
            if method == 'umap_sup' and labels is not None:
                y = np.array(labels, dtype=float)
                y[np.isnan(y)] = -1          # UMAP ignores -1 as unknown
                y = y.astype(int)
                label_str = ' (supervised)'

            reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                    n_neighbors=30, min_dist=0.1, metric='cosine')
            xy = reducer.fit_transform(emb, y=y)
            return xy, f'UMAP{label_str}'

        except ImportError:
            print('  umap-learn not installed — falling back to t-SNE')
            method = 'tsne'

    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=30,
                   n_iter=1000, metric='cosine')
    return reducer.fit_transform(emb), 't-SNE'


# ── Plotting helpers ───────────────────────────────────────────────────────────

LRADS_COLORS = {
    0: '#aaaaaa',   # 0 — incomplete
    1: '#4daf4a',   # 1 — benign
    2: '#377eb8',   # 2 — benign appearance
    3: '#ff7f00',   # 3 — probably benign
    4: '#e41a1c',   # 4 — suspicious
}
LRADS_LABELS = {
    0: 'LR-0',
    1: 'LR-1',
    2: 'LR-2',
    3: 'LR-3',
    4: 'LR-4',
}


def plot_lrads(ax, xy, lrads_cat, method_name):
    lrads_np = np.array(lrads_cat, dtype=float)
    mask_na = np.isnan(lrads_np)
    # plot unknown in light grey first
    ax.scatter(xy[mask_na, 0], xy[mask_na, 1],
               c='#dddddd', s=3, alpha=0.2, linewidths=0, label='No LR score')
    for cat in sorted(LRADS_COLORS):
        mask = (lrads_np == cat)
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=LRADS_COLORS[cat], s=6, alpha=0.7, linewidths=0,
                   label=f'{LRADS_LABELS[cat]} (n={mask.sum()})')
    ax.set_title(f'{method_name} — Lung-RADS category', fontsize=11)
    ax.set_xlabel(f'{method_name} 1'); ax.set_ylabel(f'{method_name} 2')
    ax.legend(fontsize=7, markerscale=2, loc='best')
    ax.set_aspect('equal', 'datalim')


def plot_cancer(ax, xy, cancer, method_name):
    for val, label, color in [(0, 'No cancer', '#377eb8'), (1, 'Cancer', '#e41a1c')]:
        mask = np.array(cancer) == val
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=color, s=4 if val == 0 else 10,
                   alpha=0.4 if val == 0 else 0.85,
                   linewidths=0,
                   label=f'{label} (n={mask.sum()})')
    ax.set_title(f'{method_name} — Cancer status', fontsize=11)
    ax.set_xlabel(f'{method_name} 1'); ax.set_ylabel(f'{method_name} 2')
    ax.legend(fontsize=8, markerscale=1.5, loc='best')
    ax.set_aspect('equal', 'datalim')


def plot_pred1(ax, xy, pred1, method_name):
    sc = ax.scatter(xy[:, 0], xy[:, 1],
                    c=pred1, cmap='RdYlGn_r', s=4, alpha=0.6,
                    linewidths=0, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Year-1 risk')
    ax.set_title(f'{method_name} — Predicted year-1 risk', fontsize=11)
    ax.set_xlabel(f'{method_name} 1'); ax.set_ylabel(f'{method_name} 2')
    ax.set_aspect('equal', 'datalim')


def plot_categorical(ax, xy, values, title, method_name, palette=None):
    vals = np.array(values, dtype=object)
    categories = sorted([v for v in set(vals)
                         if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if palette is None:
        base_colors = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c',
                       '#984ea3', '#a65628', '#f781bf', '#999999']
        palette = {c: base_colors[i % len(base_colors)] for i, c in enumerate(categories)}
    # unknown first
    mask_na = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in vals])
    if mask_na.sum() > 0:
        ax.scatter(xy[mask_na, 0], xy[mask_na, 1],
                   c='#dddddd', s=3, alpha=0.2, linewidths=0, label='Unknown')
    for cat in categories:
        mask = np.array([v == cat for v in vals])
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=palette[cat], s=4, alpha=0.6, linewidths=0,
                   label=f'{cat} (n={mask.sum()})')
    ax.set_title(f'{method_name} — {title}', fontsize=11)
    ax.set_xlabel(f'{method_name} 1'); ax.set_ylabel(f'{method_name} 2')
    ax.legend(fontsize=7, markerscale=2, loc='best')
    ax.set_aspect('equal', 'datalim')


def plot_age(ax, xy, age, method_name):
    age_np = np.array(age, dtype=float)
    mask_valid = ~np.isnan(age_np)
    sc = ax.scatter(xy[mask_valid, 0], xy[mask_valid, 1],
                    c=age_np[mask_valid], cmap='coolwarm', s=4, alpha=0.6,
                    linewidths=0, vmin=40, vmax=85)
    plt.colorbar(sc, ax=ax, label='Age')
    ax.set_title(f'{method_name} — Age', fontsize=11)
    ax.set_xlabel(f'{method_name} 1'); ax.set_ylabel(f'{method_name} 2')
    ax.set_aspect('equal', 'datalim')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Create subfolders: trained/, pretrain/, combined/
    # (This script runs for fine-tuned model, so save to 'trained/')
    base_output = Path(args.output_dir)
    output_dir = base_output / 'trained'
    combined_dir = base_output / 'combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {args.checkpoint}')
    model = load_model(args.checkpoint, device)
    print(f'  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)')

    # ── Load dataset(s) ───────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    all_emb, all_probs, all_pids, all_cancer, all_time, all_splits = [], [], [], [], [], []

    for split in splits:
        print(f'Processing {split} split...')
        ds = LungCancerSurvivalDataset(
            csv_file    = dataset_dir / f'{split}.csv',
            images_dir  = args.images_dir,
            patch_size  = (256, 256, 256),
            augment     = False,
            mode        = 'val',        # always center-crop, no augment
        )
        emb, probs, pids, cancer, time = extract_embeddings(
            model, ds, device, batch_size=args.batch_size, layer=args.layer)

        all_emb.append(emb)
        all_probs.append(probs)
        all_pids.extend(pids)
        all_cancer.extend(cancer)
        all_time.extend(time)
        all_splits.extend([split] * len(pids))

    emb_arr   = np.concatenate(all_emb,   axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    N = len(all_pids)
    layer_tag = f'layer{args.layer}' if args.layer != -1 else 'layer_final'
    print(f'\nTotal embeddings: {N} × {emb_arr.shape[1]}  ({layer_tag})')

    # ── Save embeddings ───────────────────────────────────────────────────────
    np.save(output_dir / f'embeddings_{layer_tag}.npy', emb_arr)
    print(f'Saved embeddings_{layer_tag}.npy  ({emb_arr.nbytes / 1e6:.1f} MB)')

    # ── Build metadata dataframe ──────────────────────────────────────────────
    meta = pd.DataFrame({
        'patient_id':   all_pids,
        'split':        all_splits,
        'cancer':       all_cancer,
        'time_at_event': all_time,
    })
    for t in range(MAX_FOLLOWUP):
        meta[f'pred_{t+1}'] = probs_arr[:, t]

    # Merge lrads from scan_master
    if args.lrads_csv and Path(args.lrads_csv).exists():
        sm = pd.read_csv(args.lrads_csv,
                         usecols=['ct_id', 'lrads_value', 'lrads_category_base'],
                         dtype={'ct_id': str, 'lrads_category_base': 'Int64'})
        sm['ct_id'] = sm['ct_id'].astype(str).str.strip()
        meta['patient_id'] = meta['patient_id'].astype(str).str.strip()
        meta = meta.merge(sm, left_on='patient_id', right_on='ct_id', how='left')
        meta.drop(columns=['ct_id'], inplace=True)
        n_lrads = meta['lrads_category_base'].notna().sum()
        print(f'Merged Lung-RADS: {n_lrads}/{N} patients have a score')
    else:
        meta['lrads_value']         = np.nan
        meta['lrads_category_base'] = np.nan
        print('No lrads_csv provided or not found — lrads columns will be NaN')

    # Merge clinical variables (age, sex, smoke, race, ct_date) from main metadata CSV
    if args.metadata_csv and Path(args.metadata_csv).exists():
        avail = pd.read_csv(args.metadata_csv, nrows=0).columns.tolist()
        want  = ['ct_id', 'age', 'sex', 'race', 'smoke', 'ct_date']
        use_cols = [c for c in want if c in avail]
        if len(use_cols) > 1:   # at least ct_id + one clinical col
            clin = pd.read_csv(args.metadata_csv, usecols=use_cols,
                               dtype={'ct_id': str})
            clin['ct_id'] = clin['ct_id'].astype(str).str.strip()
            meta = meta.merge(clin, left_on='patient_id', right_on='ct_id', how='left')
            meta.drop(columns=['ct_id'], inplace=True, errors='ignore')
            got = [c for c in use_cols if c != 'ct_id']
            print(f'Merged clinical vars: {got}')
        else:
            print('metadata_csv found but no expected clinical columns (age/sex/smoke/race)')
    else:
        for col in ['age', 'sex', 'race', 'smoke', 'ct_date']:
            meta[col] = np.nan

    # Bin CT scan date into 5-year eras
    if 'ct_date' in meta.columns:
        yr = pd.to_datetime(meta['ct_date'], errors='coerce').dt.year
        meta['ct_era'] = pd.cut(yr, bins=[2009, 2015, 2020, 2026],
                                labels=['2010–2015', '2015–2020', '2020–2025'])
        meta['ct_era'] = meta['ct_era'].astype(object).where(meta['ct_era'].notna(), None)
    else:
        meta['ct_era'] = None

    # Shorten race labels for better plot display
    if 'race' in meta.columns:
        meta['race'] = meta['race'].replace(RACE_SHORTMAP)
    if 'lrads_category_base' in meta.columns:
        meta['lrads_category_base'] = meta['lrads_category_base'].replace(RACE_SHORTMAP)

    meta.to_csv(output_dir / f'embeddings_meta_{layer_tag}.csv', index=False)
    print(f'Saved embeddings_meta_{layer_tag}.csv')

    # ── Dimensionality reduction ──────────────────────────────────────────────
    print(f'\nRunning {args.reduction.upper()} on {N} × {emb_arr.shape[1]} embeddings...')
    lrads_labels = meta['lrads_category_base'].values
    xy, method_name = reduce_embeddings(emb_arr, method=args.reduction,
                                        labels=lrads_labels)
    np.save(output_dir / f'{args.reduction}_coords_{layer_tag}.npy', xy)
    print(f'Saved {args.reduction}_coords_{layer_tag}.npy')

    lrads_cat  = meta['lrads_category_base']
    pred1      = probs_arr[:, 0]
    age_vals   = meta['age'].tolist()    if 'age'    in meta.columns else [np.nan] * N
    sex_vals   = meta['sex'].tolist()    if 'sex'    in meta.columns else [None]   * N
    smoke_vals = meta['smoke'].tolist()  if 'smoke'  in meta.columns else [None]   * N
    race_vals  = meta['race'].tolist()   if 'race'   in meta.columns else [None]   * N
    era_vals   = meta['ct_era'].tolist() if 'ct_era' in meta.columns else [None]   * N

    CT_ERA_PALETTE = {'2010–2015': '#1b7837', '2015–2020': '#762a83', '2020–2025': '#e08214'}

    # ── Individual plots ──────────────────────────────────────────────────────
    for plot_fn, fname, kwargs in [
        (plot_lrads,  f'umap_lrads_{layer_tag}.png',  dict(lrads_cat=lrads_cat)),
        (plot_cancer, f'umap_cancer_{layer_tag}.png', dict(cancer=all_cancer)),
        (plot_pred1,  f'umap_pred1_{layer_tag}.png',  dict(pred1=pred1)),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_fn(ax, xy, method_name=method_name, **kwargs)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f'Saved {fname}')

    for title, vals, fname, palette in [
        ('Sex',            sex_vals,   f'umap_sex_{layer_tag}.png',   None),
        ('Smoking status', smoke_vals, f'umap_smoke_{layer_tag}.png', None),
        ('Race',           race_vals,  f'umap_race_{layer_tag}.png',  None),
        ('CT scan era',    era_vals,   f'umap_ctera_{layer_tag}.png', CT_ERA_PALETTE),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_categorical(ax, xy, vals, title, method_name, palette=palette)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f'Saved {fname}')

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_age(ax, xy, age_vals, method_name)
    fig.tight_layout()
    fig.savefig(output_dir / f'umap_age_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'Saved umap_age_{layer_tag}.png')

    # ── Combined 3×3 figure ───────────────────────────────────────────────────
    # row1: lrads / cancer / pred1
    # row2: sex   / smoke  / age
    # row3: race  / ct_era / (off)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    plot_lrads(      axes[0, 0], xy, lrads_cat=lrads_cat, method_name=method_name)
    plot_cancer(     axes[0, 1], xy, cancer=all_cancer,    method_name=method_name)
    plot_pred1(      axes[0, 2], xy, pred1=pred1,          method_name=method_name)
    plot_categorical(axes[1, 0], xy, sex_vals,   'Sex',            method_name)
    plot_categorical(axes[1, 1], xy, smoke_vals, 'Smoking status', method_name)
    plot_age(        axes[1, 2], xy, age_vals,             method_name)
    plot_categorical(axes[2, 0], xy, race_vals,  'Race',           method_name)
    plot_categorical(axes[2, 1], xy, era_vals,   'CT scan era',    method_name,
                     palette=CT_ERA_PALETTE)
    axes[2, 2].axis('off')
    fig.suptitle(f'TANGERINE embeddings — {N} scans — {layer_tag} (dim={emb_arr.shape[1]})',
                 fontsize=13)
    fig.tight_layout()
    # Save combined plots to shared 'combined' folder
    fig.savefig(combined_dir / f'umap_combined_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'Saved umap_combined_{layer_tag}.png to combined/')

    print(f'\nAll outputs in: {output_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,  help='Path to best_model.pth')
    p.add_argument('--dataset_dir', required=True,  help='Folder with train/val/test.csv')
    p.add_argument('--images_dir',  required=True,  help='Folder with .nii.gz CT volumes')
    p.add_argument('--output_dir',  required=True,  help='Where to save outputs')
    p.add_argument('--lrads_csv',   default=None,   help='scan_master CSV with lrads columns')
    p.add_argument('--metadata_csv', default=None,
                   help='Main metadata CSV with age, sex, smoke, race columns '
                        '(e.g. lungct_with_mrn_anonacc.csv)')
    p.add_argument('--split',       default='test',
                   choices=['train', 'val', 'test', 'all'],
                   help='Which split(s) to embed (default: test; use "all" for train+val+test)')
    p.add_argument('--reduction',   default='umap',
                   choices=['umap', 'umap_sup', 'umap_pca', 'tsne'],
                   help='Reduction method: umap (default), umap_sup (supervised by LR score), '
                        'umap_pca (PCA-50 then UMAP), tsne')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--layer',       type=int, default=-1,
                   help='Transformer block to extract CLS from (0-23). '
                        '-1 = final forward_features output (default). '
                        'Try 6, 12, 18 for more general visual features.')
    main(p.parse_args())
