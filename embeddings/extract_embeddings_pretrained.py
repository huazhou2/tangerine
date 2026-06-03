"""
Embeddings from the raw MAE-pretrained TANGERINE encoder (no fine-tuning).
Parallel to extract_embeddings.py — only the model loader differs.

Usage:
    python extract_embeddings_pretrained.py \
        --checkpoint  /path/to/tangerine/pretrained/mae_pretrained.pth \
        --dataset_dir dataset_splits \
        --images_dir  /path/to/images_3d_swine \
        --output_dir  outputs/pretrained/embeddings \
        --lrads_csv   scan_master_with_lrads_value_v3_with_base.csv \
        --metadata_csv /path/to/lungct_with_mrn_anonacc.csv \
        --split       all \
        --layer       -1
"""

import sys, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# reuse dataset + all plotting helpers from the fine-tuned script
from extract_embeddings import (
    reduce_embeddings,
    plot_lrads, plot_cancer, plot_categorical, plot_age,
    RACE_SHORTMAP,  # Import race label shortening
)
from survival_dataset import LungCancerSurvivalDataset

MAX_FOLLOWUP = 6


def load_pretrained_encoder(checkpoint_path, device):
    sys.path.insert(0, '/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/3D-MAE-MedImaging')
    import models_vit

    encoder = models_vit.vit_large_patch16_yo(
        num_classes=0, drop_path_rate=0.0, global_pool=False, img_size=256)

    ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt

    # handle encoder.* prefix (if someone passes a fine-tuned ckpt by mistake)
    if any(k.startswith('encoder.') for k in state):
        state = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
    state = {k.replace('module.', ''): v for k, v in state.items()}

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys  ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')

    return encoder.eval().to(device)


def extract_embeddings(encoder, dataset, device, batch_size=4, layer=-1):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_emb, all_pids, all_cancer, all_time = [], [], [], []
    num_layers = len(encoder.blocks)
    use_hook   = (layer != -1)
    layer_idx  = layer % num_layers if use_hook else None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'layer={layer}'):
            vols = batch['volume'].to(device)
            if use_hook:
                captured = {}
                def _hook(m, inp, out):
                    captured['cls'] = out[:, 0, :].float()
                h = encoder.blocks[layer_idx].register_forward_hook(_hook)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                final_emb = encoder.forward_features(vols)
            if use_hook:
                h.remove()
                emb = captured['cls']
            else:
                emb = final_emb
            all_emb.append(emb.cpu().float().numpy())
            all_pids.extend(batch['patient_id'])
            all_cancer.extend(batch['cancer'].numpy().tolist())
            all_time.extend(batch['time_at_event'].numpy().tolist())

    return np.concatenate(all_emb, axis=0), all_pids, all_cancer, all_time


def main(args):
    # Create subfolders: trained/, pretrain/, combined/
    # (This script runs for pretrained model, so save to 'pretrain/')
    base_output = Path(args.output_dir)
    output_dir = base_output / 'pretrain'
    combined_dir = base_output / 'combined'
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading pretrained encoder: {args.checkpoint}')
    encoder = load_pretrained_encoder(args.checkpoint, device)
    print(f'  {sum(p.numel() for p in encoder.parameters()):,} params, '
          f'{len(encoder.blocks)} blocks')

    splits   = ['train','val','test'] if args.split == 'all' else [args.split]
    all_emb, all_pids, all_cancer, all_time, all_splits = [], [], [], [], []

    for split in splits:
        print(f'Processing {split} split...')
        ds = LungCancerSurvivalDataset(
            csv_file=Path(args.dataset_dir) / f'{split}.csv',
            images_dir=args.images_dir,
            patch_size=(256, 256, 256), augment=False, mode='val')
        emb, pids, cancer, time = extract_embeddings(
            encoder, ds, device, batch_size=args.batch_size, layer=args.layer)
        all_emb.append(emb); all_pids.extend(pids)
        all_cancer.extend(cancer); all_time.extend(time)
        all_splits.extend([split] * len(pids))

    emb_arr   = np.concatenate(all_emb, axis=0)
    N         = len(all_pids)
    layer_tag = f'layer{args.layer}' if args.layer != -1 else 'layer_final'
    print(f'\nTotal: {N} × {emb_arr.shape[1]}  ({layer_tag}, pretrained)')

    np.save(output_dir / f'embeddings_{layer_tag}.npy', emb_arr)

    # metadata
    meta = pd.DataFrame({'patient_id': all_pids, 'split': all_splits,
                          'cancer': all_cancer, 'time_at_event': all_time})

    if args.lrads_csv and Path(args.lrads_csv).exists():
        sm = pd.read_csv(args.lrads_csv,
                         usecols=['ct_id','lrads_value','lrads_category_base'],
                         dtype={'ct_id': str, 'lrads_category_base': 'Int64'})
        sm['ct_id'] = sm['ct_id'].astype(str).str.strip()
        meta['patient_id'] = meta['patient_id'].astype(str).str.strip()
        meta = meta.merge(sm, left_on='patient_id', right_on='ct_id', how='left')
        meta.drop(columns=['ct_id'], inplace=True)
        print(f'Lung-RADS: {meta["lrads_category_base"].notna().sum()}/{N}')
    else:
        meta['lrads_value'] = meta['lrads_category_base'] = np.nan

    if args.metadata_csv and Path(args.metadata_csv).exists():
        avail    = pd.read_csv(args.metadata_csv, nrows=0).columns.tolist()
        use_cols = [c for c in ['ct_id','age','sex','race','smoke','ct_date'] if c in avail]
        if len(use_cols) > 1:
            clin = pd.read_csv(args.metadata_csv, usecols=use_cols, dtype={'ct_id': str})
            clin['ct_id'] = clin['ct_id'].astype(str).str.strip()
            meta = meta.merge(clin, left_on='patient_id', right_on='ct_id', how='left')
            meta.drop(columns=['ct_id'], inplace=True, errors='ignore')
    else:
        for col in ['age', 'sex', 'race', 'smoke', 'ct_date']:
            if col not in meta.columns:
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

    # UMAP
    print(f'Running UMAP...')
    xy, method_name = reduce_embeddings(emb_arr, method=args.reduction,
                                        labels=meta['lrads_category_base'].values)
    np.save(output_dir / f'umap_coords_{layer_tag}.npy', xy)

    lrads_cat  = meta['lrads_category_base']
    age_vals   = meta['age'].tolist()    if 'age'    in meta.columns else [np.nan]*N
    sex_vals   = meta['sex'].tolist()    if 'sex'    in meta.columns else [None]*N
    smoke_vals = meta['smoke'].tolist()  if 'smoke'  in meta.columns else [None]*N
    race_vals  = meta['race'].tolist()   if 'race'   in meta.columns else [None]*N
    era_vals   = meta['ct_era'].tolist() if 'ct_era' in meta.columns else [None]*N

    CT_ERA_PALETTE = {'2010–2015': '#1b7837', '2015–2020': '#762a83', '2020–2025': '#e08214'}

    # individual plots
    for fn, fname, kw in [
        (plot_lrads,  f'umap_lrads_{layer_tag}.png',  dict(lrads_cat=lrads_cat)),
        (plot_cancer, f'umap_cancer_{layer_tag}.png', dict(cancer=all_cancer)),
    ]:
        fig, ax = plt.subplots(figsize=(7,6))
        fn(ax, xy, method_name=method_name, **kw)
        fig.tight_layout(); fig.savefig(output_dir/fname, dpi=150); plt.close(fig)
        print(f'Saved {fname}')

    for title, vals, fname, palette in [
        ('Sex',            sex_vals,   f'umap_sex_{layer_tag}.png',   None),
        ('Smoking status', smoke_vals, f'umap_smoke_{layer_tag}.png', None),
        ('Race',           race_vals,  f'umap_race_{layer_tag}.png',  None),
        ('CT scan era',    era_vals,   f'umap_ctera_{layer_tag}.png', CT_ERA_PALETTE),
    ]:
        fig, ax = plt.subplots(figsize=(7,6))
        plot_categorical(ax, xy, vals, title, method_name, palette=palette)
        fig.tight_layout(); fig.savefig(output_dir/fname, dpi=150); plt.close(fig)
        print(f'Saved {fname}')

    fig, ax = plt.subplots(figsize=(7,6))
    plot_age(ax, xy, age_vals, method_name)
    fig.tight_layout(); fig.savefig(output_dir/f'umap_age_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'Saved umap_age_{layer_tag}.png')

    # combined 3×3 (no pred1 panel — pretrained has no head, axes[0,2] off)
    # row1: lrads / cancer / (off)
    # row2: sex   / smoke  / age
    # row3: race  / ct_era / (off)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    plot_lrads(      axes[0,0], xy, lrads_cat=lrads_cat, method_name=method_name)
    plot_cancer(     axes[0,1], xy, cancer=all_cancer,   method_name=method_name)
    axes[0,2].axis('off')
    plot_categorical(axes[1,0], xy, sex_vals,   'Sex',            method_name)
    plot_categorical(axes[1,1], xy, smoke_vals, 'Smoking status', method_name)
    plot_age(        axes[1,2], xy, age_vals,             method_name)
    plot_categorical(axes[2,0], xy, race_vals,  'Race',           method_name)
    plot_categorical(axes[2,1], xy, era_vals,   'CT scan era',    method_name,
                     palette=CT_ERA_PALETTE)
    axes[2,2].axis('off')
    fig.suptitle(
        f'TANGERINE pretrained (no fine-tuning) — {N} scans — {layer_tag}', fontsize=13)
    fig.tight_layout()
    # Save combined plots to shared 'combined' folder
    fig.savefig(combined_dir / f'umap_combined_{layer_tag}.png', dpi=150)
    plt.close(fig)
    print(f'Saved umap_combined_{layer_tag}.png')
    print(f'\nAll outputs in: {output_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True,
                   help='mae_pretrained.pth (raw encoder, no survival head)')
    p.add_argument('--dataset_dir',  required=True)
    p.add_argument('--images_dir',   required=True)
    p.add_argument('--output_dir',   required=True)
    p.add_argument('--lrads_csv',    default=None)
    p.add_argument('--metadata_csv', default=None)
    p.add_argument('--split',        default='all',
                   choices=['train','val','test','all'])
    p.add_argument('--reduction',    default='umap',
                   choices=['umap','umap_sup','umap_pca','tsne'])
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--layer',        type=int, default=-1,
                   help='Block index (0-23). -1 = final output.')
    main(p.parse_args())
