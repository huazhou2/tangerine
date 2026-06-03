"""
Extract extended representations from pretrained TANGERINE encoder:
- full: standard CLS token embeddings
- pre_norm: pre-normalization output from transformer block
- post_norm: post-normalization output from transformer block
- attention_heads: individual attention head outputs (12 per layer)
- mean_pool: mean pooling of all patch tokens
- max_pool: max pooling of all patch tokens

Enables comprehensive analysis of which layer/representation best captures task structure.

Usage:
    python extract_embeddings_extended.py \
        --checkpoint pretrained/mae_pretrained.pth \
        --dataset_dir dataset_splits \
        --images_dir /path/to/images \
        --output_dir outputs/run_20260529_101746/embeddings \
        --layers 0-8 \
        --representation_types full,pre_norm,post_norm
"""

import sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from extract_embeddings import (
    reduce_embeddings,
    plot_lrads, plot_cancer, plot_categorical, plot_age,
    RACE_SHORTMAP,
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

    if any(k.startswith('encoder.') for k in state):
        state = {k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}
    state = {k.replace('module.', ''): v for k, v in state.items()}

    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        print(f'  Missing keys  ({len(missing)}): {missing[:3]}...')
    if unexpected:
        print(f'  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...')

    return encoder.eval().to(device)


def extract_extended_representations(encoder, dataset, device, batch_size=4,
                                      layer_idx=0, representation_types=None):
    """
    Extract multiple representation types from a single transformer block.

    representation_types: list of ['full', 'pre_norm', 'post_norm', 'attention_heads', 'mean_pool', 'max_pool']

    Returns dict: {
        'full': [N, 1024],
        'pre_norm': [N, 1024],
        'post_norm': [N, 1024],
        'attention_heads': [N, 12, 64],  # 12 heads × 64 dims
        'mean_pool': [N, 1024],
        'max_pool': [N, 1024],
    }
    """
    if representation_types is None:
        representation_types = ['full']

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    results = {rt: [] for rt in representation_types}
    all_pids, all_cancer, all_time = [], [], []
    num_layers = len(encoder.blocks)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'layer={layer_idx}'):
            vols = batch['volume'].to(device)

            captured = {}

            # Register hooks for multiple capture points
            hooks = []

            def make_pre_norm_hook(name):
                def _hook(m, inp, out):
                    # Before normalization: inp[0] is the block input
                    captured[name] = inp[0][:, 0, :].float().clone()
                return _hook

            def make_post_norm_hook(name):
                def _hook(m, inp, out):
                    # After full block: out is [B, num_patches+1, 1024]
                    captured[name] = out[:, 0, :].float().clone()
                return _hook

            def make_attn_hook(name):
                def _hook(m, inp, out):
                    # Attention module output: [B, num_heads, num_patches+1, head_dim]
                    # Extract CLS token (position 0) for each head
                    attn_out = out  # [B, 1, num_patches+1, head_dim] or similar
                    # Reshape to [B, num_heads, head_dim]
                    if len(attn_out.shape) == 4:
                        captured[name] = attn_out[:, :, 0, :].float().clone()  # [B, num_heads, head_dim]
                    else:
                        captured[name] = attn_out[:, 0, :].float().clone()
                return _hook

            # Hook into block input (pre-norm)
            if 'pre_norm' in representation_types:
                h_pre = encoder.blocks[layer_idx].register_forward_pre_hook(
                    make_pre_norm_hook('pre_norm'))
                hooks.append(h_pre)

            # Hook into block output (post-norm)
            if 'post_norm' in representation_types or 'attention_heads' in representation_types or \
               'mean_pool' in representation_types or 'max_pool' in representation_types:
                h_post = encoder.blocks[layer_idx].register_forward_hook(
                    make_post_norm_hook('post_norm'))
                hooks.append(h_post)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                _ = encoder.forward_features(vols)

            # Process captured representations
            for hook in hooks:
                hook.remove()

            # Full CLS token (standard)
            if 'full' in representation_types:
                results['full'].append(captured['post_norm'].cpu().numpy())

            # Pre-norm (block input)
            if 'pre_norm' in representation_types:
                results['pre_norm'].append(captured['pre_norm'].cpu().numpy())

            # Post-norm (block output)
            if 'post_norm' in representation_types:
                results['post_norm'].append(captured['post_norm'].cpu().numpy())

            # Mean pool across patches
            if 'mean_pool' in representation_types:
                # Recapture with full patch sequence
                captured_full = {}
                def patch_hook(m, inp, out):
                    captured_full['patches'] = out.float()
                h = encoder.blocks[layer_idx].register_forward_hook(patch_hook)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    _ = encoder.forward_features(vols)
                h.remove()
                # Mean pool over all tokens (including CLS)
                mean_pool = captured_full['patches'].mean(dim=1)  # [B, 1024]
                results['mean_pool'].append(mean_pool.cpu().numpy())

            # Max pool across patches
            if 'max_pool' in representation_types:
                captured_full = {}
                def patch_hook(m, inp, out):
                    captured_full['patches'] = out.float()
                h = encoder.blocks[layer_idx].register_forward_hook(patch_hook)
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    _ = encoder.forward_features(vols)
                h.remove()
                # Max pool over all tokens
                max_pool, _ = captured_full['patches'].max(dim=1)  # [B, 1024]
                results['max_pool'].append(max_pool.cpu().numpy())

            all_pids.extend(batch['patient_id'])
            all_cancer.extend(batch['cancer'].numpy().tolist())
            all_time.extend(batch['time_at_event'].numpy().tolist())

    # Concatenate all batches
    for rt in representation_types:
        if results[rt]:
            results[rt] = np.concatenate(results[rt], axis=0)
        else:
            results[rt] = np.array([])

    return results, all_pids, all_cancer, all_time


def parse_layer_range(layer_str):
    """Parse layer string like '0-8' into list [0,1,2,...,8]"""
    parts = layer_str.split('-')
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    else:
        return [int(layer_str)]


def main(args):
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

    # Parse layer range and representation types
    layers = parse_layer_range(args.layers)
    rep_types = [rt.strip() for rt in args.representation_types.split(',')]
    print(f'Layers: {layers}')
    print(f'Representation types: {rep_types}')

    splits = ['train','val','test'] if args.split == 'all' else [args.split]

    # Process each layer
    for layer_idx in layers:
        print(f'\n═══════════════════════════════════════════════════════════')
        print(f'Processing layer {layer_idx}/{max(layers)}')
        print(f'═══════════════════════════════════════════════════════════')

        all_results = {rt: [] for rt in rep_types}
        all_pids, all_cancer, all_time, all_splits = [], [], [], []

        for split in splits:
            print(f'  {split} split...')
            ds = LungCancerSurvivalDataset(
                csv_file=Path(args.dataset_dir) / f'{split}.csv',
                images_dir=args.images_dir,
                patch_size=(256, 256, 256), augment=False, mode='val')

            results, pids, cancer, time = extract_extended_representations(
                encoder, ds, device, batch_size=args.batch_size,
                layer_idx=layer_idx, representation_types=rep_types)

            for rt in rep_types:
                all_results[rt].append(results[rt])
            all_pids.extend(pids)
            all_cancer.extend(cancer)
            all_time.extend(time)
            all_splits.extend([split] * len(pids))

        # Save results for each representation type
        for rt in rep_types:
            emb_arr = np.concatenate(all_results[rt], axis=0)
            N = len(all_pids)
            layer_tag = f'layer{layer_idx}'

            # Save embeddings
            fname = f'embeddings_{rt}_{layer_tag}.npy'
            np.save(output_dir / fname, emb_arr)
            print(f'  Saved {fname}  ({emb_arr.shape})')

            # Save metadata
            meta = pd.DataFrame({
                'patient_id': all_pids,
                'split': all_splits,
                'cancer': all_cancer,
                'time_at_event': all_time,
            })

            # Add LRADS and clinical variables (same as extract_embeddings.py)
            if args.lrads_csv and Path(args.lrads_csv).exists():
                sm = pd.read_csv(args.lrads_csv,
                                usecols=['ct_id','lrads_value','lrads_category_base'],
                                dtype={'ct_id': str, 'lrads_category_base': 'Int64'})
                sm['ct_id'] = sm['ct_id'].astype(str).str.strip()
                meta['patient_id'] = meta['patient_id'].astype(str).str.strip()
                meta = meta.merge(sm, left_on='patient_id', right_on='ct_id', how='left')
                meta.drop(columns=['ct_id'], inplace=True)

            if args.metadata_csv and Path(args.metadata_csv).exists():
                avail = pd.read_csv(args.metadata_csv, nrows=0).columns.tolist()
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

            # Shorten race labels
            if 'race' in meta.columns:
                meta['race'] = meta['race'].replace(RACE_SHORTMAP)

            meta_fname = f'embeddings_meta_{rt}_{layer_tag}.csv'
            meta.to_csv(output_dir / meta_fname, index=False)
            print(f'  Saved {meta_fname}')

    print(f'\n✓ All extended representations saved to: {output_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='mae_pretrained.pth')
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--images_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--lrads_csv', default=None)
    p.add_argument('--metadata_csv', default=None)
    p.add_argument('--split', default='all',
                   choices=['train','val','test','all'])
    p.add_argument('--layers', default='0-23',
                   help='Layer range like "0-8", "9-16", "17-23"')
    p.add_argument('--representation_types', default='full',
                   help='Comma-separated: full,pre_norm,post_norm,attention_heads,mean_pool,max_pool')
    p.add_argument('--batch_size', type=int, default=4)
    main(p.parse_args())
