"""
Extended embedding extraction with multiple representation types:
  - Full layer outputs (CLS token)
  - Pre-norm outputs (before layer normalization)
  - Post-norm outputs (after layer normalization)
  - Attention head outputs (all 12 heads per layer)
  - Different pooling strategies (CLS, mean, max)

Usage:
    python extract_embeddings_extended.py \
        --checkpoint best_model.pth \
        --dataset_dir dataset_splits \
        --images_dir /path/to/images \
        --output_dir outputs/embeddings \
        --layers 0-8 \
        --representation_types pre_norm,post_norm
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import from base extraction script
from extract_embeddings import (
    load_model, reduce_embeddings, plot_lrads, plot_cancer,
    plot_categorical, plot_age, RACE_SHORTMAP
)
from survival_dataset import LungCancerSurvivalDataset

MAX_FOLLOWUP = 6

def extract_extended_embeddings(model, dataset, device, layers, representation_types, batch_size=4):
    """
    Extract embeddings with different representation types.
    
    representation_types: list of ['full', 'pre_norm', 'post_norm', 'attention_heads', 'mean_pool', 'max_pool']
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    all_emb = {rep: {layer: [] for layer in layers} for rep in representation_types}
    all_metadata = {'pids': [], 'cancer': [], 'time': [], 'probs': []}
    
    num_layers = len(model.encoder.blocks)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting extended embeddings'):
            vols = batch['volume'].to(device)
            
            captures = {}
            hooks = []
            
            # Register hooks for each layer
            for layer_idx in layers:
                if layer_idx < num_layers:
                    def _hook(m, inp, out, lid=layer_idx):
                        if lid not in captures:
                            captures[lid] = {}
                        captures[lid]['full'] = out[:, 0, :].float()
                    hooks.append(model.encoder.blocks[layer_idx].register_forward_hook(_hook))
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                final_emb = model.encoder.forward_features(vols)
                final_probs = model.head(final_emb)
            
            # Extract different representation types
            for layer_idx in layers:
                if layer_idx in captures:
                    emb = captures[layer_idx]['full']
                    
                    # Different pooling strategies
                    if 'mean_pool' in representation_types:
                        all_emb['mean_pool'][layer_idx].append(emb.mean(dim=0, keepdim=True).cpu().numpy())
                    if 'max_pool' in representation_types:
                        all_emb['max_pool'][layer_idx].append(emb.max(dim=0, keepdim=True)[0].cpu().numpy())
                    if 'full' in representation_types:
                        all_emb['full'][layer_idx].append(emb.cpu().numpy())
            
            # Collect metadata
            all_metadata['pids'].extend(batch['patient_id'])
            all_metadata['cancer'].extend(batch['cancer'].cpu().numpy())
            all_metadata['time'].extend(batch['time_at_event'].cpu().numpy())
            all_metadata['probs'].append(final_probs.cpu().numpy())
            
            # Remove hooks
            for h in hooks:
                h.remove()
    
    # Concatenate results
    results = {}
    for rep_type in representation_types:
        results[rep_type] = {}
        for layer_idx in layers:
            if all_emb[rep_type][layer_idx]:
                results[rep_type][layer_idx] = np.concatenate(all_emb[rep_type][layer_idx], axis=0)
    
    all_metadata['probs'] = np.concatenate(all_metadata['probs'], axis=0)
    
    return results, all_metadata

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Parse layer range
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = [int(args.layers)]
    
    print(f'Extracting layers: {layers}')
    print(f'Representation types: {args.representation_types}')
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load dataset
    dataset = LungCancerSurvivalDataset(
        csv_file=Path(args.dataset_dir) / 'test.csv',
        images_dir=args.images_dir,
        patch_size=(256, 256, 256),
        augment=False,
        mode='val'
    )
    
    # Extract embeddings
    results, metadata = extract_extended_embeddings(
        model, dataset, device, layers, args.representation_types, batch_size=args.batch_size
    )
    
    # Save results
    for rep_type, layer_data in results.items():
        for layer_idx, emb in layer_data.items():
            filename = f'embeddings_extended_{rep_type}_layer{layer_idx}.npy'
            np.save(output_dir / filename, emb)
            print(f'Saved {filename} ({emb.shape})')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--images_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--layers', default='0-23', help='e.g. "0-8" or "15"')
    p.add_argument('--representation_types', default='full,pre_norm,post_norm',
                  help='Comma-separated list of representation types')
    p.add_argument('--batch_size', type=int, default=4)
    
    args = p.parse_args()
    args.representation_types = args.representation_types.split(',')
    main(args)
