"""
Extract 3D attention maps from a trained TANGERINE survival model.

For each CT in the test set, saves:
  - attention_rollout.nii.gz   : rollout across all 24 layers (best overall view)
  - attention_layer{N}.nii.gz  : single-layer CLS attention (inspect any layer)
  - summary_slices.png         : axial / coronal / sagittal slices with attention overlay

Usage (after training):
    python extract_attention_maps.py \
        --checkpoint   outputs/run_XXXX/best_model.pth \
        --dataset_dir  dataset_splits \
        --images_dir   /path/to/images_3d_swine \
        --output_dir   outputs/run_XXXX/attention \
        --split        test \
        --layers       -1 12 0 \
        --rollout
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from tangerine_survival_model import TANGERINESurvivalModel
from survival_dataset import LungCancerSurvivalDataset

MAX_FOLLOWUP = 6


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    # Recover encoder_weights_path from checkpoint args if stored, else dummy
    model = TANGERINESurvivalModel.__new__(TANGERINESurvivalModel)
    # We only need the encoder + head weights, not the pretrained MAE weights
    # so build the model without loading MAE pretrained weights
    import models_vit
    from cumulative_probability_layer import CumulativeProbabilityLayer
    import torch.nn as nn
    model.__class__ = TANGERINESurvivalModel
    nn.Module.__init__(model)
    model.encoder = models_vit.vit_large_patch16_yo(
        num_classes=0, drop_path_rate=0.0, global_pool=False, img_size=256)
    model.num_layers = len(model.encoder.blocks)
    model.head = CumulativeProbabilityLayer(model.encoder.embed_dim, MAX_FOLLOWUP)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    print(f"Loaded checkpoint (epoch {ckpt['epoch']+1}  "
          f"val avg-AUC={ckpt['val_avg_auc']:.4f})")
    return model


def save_nifti(array_3d, output_path):
    """Save [D, H, W] float array as .nii.gz (no spacing metadata)."""
    img = sitk.GetImageFromArray(array_3d.astype(np.float32))
    sitk.WriteImage(img, str(output_path))


def summary_plot(ct_vol, attn_vol, patient_id, pred_scores, cancer, output_path,
                 title='Attention Rollout'):
    """
    3x3 figure: one row per anatomical plane (axial / coronal / sagittal).
    Each row shows: CT only | attention map | overlay.
    Slices are chosen at the peak attention voxel.
    """
    # Find peak attention voxel
    peak = np.unravel_index(np.argmax(attn_vol), attn_vol.shape)
    d_peak, h_peak, w_peak = peak

    planes = [
        ('Axial',     ct_vol[d_peak],   attn_vol[d_peak]),
        ('Coronal',   ct_vol[:, h_peak, :], attn_vol[:, h_peak, :]),
        ('Sagittal',  ct_vol[:, :, w_peak], attn_vol[:, :, w_peak]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    pred_str = '  '.join(f'Y{t+1}={p:.2f}' for t, p in enumerate(pred_scores))
    cancer_str = 'Cancer=YES' if cancer else 'Cancer=NO'
    fig.suptitle(f'{patient_id}  |  {cancer_str}  |  {pred_str}\n{title}',
                 fontsize=9)

    for row, (plane_name, ct_slice, attn_slice) in enumerate(planes):
        vmin, vmax = np.percentile(ct_slice, [1, 99])
        attn_norm = (attn_slice - attn_slice.min()) / (attn_slice.max() - attn_slice.min() + 1e-8)

        axes[row, 0].imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        axes[row, 0].set_title(f'{plane_name} — CT')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(attn_norm, cmap='hot', origin='lower')
        axes[row, 1].set_title(f'{plane_name} — Attention')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(ct_slice, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        heatmap_rgba = plt.cm.hot(attn_norm)
        heatmap_rgba[..., 3] = attn_norm ** 0.5 * 0.8  # alpha ∝ attention; transparent where low
        axes[row, 2].imshow(heatmap_rgba, origin='lower')
        axes[row, 2].set_title(f'{plane_name} — Overlay')
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def layer_attention_grid(model, volume_tensor, patient_id, output_path):
    """
    4x6 grid showing CLS attention from each of the 24 layers,
    displayed as a single representative axial slice (peak attention depth).
    Useful for understanding at which depth the model focuses.
    """
    result = model.extract_attention_rollout(volume_tensor)
    layer_attns = result['layer_attns']  # list of 24 [1, 16, 16, 16]

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle(f'{patient_id} — CLS attention per layer (axial slice at peak)',
                 fontsize=10)

    for i, la in enumerate(layer_attns):
        row, col = divmod(i, 6)
        ax = axes[row, col]
        la_np = la[0].numpy()  # [16, 16, 16]

        # pick the axial slice (depth axis) with highest total attention
        depth_attn = la_np.sum(axis=(1, 2))
        best_d = int(np.argmax(depth_attn))
        slice_2d = la_np[best_d]  # [16, 16]

        ax.imshow(slice_2d, cmap='hot', origin='lower')
        ax.set_title(f'L{i}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir  = Path(args.output_dir)
    by_ct_dir = out_dir / 'by_ct'
    by_ct_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Load split CSV
    dataset_path = Path(args.dataset_dir)
    csv_file = dataset_path / f'{args.split}.csv'
    dataset  = LungCancerSurvivalDataset(
        csv_file=csv_file,
        images_dir=args.images_dir,
        patch_size=(256, 256, 256),
        augment=False,
        mode='test',
    )

    print(f"\nProcessing {args.split} split ({len(dataset)} total samples)...")
    print(f"Output: {out_dir}")

    # Layers to extract single-layer attention for
    layers = [int(l) % model.num_layers for l in args.layers] if args.layers else [-1]

    summary_rows = []

    # Build index list — optionally filter to cancer-only patients
    all_indices = list(range(len(dataset)))
    if args.cancer_only:
        cancer_indices = [i for i in all_indices
                          if int(dataset[i]['cancer'].item()) == 1]
        print(f"  cancer_only=True: {len(cancer_indices)}/{len(dataset)} patients are cancer cases")
        indices = cancer_indices
    else:
        indices = all_indices
    if args.max_patients:
        indices = indices[:args.max_patients]

    for idx in tqdm(indices):
        sample     = dataset[idx]
        patient_id = sample['patient_id']
        cancer     = int(sample['cancer'].item())
        volume_t   = sample['volume'].unsqueeze(0).to(device)   # [1, 1, D, H, W]
        ct_np      = sample['volume'][0].numpy()                 # [D, H, W]

        pt_out = by_ct_dir / patient_id
        pt_out.mkdir(exist_ok=True)

        # Run model to get survival predictions
        with torch.no_grad():
            logits = model(volume_t)
        pred_scores = torch.sigmoid(logits)[0].cpu().numpy()    # [6]

        # ── Attention rollout (all 24 layers) ─────────────────────────
        if args.rollout:
            rollout = model.extract_attention_rollout(
                volume_t, discard_ratio=args.discard_ratio)
            attn_vol = rollout['volume'][0].cpu().numpy()        # [D, H, W]

            save_nifti(attn_vol, pt_out / 'attention_rollout.nii.gz')
            summary_plot(ct_np, attn_vol, patient_id, pred_scores, cancer,
                         pt_out / 'summary_rollout.png', title='Attention Rollout (all layers)')

            # Also save the per-layer grid
            layer_attention_grid(model, volume_t, patient_id,
                                 pt_out / 'attention_per_layer.png')

        # ── Single-layer attention ─────────────────────────────────────
        for li in layers:
            li_resolved = li % model.num_layers
            result = model.extract_attention(volume_t, layer_idx=li_resolved)
            attn_vol_l = result['volume'][0].cpu().numpy()       # [D, H, W]

            save_nifti(attn_vol_l, pt_out / f'attention_layer{li_resolved:02d}.nii.gz')
            summary_plot(ct_np, attn_vol_l, patient_id, pred_scores, cancer,
                         pt_out / f'summary_layer{li_resolved:02d}.png',
                         title=f'CLS Attention Layer {li_resolved}')

        summary_rows.append({
            'patient_id':   patient_id,
            'cancer':       cancer,
            **{f'pred_{t+1}': float(pred_scores[t]) for t in range(MAX_FOLLOWUP)},
            'peak_attn_d':  int(np.unravel_index(
                                np.argmax(rollout['volume'][0].cpu().numpy() if args.rollout
                                          else result['volume'][0].cpu().numpy()),
                                ct_np.shape)[0]) if (args.rollout or layers) else None,
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / 'attention_summary.csv', index=False)
    print(f"\nDone. Results saved to: {out_dir}")
    print(f"  Per patient: {by_ct_dir}/{{patient_id}}/attention_rollout.nii.gz + PNGs")
    print(f"  attention_summary.csv — predictions + peak attention slice per patient")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True,
                   help='Path to best_model.pth')
    p.add_argument('--dataset_dir',   required=True,
                   help='Directory with train/val/test.csv survival splits')
    p.add_argument('--images_dir',    required=True,
                   help='Directory with .nii.gz CT volumes')
    p.add_argument('--output_dir',    required=True,
                   help='Where to save attention maps')
    p.add_argument('--split',         default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--layers',        type=int, nargs='+', default=[-1],
                   help='Layer indices for single-layer attention (default: last layer). '
                        'E.g. --layers -1 12 0  extracts last, middle, first.')
    p.add_argument('--rollout',       action='store_true', default=True,
                   help='Compute attention rollout across all layers (default: True)')
    p.add_argument('--discard_ratio', type=float, default=0.0,
                   help='Fraction of lowest-attention patches to discard before rollout '
                        '(0.9 = keep only top 10%%, makes maps sharper). Default: 0.')
    p.add_argument('--max_patients',  type=int, default=None,
                   help='Limit to N patients (for quick testing)')
    p.add_argument('--cancer_only',   action='store_true', default=False,
                   help='Only extract attention maps for cancer=1 patients')
    main(p.parse_args())

