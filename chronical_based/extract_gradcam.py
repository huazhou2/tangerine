"""
Grad-CAM for TANGERINE 6-year lung cancer survival model.

Matches the paper's description:
  "hooks were introduced to capture the deep activations and gradients
   within the final transformer block. The activation maps are weighted
   by the gradient signals corresponding to the target class, creating
   a spatial heatmap."

Method:
  1. Forward pre-hook on model.encoder.blocks[-1] → captures INPUT tokens [B, N+1, D]
     (detached + requires_grad=True so backward computes grad w.r.t. these tokens)
  2. Backward hook on same layer → grad_input [B, N+1, D] (gradient w.r.t. block input)
  3. Weight input activations by mean gradient over feature dim: alpha = mean(grad, dim=-1)
  4. Grad-CAM map = ReLU( sum_d(alpha * activations) ) over patch tokens only [B, N]
  5. Reshape [B, N] → [B, 16, 16, 16]  (16 = 256//16 grid per axis)
  6. Trilinear upsample → [B, 256, 256, 256]
  7. Normalise per-patient to [0, 1]
  8. Save as .nii.gz and axial/coronal/sagittal PNG overlay

Note: forward_features returns only the CLS token [B, D], so patch token gradients
at the block OUTPUT are zero. By hooking the block INPUT instead and making it a
grad-enabled leaf, we capture the non-zero gradient that flows backward through
the attention mechanism (CLS attends to all patches in self-attention).

The target for backprop is the mean over all 6 yearly logits (overall risk),
or a specific year (--target_year 1..6).

Usage:
    python extract_gradcam.py \\
        --checkpoint   outputs/run_XXXX/best_model.pth \\
        --dataset_dir  dataset_splits \\
        --images_dir   /path/to/images_3d_swine \\
        --output_dir   outputs/run_XXXX/gradcam \\
        --split        test \\
        --cancer_only \\
        --max_patients 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/3D-MAE-MedImaging')

import models_vit
from cumulative_probability_layer import CumulativeProbabilityLayer
from survival_dataset import LungCancerSurvivalDataset

_PATCH_SIZE  = 16
_GRID_SIZE   = 256 // _PATCH_SIZE   # 16 per axis
_NUM_PATCHES = _GRID_SIZE ** 3       # 4096
MAX_FOLLOWUP = 6


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    from tangerine_survival_model import TANGERINESurvivalModel

    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    model = TANGERINESurvivalModel.__new__(TANGERINESurvivalModel)
    nn.Module.__init__(model)
    model.encoder = models_vit.vit_large_patch16_yo(
        num_classes=0, drop_path_rate=0.0, global_pool=False, img_size=256)
    model.num_layers = len(model.encoder.blocks)
    model.head = CumulativeProbabilityLayer(model.encoder.embed_dim, MAX_FOLLOWUP)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    epoch   = ckpt.get('epoch', '?')
    val_auc = ckpt.get('val_auc', float('nan'))
    print(f"Loaded checkpoint (epoch {epoch}  val_AUC={val_auc:.4f})")
    return model


# ── Grad-CAM hooks ─────────────────────────────────────────────────────────────

class GradCAMHooks:
    """
    Hooks the INPUT of the final transformer block (not the output).

    forward_features returns only the CLS token [B, D], so grad_output at patch
    positions is zero. By making the block's input a grad-enabled leaf tensor,
    grad_input[0] receives non-zero gradients: during backward, the CLS output
    gradient flows through self-attention back to all patch input tokens.
    """
    def __init__(self, model):
        self.activations = None   # input tokens [B, N+1, D]
        self.gradients   = None   # grad w.r.t. input tokens [B, N+1, D]
        target_layer     = model.encoder.blocks[-1]

        self._pre_hook = target_layer.register_forward_pre_hook(self._save_input)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_input(self, module, input):
        # Detach from upstream graph and enable grad so backward computes grad_input
        inp = input[0].detach().requires_grad_(True)
        self.activations = inp
        return (inp,) + input[1:]   # replace block's input with grad-enabled version

    def _save_gradients(self, module, grad_input, grad_output):
        if grad_input[0] is not None:
            self.gradients = grad_input[0]

    def remove(self):
        self._pre_hook.remove()
        self._bwd_hook.remove()


# ── Grad-CAM computation ───────────────────────────────────────────────────────

def compute_gradcam(model, volume_tensor, target_year=None):
    """
    Run one forward+backward pass and return Grad-CAM map [D, H, W] at 256³.

    Args:
        model:         TANGERINESurvivalModel (eval mode)
        volume_tensor: [1, 1, 256, 256, 256] float tensor on model's device
        target_year:   1..6 → backprop on that year's logit;
                       None → backprop on mean of all 6 logits (overall risk)

    Returns:
        gradcam_np:    numpy float32 [256, 256, 256] in [0, 1]
        pred_probs:    list of 6 floats (cancer probability per year, after sigmoid)
    """
    hooks = GradCAMHooks(model)
    model.zero_grad()

    with torch.enable_grad():
        volume_tensor = volume_tensor.requires_grad_(False)

        # Encode: ViT forward_features → CLS token [1, 1024]
        # Pre-hook intercepts blocks[-1] input and makes it a grad-enabled leaf
        features = model.encoder.forward_features(volume_tensor)  # [1, 1024]

        # Survival head → [1, 6] logits
        logits = model.head(features)

        pred_probs = torch.sigmoid(logits)[0].detach().cpu().tolist()

        # Backprop target
        if target_year is not None:
            score = logits[0, target_year - 1]
        else:
            score = logits[0].mean()   # overall 6-year risk

        score.backward()

    # ── Retrieve input activations and input gradients ─────────────────────────
    activations = hooks.activations  # [1, N+1, D] — input to blocks[-1]
    gradients   = hooks.gradients    # [1, N+1, D] — grad w.r.t. block input
    hooks.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Hooks did not capture activations/gradients. "
                           "Check that forward_features passes through blocks[-1].")

    # ── Compute Grad-CAM ──────────────────────────────────────────────────────
    # Drop CLS token (index 0), keep patch tokens [1, N, D]
    act  = activations[:, 1:, :].detach()   # [1, N, D] — input patch features
    grad = gradients[:, 1:, :].detach()     # [1, N, D] — grad w.r.t. input patches

    # Global-average-pool gradients over feature dimension → importance weights
    alpha = grad.mean(dim=-1)               # [1, N]

    # Weighted combination of activations
    cam = (alpha.unsqueeze(-1) * act).sum(dim=-1)  # [1, N]

    # ReLU: keep only positive contributions
    cam = F.relu(cam)                       # [1, N]

    # Reshape to 3D grid [1, 1, G, G, G]
    cam_3d = cam.reshape(1, 1, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE).float()

    # Upsample to volume resolution
    cam_upsampled = F.interpolate(cam_3d, size=(256, 256, 256),
                                  mode='trilinear', align_corners=False)
    cam_np = cam_upsampled[0, 0].cpu().numpy()   # [256, 256, 256]

    # Normalise to [0, 1]
    vmin, vmax = cam_np.min(), cam_np.max()
    if vmax > vmin:
        cam_np = (cam_np - vmin) / (vmax - vmin)
    else:
        cam_np = np.zeros_like(cam_np)

    return cam_np.astype(np.float32), pred_probs


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_overlay_png(volume_np, cam_np, out_path, patient_id, cancer, pred_probs):
    """
    Save axial / coronal / sagittal mid-slice overlays.
    volume_np, cam_np: [D, H, W] float arrays
    """
    d, h, w = volume_np.shape
    slices = {
        'Axial':    (volume_np[d//2], cam_np[d//2]),
        'Coronal':  (volume_np[:, h//2, :], cam_np[:, h//2, :]),
        'Sagittal': (volume_np[:, :, w//2], cam_np[:, :, w//2]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    status = 'Cancer' if cancer else 'No Cancer'
    yr1_prob = pred_probs[0] if pred_probs else float('nan')
    fig.suptitle(f'{patient_id}  |  {status}  |  1yr={yr1_prob:.3f}  6yr={pred_probs[-1]:.3f}',
                 fontsize=12)

    for col, (view, (vol_sl, cam_sl)) in enumerate(slices.items()):
        axes[0, col].imshow(vol_sl, cmap='gray', aspect='auto')
        axes[0, col].set_title(view)
        axes[0, col].axis('off')

        axes[1, col].imshow(vol_sl, cmap='gray', aspect='auto')
        axes[1, col].imshow(cam_sl, cmap='jet', alpha=0.4,
                            vmin=0, vmax=1, aspect='auto')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('CT', fontsize=10)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDevice: {device}")
    model = load_model(args.checkpoint, device)

    ds = LungCancerSurvivalDataset(
        csv_file   = Path(args.dataset_dir) / f'{args.split}.csv',
        images_dir = args.images_dir,
        patch_size = (256, 256, 256),
        augment    = False,
        mode       = 'test'
    )

    records    = []
    n_done     = 0
    patch_size = (256, 256, 256)

    for idx in tqdm(range(len(ds)), desc='Grad-CAM'):
        if args.max_patients and n_done >= args.max_patients:
            break

        row        = ds.df.iloc[idx]
        cancer     = int(row['cancer'])
        patient_id = str(row[ds.id_col])

        if args.cancer_only and cancer == 0:
            continue

        # ── Load volume ───────────────────────────────────────────────────────
        img_path = Path(args.images_dir) / row['image_filename']
        if not img_path.exists():
            print(f"  [SKIP] {patient_id}: image not found")
            continue

        vol_sitk = sitk.ReadImage(str(img_path))
        vol_np   = sitk.GetArrayFromImage(vol_sitk).astype(np.float32)
        while vol_np.ndim > 3:
            vol_np = vol_np.squeeze()

        # Center crop to 256³ (same as dataset loader)
        d, h, w    = vol_np.shape
        pd, ph, pw = patch_size
        if d < pd or h < ph or w < pw:
            vol_np = np.pad(vol_np,
                            ((0, max(0, pd-d)), (0, max(0, ph-h)), (0, max(0, pw-w))),
                            mode='constant', constant_values=vol_np.min())
            d, h, w = vol_np.shape
        ds_ = (d - pd) // 2
        hs_ = (h - ph) // 2
        ws_ = (w - pw) // 2
        vol_np = vol_np[ds_:ds_+pd, hs_:hs_+ph, ws_:ws_+pw]

        vol_tensor = torch.from_numpy(vol_np[np.newaxis, np.newaxis]).to(device)

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        try:
            cam_np, pred_probs = compute_gradcam(model, vol_tensor,
                                                  target_year=args.target_year)
        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}")
            continue

        # ── Save per-patient outputs ──────────────────────────────────────────
        pt_dir = out_dir / 'by_ct' / patient_id
        pt_dir.mkdir(parents=True, exist_ok=True)

        # Grad-CAM volume as NIfTI (1mm isotropic — cam is always 256³ after upsample)
        cam_sitk = sitk.GetImageFromArray(cam_np)
        cam_sitk.SetSpacing([1.0, 1.0, 1.0])
        sitk.WriteImage(cam_sitk, str(pt_dir / 'gradcam.nii.gz'))

        # PNG overlay
        save_overlay_png(vol_np, cam_np, pt_dir / 'gradcam_overlay.png',
                         patient_id, cancer, pred_probs)

        rec = {
            'patient_id': patient_id,
            'cancer':     cancer,
            'cam_max':    float(cam_np.max()),
            'cam_mean':   float(cam_np.mean()),
        }
        for yr in range(1, MAX_FOLLOWUP + 1):
            rec[f'pred_yr{yr}'] = round(pred_probs[yr - 1], 4)
        records.append(rec)
        n_done += 1

    summary_csv = out_dir / 'gradcam_summary.csv'
    pd.DataFrame(records).to_csv(summary_csv, index=False)
    print(f"\nDone. Processed {n_done} patients.")
    print(f"Summary: {summary_csv}")
    print(f"Per-patient outputs: {out_dir}/by_ct/<patient_id>/")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Grad-CAM for TANGERINE 6-year survival')
    p.add_argument('--checkpoint',   required=True,
                   help='best_model.pth from training run')
    p.add_argument('--dataset_dir',  required=True,
                   help='Directory with train/val/test.csv')
    p.add_argument('--images_dir',   required=True,
                   help='Preprocessed .nii.gz volumes')
    p.add_argument('--output_dir',   required=True,
                   help='Where to save Grad-CAM outputs')
    p.add_argument('--split',        default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--cancer_only',  action='store_true',
                   help='Only process cancer-positive patients')
    p.add_argument('--max_patients', type=int, default=None,
                   help='Cap number of patients (None = all)')
    p.add_argument('--target_year',  type=int, default=None,
                   choices=[1, 2, 3, 4, 5, 6],
                   help='Backprop on this year logit (default: mean of all 6)')
    main(p.parse_args())
