"""
TANGERINE + Sybil-style Survival Head for 6-year lung cancer risk prediction.

Encoder: ViT-Large pretrained via MAE on 98k chest CTs (TANGERINE).
Head:    CumulativeProbabilityLayer -> 6 yearly risk logits.

Attention extraction
--------------------
TANGERINE is a ViT, so every transformer block computes multi-head self-attention
over the sequence of 16^3 patch tokens plus one CLS token.

  Input 256^3  ->  patch_size=16  ->  grid 16x16x16 = 4096 patch tokens
  Attention shape per block: [B, n_heads, 4097, 4097]  (4097 = 4096 patches + 1 CLS)

Two methods are provided:

  extract_attention(x, layer_idx=-1)
      CLS-to-patch attention from one specific layer (default: last block).
      Fast, good for quick inspection.

  extract_attention_rollout(x)
      Attention rollout across ALL 24 layers (Abnar & Zuidema 2020).
      Accounts for how information propagates through the network.
      Better represents what the final CLS token "sees".

Both return a dict with:
  'patch_grid'   : [B, 16, 16, 16]  attention per patch (depth x height x width)
  'volume'       : [B, D, H, W]     trilinearly upsampled to full CT size
  'layer_idx'    : which layer(s) were used
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# TANGERINE model code — cloned from github.com/niccolo246/3D-MAE-MedImaging
# Use relative path to tangerine model (works both locally and on cluster)
tangerine_path = Path(__file__).parent.parent / 'tangerine' / '3D-MAE-MedImaging'
sys.path.insert(0, str(tangerine_path))

import models_vit
from cumulative_probability_layer import CumulativeProbabilityLayer

# ViT-Large patch grid for 256^3 input with patch_size=16
_PATCH_SIZE  = 16
_GRID_SIZE   = 256 // _PATCH_SIZE   # 16 per axis
_NUM_PATCHES = _GRID_SIZE ** 3       # 4096


class TANGERINESurvivalModel(nn.Module):

    def __init__(
        self,
        encoder_weights_path: str,
        max_followup: int = 6,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # ── Build ViT-Large encoder (no classification head) ──────────────
        # global_pool=False -> forward_features returns CLS token [B, 1024]
        self.encoder = models_vit.vit_large_patch16_yo(
            num_classes=0,
            drop_path_rate=0.0,
            global_pool=False,
            img_size=256,
        )
        encoder_dim = self.encoder.embed_dim  # 1024
        self.num_layers = len(self.encoder.blocks)  # 24 for ViT-Large

        # ── Load pretrained MAE weights ───────────────────────────────────
        print(f"\n{'='*60}")
        print(f"LOADING TANGERINE PRETRAINED ENCODER")
        print(f"{'='*60}")
        checkpoint = torch.load(encoder_weights_path, map_location='cpu')
        ckpt = checkpoint.get('model', checkpoint.get('model_state', checkpoint))

        model_dict = self.encoder.state_dict()
        filtered   = {k: v for k, v in ckpt.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
        msg = self.encoder.load_state_dict(filtered, strict=False)
        print(f"  Loaded {len(filtered)}/{len(ckpt)} keys")
        print(f"  Missing : {len(msg.missing_keys)}")
        print(f"  Unexpected: {len(msg.unexpected_keys)}")
        print(f"  ViT-Large: {self.num_layers} transformer blocks")
        print(f"  Patch grid: {_GRID_SIZE}x{_GRID_SIZE}x{_GRID_SIZE} = {_NUM_PATCHES} patches")

        if freeze_encoder:
            self._freeze_encoder()
            print(f"  Encoder frozen for warmup")
        print(f"{'='*60}\n")

        # ── Survival head ─────────────────────────────────────────────────
        self.head = CumulativeProbabilityLayer(encoder_dim, max_followup)
        print(f"Survival head: Linear({encoder_dim}, {max_followup}) cumulative")

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)  # [B, 1024]
        return self.head(features)                    # [B, 6]

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("  Encoder unfrozen")

    # ── Attention extraction ───────────────────────────────────────────

    def _register_attn_hooks(self):
        """
        Register forward hooks to capture attention weight tensors [B, n_heads, N, N].

        Two strategies tried in order:
          1. Hook blk.attn.attn_drop  — standard timm path (requires fused_attn=False)
          2. Monkey-patch blk.attn.forward — fallback for 3D-MAE custom Attention that
             has no attn_drop submodule or skips it.

        Returns (storage_list, hook_list, patch_restores_list).
        """
        # Disable fused attention so standard code path is used
        _orig_fused = {}
        for i, blk in enumerate(self.encoder.blocks):
            if hasattr(blk.attn, 'fused_attn'):
                _orig_fused[i] = blk.attn.fused_attn
                blk.attn.fused_attn = False

        storage = []
        hooks   = []
        patch_restores = []

        # Strategy 1: hook attn_drop if it exists on every block
        has_attn_drop = all(hasattr(blk.attn, 'attn_drop') for blk in self.encoder.blocks)
        if has_attn_drop:
            for blk in self.encoder.blocks:
                def _hook(module, inp, out, _s=storage):
                    _s.append(out.detach().cpu())
                hooks.append(blk.attn.attn_drop.register_forward_hook(_hook))
        else:
            # Strategy 2: monkey-patch each block's attn.forward to intercept attn weights
            for blk in self.encoder.blocks:
                orig_fwd = blk.attn.forward

                def _patched_fwd(x, _orig=orig_fwd, _s=storage, _attn=blk.attn):
                    # Manually compute attention so we can save the weight matrix
                    B, N, C = x.shape
                    qkv = _attn.qkv(x)
                    qkv = qkv.reshape(B, N, 3, _attn.num_heads, C // _attn.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale = q.shape[-1] ** -0.5
                    attn_w = (q * scale) @ k.transpose(-2, -1)
                    attn_w = attn_w.softmax(dim=-1)
                    _s.append(attn_w.detach().cpu())
                    x_out = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
                    x_out = _attn.proj(x_out)
                    if hasattr(_attn, 'proj_drop'):
                        x_out = _attn.proj_drop(x_out)
                    return x_out

                blk.attn.forward = _patched_fwd
                patch_restores.append((blk.attn, orig_fwd))

        self._orig_fused_attn = _orig_fused
        self._patch_restores  = patch_restores
        return storage, hooks

    def _restore_fused_attn(self):
        for i, blk in enumerate(self.encoder.blocks):
            if i in self._orig_fused_attn:
                blk.attn.fused_attn = self._orig_fused_attn[i]
        self._orig_fused_attn = {}
        # Restore monkey-patched forwards
        for attn_module, orig_fwd in getattr(self, '_patch_restores', []):
            attn_module.forward = orig_fwd
        self._patch_restores = []

    def extract_attention(self, x: torch.Tensor, layer_idx: int = -1):
        """
        CLS-to-patch attention from a single transformer block.

        Parameters
        ----------
        x         : [B, 1, D, H, W]  input CT volume
        layer_idx : which block to use; -1 = last block (block 23 of 24)
                    valid range: 0 .. num_layers-1

        Returns
        -------
        dict with
          'patch_grid' : [B, 16, 16, 16]  mean-head CLS attention per patch
          'volume'     : [B, D, H, W]     upsampled to full CT size
          'layer_idx'  : resolved integer index
          'per_head'   : [B, n_heads, 16, 16, 16]  per-head maps (for inspection)
        """
        layer_idx = layer_idx % self.num_layers  # handle -1 etc.

        storage, hooks = self._register_attn_hooks()
        self.eval()
        with torch.no_grad():
            self.encoder.forward_features(x)
        for h in hooks:
            h.remove()
        self._restore_fused_attn()

        # storage[layer_idx]: [B, n_heads, N+1, N+1]  (N+1 = 4097)
        attn = storage[layer_idx]                           # [B, H, 4097, 4097]
        cls_attn = attn[:, :, 0, 1:]                       # [B, H, 4096] CLS -> patches
        cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)

        B, n_heads, _ = cls_attn.shape
        g = _GRID_SIZE

        # per-head map
        per_head = cls_attn.reshape(B, n_heads, g, g, g)   # [B, H, 16, 16, 16]

        # mean over heads
        patch_grid = per_head.mean(dim=1)                   # [B, 16, 16, 16]

        # upsample to full CT size [B, D, H, W]
        D, H, W  = x.shape[2], x.shape[3], x.shape[4]
        volume   = F.interpolate(
            patch_grid.unsqueeze(1).float(),
            size=(D, H, W), mode='trilinear', align_corners=False
        ).squeeze(1)                                        # [B, D, H, W]

        return {
            'patch_grid': patch_grid,
            'volume':     volume,
            'layer_idx':  layer_idx,
            'per_head':   per_head,
        }

    def extract_attention_rollout(self, x: torch.Tensor, discard_ratio: float = 0.0):
        """
        Attention rollout across all transformer blocks.

        Propagates attention through the residual stream following
        Abnar & Zuidema (2020): at each layer, A_eff = 0.5*A + 0.5*I,
        then rollout = A_eff[L] @ A_eff[L-1] @ ... @ A_eff[0].

        Parameters
        ----------
        x             : [B, 1, D, H, W]
        discard_ratio : fraction of lowest-attention patches to zero out
                        before rollout (reduces noise), default 0 = keep all

        Returns
        -------
        dict with
          'patch_grid'    : [B, 16, 16, 16]  rollout CLS attention per patch
          'volume'        : [B, D, H, W]     upsampled to full CT size
          'layer_attns'   : list of 24 [B, 16, 16, 16] per-layer mean-head maps
        """
        storage, hooks = self._register_attn_hooks()
        self.eval()
        with torch.no_grad():
            self.encoder.forward_features(x)
        for h in hooks:
            h.remove()
        self._restore_fused_attn()

        B   = x.shape[0]
        N   = _NUM_PATCHES + 1   # 4097 (patches + CLS)
        g   = _GRID_SIZE
        D, H, W = x.shape[2], x.shape[3], x.shape[4]

        # Build per-layer mean-head attention matrices [B, N, N]
        layer_attns = []
        result = torch.eye(N).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        for attn in storage:
            # attn: [B, n_heads, N, N]
            attn_mean = attn.mean(dim=1)   # [B, N, N]  mean over heads

            if discard_ratio > 0.0:
                # Zero out lowest-attention entries (per row) to reduce noise
                flat = attn_mean.reshape(B, N, N)
                k    = int(N * discard_ratio)
                if k > 0:
                    idx   = flat.topk(k, dim=-1, largest=False).indices
                    flat.scatter_(-1, idx, 0.0)
                attn_mean = flat

            # Residual connection: A_eff = 0.5 * A + 0.5 * I
            eye       = torch.eye(N).unsqueeze(0).expand(B, -1, -1)
            attn_eff  = 0.5 * attn_mean + 0.5 * eye
            attn_eff  = attn_eff / attn_eff.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            result = torch.bmm(attn_eff, result)  # [B, N, N]

            # Store this layer's CLS attention as a 3D map
            cls_layer = attn_mean[:, 0, 1:]                      # [B, 4096]
            cls_layer = cls_layer / (cls_layer.sum(dim=-1, keepdim=True) + 1e-8)
            layer_attns.append(cls_layer.reshape(B, g, g, g))    # [B, 16, 16, 16]

        # Final rollout: CLS row, patch columns
        rollout    = result[:, 0, 1:]                             # [B, 4096]
        rollout    = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)
        patch_grid = rollout.reshape(B, g, g, g)                  # [B, 16, 16, 16]

        volume = F.interpolate(
            patch_grid.unsqueeze(1).float(),
            size=(D, H, W), mode='trilinear', align_corners=False
        ).squeeze(1)                                              # [B, D, H, W]

        return {
            'patch_grid':  patch_grid,
            'volume':      volume,
            'layer_attns': layer_attns,   # list[24] of [B, 16, 16, 16]
        }


def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*60}")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen:    {total - trainable:,}")
    print(f"{'='*60}\n")
