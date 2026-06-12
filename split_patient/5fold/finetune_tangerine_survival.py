"""
Fine-tune TANGERINE with Sybil-style 6-year survival head.

Key features:
  - Loss:    BCE (standard or focal) weighted by y_mask for per-year masking
  - Output:  6 logits per CT (one per year)
  - Metrics: Both ROC-AUC and PR-AUC per year; best model selection by either metric
  - Calibration: CalibratedClassifierCV (sigmoid) keyed "Year1".."Year6" — exact Sybil format
  - Predictions saved as pred_1 ... pred_6 (calibrated, matches R analysis script)

Optional improvements:
  - Focal Loss: --use_focal_loss with --focal_gamma (default 2.0) for imbalanced data
  - PR-AUC monitoring: Tracks precision-recall curves during training
  - Best model metric: --best_model_metric (pr_auc or roc_auc, default pr_auc)
  - LLRD: per-block learning rate decay (--llrd_decay, default 0.75)
  - Gradient accumulation: --grad_accum steps (default 1, effective batch = batch*accum)
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import LinearSVC
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from survival_dataset import create_survival_dataloaders
from tangerine_survival_model import TANGERINESurvivalModel, count_parameters

MAX_FOLLOWUP = 6


# ── Calibration — Sybil-exact format ──────────────────────────────────────────
# Sybil uses CalibratedClassifierCV(LinearSVC(), cv=5, method='isotonic').
# We add class_weight='balanced' to handle extreme cancer class imbalance.
# Input X = raw model prob reshaped to (N,1); LinearSVC learns a 1D linear
# boundary, IsotonicRegression maps its decision values to calibrated probs.
# Format: {"Year1": CalibratedClassifierCV, ..., "Year6": CalibratedClassifierCV}

# ── Survival loss ──────────────────────────────────────────────────────────────

def survival_loss(logits: torch.Tensor,
                  y_seq:  torch.Tensor,
                  y_mask: torch.Tensor,
                  pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Masked BCE loss — only backprop through observed timepoints.
    logits, y_seq, y_mask: [B, T]
    pos_weight: [T] tensor to upweight positive timepoints (corrects cancer class imbalance)
    """
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y_seq, weight=y_mask, pos_weight=pos_weight, reduction='sum'
    )
    denom = y_mask.sum().clamp(min=1.0)
    return loss / denom


def focal_survival_loss(logits: torch.Tensor,
                        y_seq:  torch.Tensor,
                        y_mask: torch.Tensor,
                        pos_weight: torch.Tensor = None,
                        gamma: float = 2.0,
                        alpha: float = 0.25) -> torch.Tensor:
    """
    Focal loss for imbalanced survival classification.
    Down-weights easy negatives, focuses training on hard examples and positive class.
    logits, y_seq, y_mask: [B, T]
    pos_weight: [T] tensor to upweight positive timepoints
    gamma: focal loss exponent (higher = more focus on hard examples)
    alpha: balance factor (higher = more weight to positive class)
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, y_seq, reduction='none'
    )
    probs = torch.sigmoid(logits)
    p_t = probs * y_seq + (1 - probs) * (1 - y_seq)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * y_seq + (1 - alpha) * (1 - y_seq)

    if pos_weight is not None:
        pw_t = pos_weight.unsqueeze(0) * y_seq + (1 - y_seq)
        focal_weight = focal_weight * pw_t

    loss = alpha_t * focal_weight * bce * y_mask
    denom = y_mask.sum().clamp(min=1.0)
    return loss.sum() / denom


# ── Per-year AUC (Sybil logic) ─────────────────────────────────────────────────

def compute_auc_at_year(probs_all, cancer_all, time_all, year_idx):
    """
    year_idx: 0-based (0 = year 1, ..., 5 = year 6)
    Positive:  cancer=1 AND time_at_event <= year_idx
    Negative:  cancer=0 AND time_at_event >= year_idx
    """
    probs_all  = np.array(probs_all)
    cancer_all = np.array(cancer_all)
    time_all   = np.array(time_all)

    mask = ((cancer_all == 1) & (time_all <= year_idx)) | \
           ((cancer_all == 0) & (time_all >= year_idx))

    labels = ((cancer_all == 1) & (time_all <= year_idx)).astype(int)

    y_true = labels[mask]
    y_prob = probs_all[mask, year_idx]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    return roc_auc_score(y_true, y_prob)


def compute_pr_auc_at_year(probs_all, cancer_all, time_all, year_idx):
    """
    Compute Precision-Recall AUC per year using Sybil-style masking.
    year_idx: 0-based (0 = year 1, ..., 5 = year 6)
    Positive:  cancer=1 AND time_at_event <= year_idx
    Negative:  cancer=0 AND time_at_event >= year_idx
    """
    cancer_all = np.array(cancer_all)
    time_all   = np.array(time_all)
    mask = ((cancer_all == 1) & (time_all <= year_idx)) | \
           ((cancer_all == 0) & (time_all >= year_idx))
    labels = ((cancer_all == 1) & (time_all <= year_idx)).astype(int)
    y_true = labels[mask]
    y_prob = probs_all[mask, year_idx]
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float('nan')
    return average_precision_score(y_true, y_prob)


# ── Trainer ────────────────────────────────────────────────────────────────────

class SurvivalTrainer:

    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, output_dir,
                 lr=1e-4, encoder_lr_ratio=0.1, weight_decay=1e-4,
                 epochs=50, warmup_epochs=5, patience=10,
                 gradient_clip=1.0, use_amp=True,
                 llrd_decay=0.75, grad_accum=1,
                 use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25,
                 best_model_metric='pr_auc'):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = device
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.epochs        = epochs
        self.warmup_epochs = warmup_epochs
        self.patience      = patience
        self.gradient_clip = gradient_clip
        self.use_amp       = use_amp
        self.grad_accum    = max(1, grad_accum)

        self._lr          = lr
        self._encoder_lr  = lr * encoder_lr_ratio
        self._wd          = weight_decay
        self._llrd_decay  = llrd_decay

        self.use_focal_loss     = use_focal_loss
        self.focal_gamma        = focal_gamma
        self.focal_alpha        = focal_alpha
        self.best_model_metric  = best_model_metric

        # Warmup: only train head
        self.optimizer = optim.AdamW(model.head.parameters(),
                                     lr=lr, weight_decay=weight_decay)

        def _warmup_cosine(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _warmup_cosine)
        self.scaler    = torch.cuda.amp.GradScaler() if use_amp else None
        self.writer    = SummaryWriter(self.output_dir / 'tensorboard')

        self.best_score       = 0.0
        self.best_epoch       = 0
        self.patience_counter = 0

        # Compute per-year pos_weight from training data.
        # Each year has a different positive/negative ratio because y_mask varies
        # by time_at_event. A single global weight under-serves rare years (e.g.
        # year 2: 162:1, year 5: 188:1). Per-year weights balance each independently.
        # Bug fix: apply y_mask when counting positives (previous code did not).
        df_train = train_loader.dataset.df
        MAX_T = MAX_FOLLOWUP
        pw_per_year = []
        for t in range(MAX_T):
            pos = (df_train[f'y_seq_{t}'] * df_train[f'y_mask_{t}']).sum()
            neg = ((1 - df_train[f'y_seq_{t}']) * df_train[f'y_mask_{t}']).sum()
            pw_per_year.append(float(neg / max(pos, 1.0)))
        self.pos_weight = torch.tensor(pw_per_year, dtype=torch.float32).to(device)
        print(f"pos_weight per year: {[round(w,1) for w in pw_per_year]}")

        print(f"Optimizer: head-only LR={lr:.1e}  encoder LR after warmup={lr*encoder_lr_ratio:.1e}")
        print(f"Warmup epochs: {warmup_epochs}  |  Patience: {patience}  |  AMP: {use_amp}")
        print(f"LLRD decay: {llrd_decay}  |  Grad accum steps: {self.grad_accum}  "
              f"(effective batch={train_loader.batch_size * self.grad_accum})")
        if use_focal_loss:
            print(f"Focal loss: enabled  gamma={focal_gamma:.2f}  alpha={focal_alpha:.2f}")
        print(f"Best model metric: {best_model_metric.upper()}")

    # ── Train one epoch ────────────────────────────────────────────────

    def train_epoch(self, epoch):
        self.model.train()

        # Unfreeze encoder with LLRD optimizer at end of warmup
        if epoch == self.warmup_epochs:
            self.model.unfreeze_encoder()
            num_blocks = len(self.model.encoder.blocks)
            param_groups = []

            # Per-block LLRD: block 0 (earliest) gets lowest LR
            for i, block in enumerate(self.model.encoder.blocks):
                lr_scale = self._llrd_decay ** (num_blocks - 1 - i)
                param_groups.append({
                    'params': list(block.parameters()),
                    'lr': self._encoder_lr * lr_scale,
                })

            # patch_embed + norm: lowest LR
            param_groups.append({
                'params': (list(self.model.encoder.patch_embed.parameters()) +
                           list(self.model.encoder.norm.parameters())),
                'lr': self._encoder_lr * (self._llrd_decay ** num_blocks),
            })

            # Head: unchanged highest LR
            param_groups.append({
                'params': list(self.model.head.parameters()),
                'lr': self._lr,
            })

            self.optimizer = optim.AdamW(param_groups, weight_decay=self._wd)

            # Rebuild scheduler for cosine decay post-warmup
            remaining = max(1, self.epochs - self.warmup_epochs)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=remaining, eta_min=1e-7)

            lr_first = self._encoder_lr * self._llrd_decay ** (num_blocks - 1)
            lr_last  = self._encoder_lr
            print(f"  Encoder unfrozen — LLRD optimizer + cosine scheduler rebuilt")
            print(f"  Block LRs: block_0={lr_first:.2e} ... block_{num_blocks-1}={lr_last:.2e}  "
                  f"head={self._lr:.2e}")

        total_loss = 0.0
        all_probs, all_cancer, all_time = [], [], []

        pbar = tqdm(self.train_loader,
                    desc=f'Epoch {epoch+1}/{self.epochs} [Train]')

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            volumes = batch['volume'].to(self.device)
            y_seq   = batch['y_seq'].to(self.device)
            y_mask  = batch['y_mask'].to(self.device)

            is_last   = (step + 1 == len(self.train_loader))
            do_update = ((step + 1) % self.grad_accum == 0) or is_last

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits = self.model(volumes)
                    loss_fn = focal_survival_loss if self.use_focal_loss else survival_loss
                    if self.use_focal_loss:
                        loss = loss_fn(logits, y_seq, y_mask, self.pos_weight,
                                      gamma=self.focal_gamma, alpha=self.focal_alpha) / self.grad_accum
                    else:
                        loss = loss_fn(logits, y_seq, y_mask, self.pos_weight) / self.grad_accum
                self.scaler.scale(loss).backward()
                if do_update:
                    if self.gradient_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(volumes)
                loss_fn = focal_survival_loss if self.use_focal_loss else survival_loss
                if self.use_focal_loss:
                    loss = loss_fn(logits, y_seq, y_mask, self.pos_weight,
                                  gamma=self.focal_gamma, alpha=self.focal_alpha) / self.grad_accum
                else:
                    loss = loss_fn(logits, y_seq, y_mask, self.pos_weight) / self.grad_accum
                loss.backward()
                if do_update:
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Undo /grad_accum for logging purposes
            total_loss += loss.item() * self.grad_accum
            probs = torch.sigmoid(logits).detach().cpu().numpy()  # [B, 6]
            all_probs.extend(probs)
            all_cancer.extend(batch['cancer'].numpy())
            all_time.extend(batch['time_at_event'].numpy())
            pbar.set_postfix({'loss': f'{loss.item() * self.grad_accum:.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        train_auc1 = compute_auc_at_year(all_probs, all_cancer, all_time, year_idx=0)
        return avg_loss, train_auc1

    # ── Validate ───────────────────────────────────────────────────────

    def validate(self, loader, split='Val'):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_cancer, all_time, all_pids = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc=split):
                volumes = batch['volume'].to(self.device)
                y_seq   = batch['y_seq'].to(self.device)
                y_mask  = batch['y_mask'].to(self.device)

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(volumes)
                        loss_fn = focal_survival_loss if self.use_focal_loss else survival_loss
                        if self.use_focal_loss:
                            loss = loss_fn(logits, y_seq, y_mask, self.pos_weight,
                                          gamma=self.focal_gamma, alpha=self.focal_alpha)
                        else:
                            loss = loss_fn(logits, y_seq, y_mask, self.pos_weight)
                else:
                    logits = self.model(volumes)
                    loss_fn = focal_survival_loss if self.use_focal_loss else survival_loss
                    if self.use_focal_loss:
                        loss = loss_fn(logits, y_seq, y_mask, self.pos_weight,
                                      gamma=self.focal_gamma, alpha=self.focal_alpha)
                    else:
                        loss = loss_fn(logits, y_seq, y_mask, self.pos_weight)

                total_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_cancer.extend(batch['cancer'].numpy())
                all_time.extend(batch['time_at_event'].numpy())
                all_pids.extend(batch['patient_id'])

        avg_loss = total_loss / len(loader)

        year_aucs = []
        for t in range(MAX_FOLLOWUP):
            auc = compute_auc_at_year(all_probs, all_cancer, all_time, year_idx=t)
            year_aucs.append(auc)

        valid_aucs = [a for a in year_aucs if not np.isnan(a)]
        avg_auc    = float(np.mean(valid_aucs)) if valid_aucs else 0.0

        pr_year_aucs = []
        for t in range(MAX_FOLLOWUP):
            pr = compute_pr_auc_at_year(np.array(all_probs), np.array(all_cancer),
                                        np.array(all_time), year_idx=t)
            pr_year_aucs.append(pr)

        valid_prs = [p for p in pr_year_aucs if not np.isnan(p)]
        avg_pr_auc = float(np.mean(valid_prs)) if valid_prs else 0.0

        return {
            'loss':      avg_loss,
            'avg_auc':   avg_auc,
            'year_aucs': year_aucs,
            'avg_pr_auc':   avg_pr_auc,
            'pr_year_aucs': pr_year_aucs,
            'probs':     np.array(all_probs),   # [N, 6]
            'cancer':    np.array(all_cancer),
            'time':      np.array(all_time),
            'patient_ids': all_pids,
        }

    # ── Full training loop ─────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*65}")
        print(f"STARTING TANGERINE SURVIVAL FINE-TUNING")
        print(f"{'='*65}\n")

        for epoch in range(self.epochs):
            train_loss, train_auc1 = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader, 'Val')

            auc_str = '  '.join(
                f'Y{t+1}={a:.3f}' if not np.isnan(a) else f'Y{t+1}=nan'
                for t, a in enumerate(val_metrics['year_aucs'])
            )
            pr_str = '  '.join(
                f'Y{t+1}={p:.3f}' if not np.isnan(p) else f'Y{t+1}=nan'
                for t, p in enumerate(val_metrics['pr_year_aucs'])
            )
            phase = 'WARMUP (head only)' if epoch < self.warmup_epochs else 'FINETUNE (full)'
            print(f"\nEpoch {epoch+1}/{self.epochs}  [{phase}]")
            print(f"  Train loss={train_loss:.4f}  Y1-AUC={train_auc1:.4f}")
            print(f"  Val   loss={val_metrics['loss']:.4f}  avg-AUC={val_metrics['avg_auc']:.4f}  avg-PR-AUC={val_metrics['avg_pr_auc']:.4f}")
            print(f"  Val ROC-AUC: {auc_str}")
            print(f"  Val PR-AUC:  {pr_str}")

            self.writer.add_scalar('Loss/train', train_loss,            epoch)
            self.writer.add_scalar('Loss/val',   val_metrics['loss'],   epoch)
            self.writer.add_scalar('AUC/train_Y1', train_auc1,          epoch)
            self.writer.add_scalar('AUC/val_avg',  val_metrics['avg_auc'], epoch)
            for t, a in enumerate(val_metrics['year_aucs']):
                if not np.isnan(a):
                    self.writer.add_scalar(f'AUC/val_Y{t+1}', a, epoch)
            self.writer.add_scalar('PR_AUC/val_avg', val_metrics['avg_pr_auc'], epoch)
            for t, pr in enumerate(val_metrics['pr_year_aucs']):
                if not np.isnan(pr):
                    self.writer.add_scalar(f'PR_AUC/val_Y{t+1}', pr, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[-1]['lr'], epoch)

            val_score = val_metrics['avg_pr_auc'] if self.best_model_metric == 'pr_auc' \
                        else val_metrics['avg_auc']
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_epoch      = epoch
                self.patience_counter = 0
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_avg_auc':      val_metrics['avg_auc'],
                    'val_year_aucs':    val_metrics['year_aucs'],
                    'val_avg_pr_auc':   val_metrics['avg_pr_auc'],
                    'val_pr_year_aucs': val_metrics['pr_year_aucs'],
                }, self.output_dir / 'best_model.pth')
                metric_name = 'PR-AUC' if self.best_model_metric == 'pr_auc' else 'ROC-AUC'
                print(f"  New best  {metric_name}={val_score:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

            self.scheduler.step()

        print(f"\n{'='*65}")
        metric_name = 'PR-AUC' if self.best_model_metric == 'pr_auc' else 'ROC-AUC'
        print(f"TRAINING COMPLETE  best avg-{metric_name}={self.best_score:.4f}  "
              f"(epoch {self.best_epoch+1})")
        print(f"{'='*65}\n")

        self.evaluate_test()

    # ── Calibration — Sybil format ─────────────────────────────────────
    # Calibrator: {"Year1": CalibratedClassifierCV, ..., "Year6": ...}
    # Input to each: raw sigmoid score reshaped to (N, 1)
    # Exactly matches Sybil's _calibrate() interface and pickle format.

    def calibrate(self, val_metrics):
        print(f"\n{'='*65}")
        print(f"CALIBRATION (Sybil-format: CalibratedClassifierCV per year, val set)")
        print(f"{'='*65}\n")

        probs   = val_metrics['probs']    # [N, 6]
        cancer  = val_metrics['cancer']
        time    = val_metrics['time']

        calibrators = {}
        for t in range(MAX_FOLLOWUP):
            year_key = f"Year{t + 1}"
            mask   = ((cancer == 1) & (time <= t)) | ((cancer == 0) & (time >= t))
            labels = ((cancer == 1) & (time <= t)).astype(int)

            y_true = labels[mask]
            y_raw  = probs[mask, t].reshape(-1, 1)

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                print(f"  Year {t+1}: skipped (no positive/negative mix) — identity mapping")
                calibrators[year_key] = None
                continue

            # Sybil-exact: LinearSVC base + isotonic calibration, cv=5
            # class_weight='balanced' handles severe cancer class imbalance
            cal = CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=5, method='isotonic')
            cal.fit(y_raw, y_true)

            cal_probs = cal.predict_proba(y_raw)[:, 1]
            cal_auc   = roc_auc_score(y_true, cal_probs)
            print(f"  Year {t+1}: n={mask.sum():4d}  pos={y_true.sum():3d}  "
                  f"val-AUC (calibrated)={cal_auc:.4f}")
            calibrators[year_key] = cal

        with open(self.output_dir / 'calibrator.pkl', 'wb') as f:
            pickle.dump(calibrators, f)
        print(f"\n  Saved calibrator.pkl  (format: {{\"Year1\": CalibratedClassifierCV, ...}})")
        return calibrators

    def apply_calibration(self, probs, calibrators):
        """Apply per-year calibrators to a [N, T] probability array.
        Matches Sybil's _calibrate(): calibrators["YearN"].predict_proba(score)[:, 1]
        Enforces monotonicity post-hoc with cummax — independent per-year calibrators
        can otherwise produce pred_t > pred_{t+1} for some patients.
        """
        cal_probs = probs.copy()
        for t in range(MAX_FOLLOWUP):
            year_key = f"Year{t + 1}"
            cal = calibrators.get(year_key)
            if cal is not None:
                cal_probs[:, t] = cal.predict_proba(probs[:, t].reshape(-1, 1))[:, 1]
        # Enforce cumulative monotonicity: pred_t <= pred_{t+1} for all patients
        cal_probs = np.maximum.accumulate(cal_probs, axis=1)
        return cal_probs

    # ── Test evaluation ────────────────────────────────────────────────

    def evaluate_test(self):
        print(f"\n{'='*65}")
        print(f"FINAL TEST SET EVALUATION")
        print(f"{'='*65}\n")

        ckpt = torch.load(self.output_dir / 'best_model.pth', weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model (epoch {ckpt['epoch']+1}  "
              f"val avg-AUC={ckpt['val_avg_auc']:.4f})")

        val_metrics  = self.validate(self.val_loader, 'Val (for calibration)')
        calibrators  = self.calibrate(val_metrics)
        m            = self.validate(self.test_loader, 'Test')

        print(f"\nTest Results (raw):")
        print(f"  Average AUC: {m['avg_auc']:.4f}")
        for t, a in enumerate(m['year_aucs']):
            print(f"  Year {t+1} AUC: {f'{a:.4f}' if not np.isnan(a) else 'n/a'}")

        cal_probs     = self.apply_calibration(m['probs'], calibrators)
        cal_year_aucs = []
        for t in range(MAX_FOLLOWUP):
            mask   = ((m['cancer'] == 1) & (m['time'] <= t)) | \
                     ((m['cancer'] == 0) & (m['time'] >= t))
            labels = ((m['cancer'] == 1) & (m['time'] <= t)).astype(int)
            y_true = labels[mask]
            y_prob = cal_probs[mask, t]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                cal_year_aucs.append(float('nan'))
            else:
                cal_year_aucs.append(roc_auc_score(y_true, y_prob))

        valid_cal   = [a for a in cal_year_aucs if not np.isnan(a)]
        cal_avg_auc = float(np.mean(valid_cal)) if valid_cal else 0.0

        print(f"\nTest Results (calibrated):")
        print(f"  Average AUC: {cal_avg_auc:.4f}")
        for t, a in enumerate(cal_year_aucs):
            print(f"  Year {t+1} AUC: {f'{a:.4f}' if not np.isnan(a) else 'n/a'}")

        results = {
            'best_epoch':          int(ckpt['epoch']),
            'best_val_avg_auc':    float(ckpt['val_avg_auc']),
            'test_avg_auc_raw':    float(m['avg_auc']),
            'test_year_aucs_raw':  [float(a) if not np.isnan(a) else None
                                    for a in m['year_aucs']],
            'test_avg_auc_cal':    float(cal_avg_auc),
            'test_year_aucs_cal':  [float(a) if not np.isnan(a) else None
                                    for a in cal_year_aucs],
        }
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        pred_df = pd.DataFrame({'patient_id':    m['patient_ids'],
                                'cancer':         m['cancer'].tolist(),
                                'time_at_event':  m['time'].tolist()})
        for t in range(MAX_FOLLOWUP):
            pred_df[f'pred_{t+1}']     = cal_probs[:, t]
            pred_df[f'pred_{t+1}_raw'] = m['probs'][:, t]
        pred_df.to_csv(self.output_dir / 'test_predictions.csv', index=False)

        print(f"\nSaved to: {self.output_dir}")
        print(f"  test_results.json      — raw + calibrated AUC per year")
        print(f"  test_predictions.csv   — pred_1..pred_6 (calibrated, R-ready)")
        print(f"  calibrator.pkl         — Sybil-format per-year CalibratedClassifierCV")
        return m


# ── Main ───────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    print(f"\n{'='*65}")
    print(f"TANGERINE — 6-YEAR SURVIVAL FINE-TUNING")
    print(f"{'='*65}\n")
    print(f"Random seed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = create_survival_dataloaders(
        dataset_dir   = args.dataset_dir,
        images_dir    = args.images_dir,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        patch_size    = tuple(args.patch_size),
        augment_train = args.augment,
    )

    model = TANGERINESurvivalModel(
        encoder_weights_path = args.encoder_weights,
        max_followup         = MAX_FOLLOWUP,
        freeze_encoder       = (args.warmup_epochs > 0),
    )
    count_parameters(model)

    trainer = SurvivalTrainer(
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        test_loader   = test_loader,
        device        = device,
        output_dir    = args.output_dir,
        lr            = args.lr,
        encoder_lr_ratio = args.encoder_lr_ratio,
        weight_decay  = args.weight_decay,
        epochs        = args.epochs,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        gradient_clip = args.gradient_clip,
        use_amp       = args.use_amp,
        llrd_decay    = args.llrd_decay,
        grad_accum    = args.grad_accum,
        use_focal_loss = args.use_focal_loss,
        focal_gamma   = args.focal_gamma,
        focal_alpha   = args.focal_alpha,
        best_model_metric = args.best_model_metric,
    )

    if args.eval_only:
        # Skip training — load existing checkpoint and run calibration + test
        print("eval_only mode: skipping training, running calibration + test evaluation.")
        trainer.evaluate_test()
    else:
        trainer.train()
    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument('--dataset_dir',     required=True)
    p.add_argument('--images_dir',      required=True)
    p.add_argument('--output_dir',      required=True)
    p.add_argument('--encoder_weights', required=True)
    # Training
    p.add_argument('--epochs',           type=int,   default=50)
    p.add_argument('--batch_size',       type=int,   default=4)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--encoder_lr_ratio', type=float, default=0.1)
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--warmup_epochs',    type=int,   default=5)
    p.add_argument('--patience',         type=int,   default=10)
    p.add_argument('--gradient_clip',    type=float, default=1.0)
    p.add_argument('--num_workers',      type=int,   default=8)
    p.add_argument('--patch_size',       type=int, nargs=3, default=[256, 256, 256])
    p.add_argument('--use_amp',          action='store_true', default=True)
    p.add_argument('--augment',          action='store_true', default=True)
    # v3 additions
    p.add_argument('--llrd_decay',       type=float, default=0.75,
                   help='Layer-wise LR decay factor per block (0.75 = standard MAE fine-tuning)')
    p.add_argument('--grad_accum',       type=int,   default=1,
                   help='Gradient accumulation steps (effective_batch = batch_size * grad_accum)')
    p.add_argument('--use_focal_loss',    action='store_true', default=False,
                   help='Use focal loss instead of standard BCE (improves imbalanced learning)')
    p.add_argument('--focal_gamma',       type=float, default=2.0,
                   help='Focal loss exponent (higher = more focus on hard examples)')
    p.add_argument('--focal_alpha',       type=float, default=0.25,
                   help='Focal loss alpha (higher = more weight to positive class)')
    p.add_argument('--best_model_metric', default='pr_auc', choices=['roc_auc', 'pr_auc'],
                   help='Metric for best model selection (default: pr_auc for PR-AUC improvement branch)')
    p.add_argument('--eval_only',        action='store_true', default=False,
                   help='Skip training — load best_model.pth and run calibration + test evaluation only')
    p.add_argument('--seed',             type=int, default=42,
                   help='Global random seed for reproducibility')
    main(p.parse_args())
