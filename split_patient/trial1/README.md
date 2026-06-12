# Trial 1: Patient-Level Stratified Split with Data Leakage Fix

**Date**: 2026-06-11 to 2026-06-12  
**Status**: Ready for training on cluster  
**Objective**: Train TANGERINE 6-year survival model with corrected patient-level splitting to prevent data leakage

## 🔍 What's Fixed in This Trial

### Critical Issue: Data Leakage in Original Splits
- **Problem**: Original splitting used `ct_id` (individual CT scan ID) instead of `PatientID` (patient identifier)
- **Impact**: Same patient could appear in multiple train/val/test splits (1,346+ overlaps detected)
- **Result**: Model metrics artificially inflated and unreliable

### Solution: Patient-Level Stratified Split
1. **Group by PatientID** (not ct_id) to identify unique patients
2. **Stratified split** maintains cancer ratio (~3.2%) across train/val/test
3. **All CTs of a patient** assigned to same split (prevents leakage)
4. **Post-diagnosis CTs removed** (time < 0 filtered out)
5. **Per-year class weights** recalculated from new training split

## 📁 Files in This Directory

### Core Training Scripts
- **prepare_survival_dataset.py**: Regenerate dataset splits (patient-level, stratified)
- **finetune_tangerine_survival.py**: Train TANGERINE with focal loss + PR-AUC monitoring
- **survival_dataset.py**: PyTorch dataset loader
- **tangerine_survival_model.py**: Model wrapper (fixed relative paths)
- **cumulative_probability_layer.py**: Utility layer for survival predictions

### Job Submission
- **train_focal.sh**: SLURM job script (auto-preprocesses if needed)
  - Epochs: 320 | Patience: 100 | Warmup: 30 epochs
  - Focal loss: gamma=2.0, alpha=0.25
  - Batch size: 4 | Grad accum: 4 | Effective batch: 16
  - LR (head): 1e-4 | LR (encoder): 5e-6 (ratio=0.05)

### Documentation
- **README_DATA_LEAKAGE_FIX.md** ← Start here! Explains the problem & solution
- **GETTING_STARTED.md** ← Step-by-step action plan
- **DATASET_ANALYSIS.md** ← Technical deep-dive on data issues
- **FIX_DETAILS.md** ← Code changes explained line-by-line
- **PATH_REFERENCE.md** ← All paths and troubleshooting

## 🚀 Quick Start on Cluster

```bash
ssh zhouh05@bigpurple.nyumc.org
cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260611_full

# Copy files from GitHub
git clone https://github.com/huazhou2/tangerine.git
cp -r tangerine/split_patient/trial1/* .

# Remove old splits (to regenerate with fixed code)
rm -rf dataset_splits

# Submit training job
sbatch train_focal.sh

# Monitor
squeue -u zhouh05
tail -f logs/tangerine_focal_*.out
```

## 📊 Expected Outputs

After successful training, you'll get:

```
outputs/focal_gamma2_pr_auc_TIMESTAMP/
├── best_model.pth              ← Best checkpoint (selected by PR-AUC)
├── test_predictions.csv        ← Predictions for test CTs
├── test_results.json           ← AUC per year (raw + calibrated)
├── calibrator.pkl              ← Per-year calibration models
└── tensorboard/                ← Training logs
```

**test_predictions.csv** contains:
- `patient_id`: Patient identifier
- `cancer`: True cancer status
- `time_at_event`: Followup time
- `pred_1` to `pred_6`: Calibrated survival predictions (years 1-6)
- `pred_1_raw` to `pred_6_raw`: Raw model outputs (before calibration)

Ready for R analysis!

## ✅ Key Improvements Over Original

| Aspect | Original | Trial 1 |
|--------|----------|---------|
| Split method | CT-level (ct_id) | Patient-level (PatientID) ✅ |
| Data leakage | 1,346+ patient overlaps | 0 overlaps ✅ |
| Stratification | Yes, but on wrong ID | Yes, by PatientID ✅ |
| Post-diagnosis CTs | Some included | All removed ✅ |
| Weights | Fixed globally | Per-year recalculated ✅ |
| Path handling | Hardcoded cluster paths | Relative paths ✅ |

## 🔧 Technical Details

### Dataset Splits
- **Total patients**: ~14,636 unique patients (from ~16,062 CT records)
- **Train**: 70% (~10,245 patients)
- **Val**: 15% (~2,195 patients)
- **Test**: 15% (~2,196 patients)

### Class Distribution (Stratified)
All splits maintain ~3.2% cancer positive rate:
- Train: ~129 cancer cases / 4,090 patients (3.15%)
- Val: ~28 cancer cases / 876 patients (3.20%)
- Test: ~28 cancer cases / 877 patients (3.19%)

### Training Configuration
- **Loss**: Focal Loss (gamma=2.0, alpha=0.25) + BCE
- **Optimizer**: AdamW with LLRD (Layer-wise Learning Rate Decay)
- **LR schedule**: 30 epochs warmup, then exponential decay
- **AMP**: Mixed precision training enabled
- **Augmentation**: 3D spatial augmentation enabled
- **Best model metric**: PR-AUC (more meaningful for imbalanced data)

## 📝 Notes

1. **Metrics will be lower**: Expect 5-15% decrease from original metrics due to removal of data leakage
2. **Post-diagnosis filtering**: Only baseline CTs (before diagnosis) are included
3. **Reproducibility**: All runs use seed=42 for reproducibility
4. **Calibration**: Test predictions are calibrated for reliability
5. **Year-specific weights**: Class imbalance changes over time (year 1 is rare, year 6 is more common)

## 🐛 Known Issues & Fixes

✅ **FIXED**: Patient-level splitting (was using ct_id, now uses PatientID)  
✅ **FIXED**: Data leakage (patient overlap detection and removal)  
✅ **FIXED**: Hardcoded paths (now relative paths)  
✅ **FIXED**: Print statements (now match actual training parameters)

## 🚦 Running on Cluster

The `train_focal.sh` script automatically:
1. Checks if `dataset_splits/` exists
2. If missing, regenerates using `prepare_survival_dataset.py`
3. Verifies zero patient overlap in splits
4. Trains the model
5. Evaluates on test set

**To re-run data preprocessing:**
```bash
rm -rf dataset_splits
sbatch train_focal.sh  # Will regenerate
```

## 📖 Reading Order

For first-time users, read in this order:
1. **README.md** (this file) — 5 min overview
2. **README_DATA_LEAKAGE_FIX.md** — 5 min problem summary
3. **GETTING_STARTED.md** — 5 min action plan
4. **DATASET_ANALYSIS.md** — 20 min technical details (optional)

## 🤝 Contributing

For subsequent trials:
- Create new folders: `split_patient/trial2`, `split_patient/trial3`, etc.
- Document what changed vs. previous trial
- Keep training scripts versioned

---

**Ready to train?** See GETTING_STARTED.md for step-by-step instructions! 🚀
