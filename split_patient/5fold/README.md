# 5-Fold Cross-Validation for TANGERINE Survival Model

**Date**: 2026-06-12  
**Status**: Ready for deployment  
**Objective**: Implement 5-fold cross-validation for robust model evaluation with round-robin test predictions

## 📊 Approach: 5-Fold CV Strategy

Instead of a single train/val/test split, this implementation uses 5-fold cross-validation:

### Split Ratios per Fold
```
Each fold uses:
  Training:   70% of patients
  Validation: 10% of patients  
  Test:       20% of patients (left-out fold)
```

### 5 Fold Structure
```
Fold 0: Patients [0-20%]   → Test,  [20-30%] → Val,  [30-100%] → Train
Fold 1: Patients [20-40%]  → Test,  [40-50%] → Val,  [0-20%, 50-100%] → Train
Fold 2: Patients [40-60%]  → Test,  [60-70%] → Val,  [0-40%, 70-100%] → Train
Fold 3: Patients [60-80%]  → Test,  [80-90%] → Val,  [0-60%, 90-100%] → Train
Fold 4: Patients [80-100%] → Test,  [0-10%]  → Val,  [10-80%] → Train
```

### Key Properties
- **Stratified**: Each fold maintains cancer ratio (~3.2%)
- **Patient-level**: No patient appears in multiple folds
- **Round-robin**: All patients appear in test exactly once
- **Reproducible**: Each fold uses seed=42

## 📁 Files in This Directory

### Core Training Scripts (Modified from trial1)
- **prepare_survival_dataset.py** ← Modified for 5-fold support
  - New parameter: `--fold_idx` (0-4 for fold, None for single split)
  - Automatically splits 5 ways using StratifiedKFold
- **finetune_tangerine_survival.py** — Training script (unchanged)
- **survival_dataset.py** — PyTorch dataset loader (unchanged)
- **tangerine_survival_model.py** — Model wrapper (unchanged)
- **cumulative_probability_layer.py** — Survival utility (unchanged)

### Fold-Specific Scripts (Auto-generated)
- **train_fold0.sh** to **train_fold4.sh** — SLURM scripts for each fold
  - Auto-generated from template via `generate_fold_scripts.py`
  - Each trains independently on its fold's data
  - Creates outputs/fold0_gamma2_pr_auc_TIMESTAMP/ etc.

### Setup & Management Scripts
- **generate_fold_scripts.py** — Create train_fold0-4.sh from template
- **run_5fold_cv.sh** — Generate all splits at once
- **collect_5fold_predictions.py** — Aggregate results from all folds

### Documentation
- **README.md** (this file) — Overview and usage
- (Inherited from trial1: README_DATA_LEAKAGE_FIX.md, GETTING_STARTED.md, etc.)

## 🚀 Quick Start on Cluster

### Step 1: Copy Files to Cluster
```bash
# On local machine
rsync -avz /path/to/5fold/ zhouh05@bigpurple.nyumc.org:/gpfs/.../tangerine_6yrs_20260611_full/5fold/

# On cluster
ssh zhouh05@bigpurple.nyumc.org
cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260611_full/5fold
```

### Step 2: Generate Splits for All Folds
```bash
bash run_5fold_cv.sh \
    /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv \
    /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
    .
```

This generates:
- `dataset_splits_fold0/` through `dataset_splits_fold4/`
- Each with `train.csv`, `val.csv`, `test.csv`, `config.json`

### Step 3: Generate Training Scripts
```bash
python generate_fold_scripts.py
```

This creates `train_fold0.sh` through `train_fold4.sh` from the template.

### Step 4: Submit All Training Jobs
```bash
# Option A: Submit all at once
for i in {0..4}; do sbatch train_fold$i.sh; done

# Option B: One at a time
sbatch train_fold0.sh
sbatch train_fold1.sh
# ... etc
```

### Step 5: Monitor Progress
```bash
# Check job status
squeue -u zhouh05 | grep tangerine_fold

# Monitor specific fold (replace JOBID)
tail -f logs/tangerine_fold0_JOBID.out
```

### Step 6: Collect Results
Once all 5 folds complete training, aggregate predictions:
```bash
python collect_5fold_predictions.py
```

This generates:
- `5fold_combined_test_predictions.csv` — All test predictions
- `5fold_summary.json` — Metrics summary

## 📊 Expected Outputs

### Per-Fold Outputs
```
outputs/fold0_gamma2_pr_auc_TIMESTAMP/
├── best_model.pth              ← Best checkpoint for fold 0
├── test_predictions.csv        ← ~20% of patients
├── test_results.json
├── calibrator.pkl
└── tensorboard/

outputs/fold1_gamma2_pr_auc_TIMESTAMP/
├── best_model.pth
├── test_predictions.csv
... (similar for folds 2-4)
```

### Aggregated Output
```
5fold_combined_test_predictions.csv
├── patient_id
├── cancer
├── time_at_event
├── pred_1 to pred_6        ← Calibrated predictions
└── pred_1_raw to pred_6_raw ← Raw predictions

5fold_summary.json
├── total_samples: ~all patients
├── unique_patients: ~all patients
├── cancer_positive: count
└── fold_info: {fold 0-4 stats}
```

## 🔄 Cross-Validation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ All Patients (stratified by cancer status)                  │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
         5-Fold Split       │            Fold 0: 20% test
            │               │
   ┌────────┴────────┐     Fold 1: 20% test
   │                 │     │
Fold 2: 20% test     Fold 3: 20% test
   │                 │
   └────────┬────────┘
            │
         Fold 4: 20% test

Each fold:
  ├─ Train 70% → best_model_fold_i.pth
  ├─ Val   10% → validation during training
  └─ Test  20% → test_predictions_fold_i.csv

Result: All 100% of patients have test predictions!
```

## 🎯 Advantages of 5-Fold CV

| Aspect | Single Split | 5-Fold CV |
|--------|-------------|-----------|
| Test set | 20% of patients | 100% of patients (5 × 20%) |
| Generalization | Single estimate | 5 independent estimates |
| Test performance variance | N/A | Can compute std dev |
| Computational cost | 1× training | 5× training |
| Statistical robustness | Lower | Higher (more data) |
| Overfitting risk | Single model | 5 independent models |

## 📈 Analysis After CV

### Load Results in R
```R
library(tidyverse)

# Load combined predictions
df_pred <- read_csv("5fold_combined_test_predictions.csv")

# Compute CV-AUC across all folds
library(pROC)
cv_auc <- roc(df_pred$cancer, df_pred$pred_6)  # Year 6 predictions
plot(cv_auc)
print(cv_auc)

# Per-fold performance
df_summary <- read_json("5fold_summary.json")
```

### Expected Results
- **CV-AUC**: Estimate of true generalization performance
- **Variance**: Stability of model across different data splits
- **Year-specific AUCs**: Performance degradation over time

## 🔧 Technical Details

### Stratified 5-Fold Split
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
    # test_idx = 20% of patients (fold)
    # train_val_idx = 80% of patients (remaining)
    
    # Split train_val into train (70%) and val (10%)
    # val_size = 10% / 80% = 0.125
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, stratify=y[train_val_idx]
    )
```

### Patient-Level Stratification
- Groups by `PatientID` (actual patient)
- Not by `ct_id` (individual scan)
- Ensures no patient appears in multiple folds

### Per-Year Weights
- Recalculated for each fold
- Based on fold's training data
- Handles year-specific class imbalance

## ⚡ Timeline & Compute

| Task | Time | Notes |
|------|------|-------|
| Generate splits (all 5) | ~5 min | Single process |
| Train fold 0 | ~2-4 hours | 1 GPU |
| Train fold 1 | ~2-4 hours | Parallel if multiple GPUs |
| ... | ... | Can run all 5 in parallel! |
| Collect results | ~5 min | Single process |
| **Total (sequential)** | **~12-20 hours** | 1 GPU |
| **Total (parallel)** | **~2-4 hours** | 5 GPUs |

## 🐛 Troubleshooting

### Issue: "fold_idx must be 0-4"
```bash
# Solution: Check parameter is valid
python prepare_survival_dataset.py --fold_idx 0  # OK
python prepare_survival_dataset.py --fold_idx 5  # Error
```

### Issue: Jobs fail with "No test_predictions.csv"
```bash
# Check job output
tail -f logs/tangerine_fold0_JOBID.out

# Common causes:
# - Dataset splits not generated (train_fold.sh regenerates automatically)
# - Out of memory (reduce batch size)
# - CUDA error (check GPU availability)
```

### Issue: Inconsistent patient counts across folds
```bash
# Verify splits were generated correctly
for i in {0..4}; do
    echo "Fold $i:"
    wc -l dataset_splits_fold$i/train.csv
    wc -l dataset_splits_fold$i/val.csv
    wc -l dataset_splits_fold$i/test.csv
done
```

## 📝 Modifying for Different Splits

To change ratios (e.g., 80/10/10 instead of 70/10/20):

Edit `prepare_survival_dataset.py` line ~145:
```python
# Current (70/10/20):
train_pats, val_pats = train_test_split(
    train_val_pats, test_size=0.125,  # 10% / 80%
    stratify=train_val_pats['cancer'], random_state=args.seed
)

# For 80/10/10:
# Use 20% test in StratifiedKFold loop
# Then split remaining 80% as: 87.5% train, 12.5% val
train_pats, val_pats = train_test_split(
    train_val_pats, test_size=0.125,
    ...
)
```

## 🔗 Related Documentation

From parent trial1/:
- `README_DATA_LEAKAGE_FIX.md` — Data leakage issues
- `GETTING_STARTED.md` — Single-split setup
- `DATASET_ANALYSIS.md` — Dataset structure
- `FIX_DETAILS.md` — Code changes

## 📚 Citation

If using 5-fold CV results:
> Models were evaluated using 5-fold cross-validation with stratified patient-level splitting (70% train, 10% validation, 20% test per fold). All predictions were made on held-out test sets, ensuring no data leakage.

---

**Ready to start?** Run these commands:
```bash
# Step 1: Generate splits
bash run_5fold_cv.sh /path/to/metadata.csv /path/to/images

# Step 2: Generate training scripts  
python generate_fold_scripts.py

# Step 3: Submit all training jobs
for i in {0..4}; do sbatch train_fold$i.sh; done

# Step 4: After all complete, collect results
python collect_5fold_predictions.py
```

🚀 **Good luck with your cross-validation!**
