# Path Reference & Configuration Guide

**Updated**: 2026-06-11  
**Status**: ✅ All paths fixed for local use

---

## Directory Structure

```
tangerine (parent directory)
├── tangerine_6yrs_20260611_full/  ← YOU ARE HERE
│   ├── dataset_splits/            ✓ PRESENT
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── config.json
│   ├── *.py                       (Python scripts)
│   ├── DATASET_ANALYSIS.md        (issue documentation)
│   ├── FIX_DETAILS.md
│   ├── README_DATA_LEAKAGE_FIX.md
│   └── PATH_REFERENCE.md          (this file)
│
└── tangerine/  ← REQUIRED (1 level up)
    ├── 3D-MAE-MedImaging/        ✓ FOUND
    │   ├── models_vit.py         (imported by tangerine_survival_model.py)
    │   ├── [other model files]
    │   └── [git files]
    └── QUICK_START_AUTO.sh
```

---

## Path Issues & Fixes

### ✅ FIXED: Hardcoded Cluster Path

**File**: `tangerine_survival_model.py` (line 39)

**Before** (❌ BROKEN on local machine):
```python
sys.path.insert(0, '/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/3D-MAE-MedImaging')
```

**After** (✅ WORKS locally and on cluster):
```python
tangerine_path = Path(__file__).parent.parent / 'tangerine' / '3D-MAE-MedImaging'
sys.path.insert(0, str(tangerine_path))
```

**Why**: 
- Relative path works from any location
- `Path(__file__).parent.parent` = go up 2 levels from script
- Then navigate to `tangerine/3D-MAE-MedImaging`
- Works on cluster AND locally ✓

---

## Script Arguments & Paths

### `prepare_survival_dataset.py`
Splits metadata and images into train/val/test.

**Arguments**:
```bash
python prepare_survival_dataset.py \
  --metadata_csv <path/to/metadata.csv>        # REQUIRED: CSV with patient data
  --images_dir <path/to/images_3d>             # REQUIRED: Directory with .nii.gz files
  --output_dir ./dataset_splits                # Optional: default shown
  --max_followup 6                             # Optional
  --train_ratio 0.70                           # Optional
  --val_ratio 0.15                             # Optional
  --test_ratio 0.15                            # Optional
  --seed 42                                    # Optional
```

**Outputs**:
- `./dataset_splits/train.csv` (10,245 samples)
- `./dataset_splits/val.csv` (2,195 samples)
- `./dataset_splits/test.csv` (2,196 samples)
- `./dataset_splits/config.json` (metadata)

**Dependencies**: ✓ None (all standard library)

---

### `finetune_tangerine_survival.py`
Trains TANGERINE survival prediction model.

**Arguments**:
```bash
python finetune_tangerine_survival.py \
  --dataset_dir ./dataset_splits                # REQUIRED: Output from prepare_survival_dataset.py
  --images_dir <path/to/images_3d>             # REQUIRED: Same as above
  --output_dir ./checkpoints                   # REQUIRED: Where to save models
  --encoder_weights <path/to/model.pth>        # REQUIRED: Pretrained TANGERINE weights
  --epochs 50                                  # Optional
  --batch_size 4                               # Optional
  --lr 1e-4                                    # Optional
  # ... many other optional hyperparameters
```

**Dependencies**:
- ✅ `survival_dataset.py` (local)
- ✅ `tangerine_survival_model.py` (local, now fixed)
- ✅ `cumulative_probability_layer.py` (local)
- ✅ `models_vit` (from `../tangerine/3D-MAE-MedImaging`)

---

### `survival_dataset.py`
Dataset loader for training.

**Used by**: `finetune_tangerine_survival.py`

**Key function**:
```python
create_survival_dataloaders(
    dataset_dir='./dataset_splits',           # Path to splits from prepare_survival_dataset.py
    images_dir='/path/to/images_3d',          # Path to 3D CT volumes
    batch_size=4,
    num_workers=8
)
```

**Expected files in `dataset_dir`**:
- `train.csv`, `val.csv`, `test.csv` (created by prepare_survival_dataset.py)
- Must have columns: `cancer`, `time_at_event`, `y_seq_*`, `y_mask_*`, `image_filename`

---

### Analysis & Evaluation Scripts

These scripts evaluate trained models:

- `extract_embeddings.py`
- `extract_embeddings_pretrained.py`
- `extract_gradcam.py`
- `extract_attention_maps.py`
- `extract_test_predictions.py`

**Common arguments**:
```bash
--dataset_dir ./dataset_splits                # From prepare_survival_dataset.py
--images_dir <path/to/images_3d>             # 3D CT volumes
--model_path <path/to/trained_model.pth>     # Output from finetune_tangerine_survival.py
--output_dir ./results                       # Where to save outputs
```

**Path handling**: ✅ All use relative paths or user-provided arguments

---

## Required External Files

### 1. Metadata CSV
**File**: `metadata.csv` or similar (you provide)

**Required columns**:
```
ct_id, PatientID, cancer, time_ct_to_last_event_or_followup, 
image_filename, [other columns]
```

**Example path**:
```bash
/Volumes/hua_mac/research/aris/data/lungct_with_mrn_anonacc.csv
```

### 2. 3D CT Images
**Directory**: `.nii.gz` files (you provide)

**File naming**:
- Matches `ct_id` from metadata (if available)
- Or matches `image_filename` column
- Example: `221502351466.nii.gz`

**Example path**:
```bash
/Volumes/hua_mac/research/aris/data/images_3d_swine/
```

### 3. Pretrained TANGERINE Weights
**File**: `mae_vit_large_patch16_dec512d8b.pth` or similar

**From**: Original TANGERINE paper / GitHub  
**Used for**: Initializing encoder in `finetune_tangerine_survival.py`

**Example argument**:
```bash
--encoder_weights /path/to/mae_vit_large_patch16.pth
```

---

## Path Verification Checklist

Before running any script, verify:

```bash
# 1. Dataset splits exist
[ -d "./dataset_splits" ] && echo "✓ dataset_splits found" || echo "✗ dataset_splits missing"
[ -f "./dataset_splits/train.csv" ] && echo "✓ train.csv" || echo "✗"
[ -f "./dataset_splits/val.csv" ] && echo "✓ val.csv" || echo "✗"
[ -f "./dataset_splits/test.csv" ] && echo "✓ test.csv" || echo "✗"

# 2. TANGERINE model available
[ -d "../tangerine/3D-MAE-MedImaging" ] && echo "✓ tangerine model found" || echo "✗"
[ -f "../tangerine/3D-MAE-MedImaging/models_vit.py" ] && echo "✓ models_vit.py" || echo "✗"

# 3. Python scripts in place
[ -f "survival_dataset.py" ] && echo "✓ survival_dataset.py" || echo "✗"
[ -f "tangerine_survival_model.py" ] && echo "✓ tangerine_survival_model.py" || echo "✗"
[ -f "finetune_tangerine_survival.py" ] && echo "✓ finetune_tangerine_survival.py" || echo "✗"
[ -f "cumulative_probability_layer.py" ] && echo "✓ cumulative_probability_layer.py" || echo "✗"

# 4. External data (you provide these)
[ -f "/path/to/metadata.csv" ] && echo "✓ metadata CSV" || echo "✗ metadata CSV missing"
[ -d "/path/to/images" ] && echo "✓ images directory" || echo "✗ images directory missing"
```

---

## Common Path Errors & Solutions

### Error: `ModuleNotFoundError: No module named 'models_vit'`

**Cause**: tangerine/3D-MAE-MedImaging directory not found

**Fix**:
```bash
# Check if it exists
ls -la ../tangerine/3D-MAE-MedImaging/

# If not, clone it:
cd ../tangerine
git clone https://github.com/niccolo246/3D-MAE-MedImaging.git
```

---

### Error: `FileNotFoundError: train.csv`

**Cause**: dataset_splits directory not created or wrong path

**Fix**:
```bash
# Regenerate splits (assumes metadata and images available)
python prepare_survival_dataset.py \
  --metadata_csv /path/to/metadata.csv \
  --images_dir /path/to/images_dir \
  --output_dir ./dataset_splits
```

---

### Error: `No such file or directory: /gpfs/data/...`

**Cause**: Old version of tangerine_survival_model.py with hardcoded cluster path

**Status**: ✅ FIXED in current version

**If you encounter it**: Update to latest tangerine_survival_model.py (line 39-41 should use `Path(__file__).parent.parent`)

---

## Environment Variables (Optional)

You can set these for convenience:

```bash
export DATASET_DIR=$(pwd)/dataset_splits
export IMAGES_DIR=/path/to/images_3d
export OUTPUT_DIR=$(pwd)/checkpoints
export ENCODER_WEIGHTS=/path/to/pretrained_model.pth

# Then use in scripts:
python finetune_tangerine_survival.py \
  --dataset_dir $DATASET_DIR \
  --images_dir $IMAGES_DIR \
  --output_dir $OUTPUT_DIR \
  --encoder_weights $ENCODER_WEIGHTS
```

---

## Summary Table

| Component | Type | Status | Location |
|-----------|------|--------|----------|
| Dataset splits | Directory | ✓ Present | `./dataset_splits/` |
| Python scripts | Files | ✓ Present | `./*.py` |
| TANGERINE model code | Directory | ✓ Present | `../tangerine/3D-MAE-MedImaging/` |
| Hardcoded paths | Code | ✅ FIXED | tangerine_survival_model.py:39 |
| Relative paths | Code | ✓ OK | All scripts |
| Metadata CSV | User-provided | ⏳ Need path | (you specify) |
| CT images | User-provided | ⏳ Need path | (you specify) |
| Pretrained weights | User-provided | ⏳ Need path | (you specify) |

---

## Next Steps

1. ✅ Verify TANGERINE model directory exists
2. ✅ Confirm dataset_splits is present (already synced)
3. ⏳ Prepare paths to your metadata CSV and images
4. ⏳ Obtain pretrained TANGERINE weights
5. 🚀 Run training: `python finetune_tangerine_survival.py ...`

---

**Questions?** Check that all required external files (metadata, images, weights) are accessible and paths are correct in your command-line arguments.
