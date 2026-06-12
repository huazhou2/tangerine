# Getting Started - Action Plan

**Status**: Ready to use (with caveats)  
**Last Updated**: 2026-06-11

---

## ✅ What's Ready

- ✅ All Python scripts synced and verified
- ✅ Dataset splits downloaded (but NEED REGENERATION)
- ✅ Path references fixed (relative paths work)
- ✅ Code issues documented and fixed
- ✅ Verification script provided

---

## ⚠️ Critical: Data Leakage Issue

**The current dataset_splits have patient-level overlap** (1,346-1,360 shared patients).

**You MUST regenerate them** using the fixed `prepare_survival_dataset.py`:

```bash
python prepare_survival_dataset.py \
  --metadata_csv /path/to/your/metadata.csv \
  --images_dir /path/to/your/images_3d_dir \
  --output_dir ./dataset_splits_fixed
```

Then either:
- Use `./dataset_splits_fixed` in all subsequent commands, OR
- Rename: `mv dataset_splits dataset_splits_old && mv dataset_splits_fixed dataset_splits`

**⏳ You cannot train models with the old splits - metrics will be invalid.**

---

## 📋 Complete Checklist

### Phase 1: Environment Setup

```bash
# 1. Install required Python packages
pip install torch SimpleITK numpy pandas scikit-learn

# 2. Verify installation
bash verify_paths.sh
# (Should show all green checkmarks, except external data warnings)
```

### Phase 2: Data Preparation

```bash
# 3. Find your data paths
export METADATA_CSV=/path/to/your/metadata.csv
export IMAGES_DIR=/path/to/your/3d_ct_images
export ENCODER_WEIGHTS=/path/to/pretrained/tangerine/weights.pth

# 4. Verify paths exist
verify_paths.sh  # Run again with env vars set

# 5. Regenerate dataset splits (FIX DATA LEAKAGE!)
python prepare_survival_dataset.py \
  --metadata_csv $METADATA_CSV \
  --images_dir $IMAGES_DIR \
  --output_dir ./dataset_splits_fixed \
  --seed 42

# 6. Verify new splits have no leakage
python -c "
import pandas as pd
t = set(pd.read_csv('dataset_splits_fixed/train.csv')['PatientID'])
v = set(pd.read_csv('dataset_splits_fixed/val.csv')['PatientID'])
s = set(pd.read_csv('dataset_splits_fixed/test.csv')['PatientID'])
assert len(t & v) == 0 and len(t & s) == 0 and len(v & s) == 0
print('✓ No data leakage detected')
"

# 7. Use the fixed splits
mv dataset_splits dataset_splits_old
mv dataset_splits_fixed dataset_splits
```

### Phase 3: Model Training

```bash
# 8. Train the model
python finetune_tangerine_survival.py \
  --dataset_dir ./dataset_splits \
  --images_dir $IMAGES_DIR \
  --output_dir ./checkpoints \
  --encoder_weights $ENCODER_WEIGHTS \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --warmup_epochs 5 \
  --patience 10 \
  --use_amp \
  --seed 42

# 9. Monitor training
# - Check tensorboard: tensorboard --logdir ./checkpoints/tensorboard
# - Watch for decreasing loss and increasing AUC
```

### Phase 4: Evaluation & Analysis

```bash
# 10. Extract predictions
python extract_test_predictions.py \
  --dataset_dir ./dataset_splits \
  --images_dir $IMAGES_DIR \
  --model_path ./checkpoints/best_model.pth \
  --output_dir ./results

# 11. Generate reports
python generate_patient_reports.py \
  --predictions ./results/test_predictions.csv \
  --output_dir ./reports

# 12. Plot results
python plot_roc_by_interval.py \
  --split_csv ./dataset_splits/test.csv \
  --predictions ./results/test_predictions.csv \
  --output_dir ./results/plots
```

---

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'models_vit'"

```bash
# Check tangerine model exists
ls ../tangerine/3D-MAE-MedImaging/models_vit.py

# If missing, clone it
cd ../tangerine
git clone https://github.com/niccolo246/3D-MAE-MedImaging.git
```

### "FileNotFoundError: metadata.csv"

```bash
# Set the correct path
export METADATA_CSV=/correct/path/to/metadata.csv
echo $METADATA_CSV  # verify it's set
```

### "CUDA out of memory"

```bash
# Reduce batch size
python finetune_tangerine_survival.py ... --batch_size 2

# Or use CPU (slow but works)
# Set CUDA_VISIBLE_DEVICES=-1 or add --device cpu
```

### "Model metrics are suspiciously high"

Check if you're using:
- Old dataset_splits with data leakage? → Regenerate
- Different split method? → Check config.json for "split_method"
- Different data? → Retrain on new dataset

---

## 📚 Documentation Files

| File | Purpose | Read Time |
|------|---------|-----------|
| **README_DATA_LEAKAGE_FIX.md** | Quick summary of data leakage issue | 5 min |
| **DATASET_ANALYSIS.md** | Detailed analysis of all issues found | 20 min |
| **FIX_DETAILS.md** | Technical code changes | 10 min |
| **PATH_REFERENCE.md** | All path configurations | 10 min |
| **GETTING_STARTED.md** | This file - action plan | 5 min |

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Install packages
pip install torch SimpleITK pandas scikit-learn numpy

# 2. Set data paths
export METADATA_CSV=/path/to/metadata.csv
export IMAGES_DIR=/path/to/images
export ENCODER_WEIGHTS=/path/to/weights.pth

# 3. Regenerate splits (FIX DATA LEAKAGE!)
python prepare_survival_dataset.py \
  --metadata_csv $METADATA_CSV \
  --images_dir $IMAGES_DIR \
  --output_dir ./dataset_splits_fixed --seed 42

# 4. Use new splits
rm -rf dataset_splits && mv dataset_splits_fixed dataset_splits

# 5. Train
python finetune_tangerine_survival.py \
  --dataset_dir ./dataset_splits \
  --images_dir $IMAGES_DIR \
  --output_dir ./checkpoints \
  --encoder_weights $ENCODER_WEIGHTS \
  --epochs 50 --batch_size 4 --lr 1e-4
```

---

## 📊 Expected Results

After training, you should see:

**Training Progress**:
```
Epoch 1/50 [Train]: loss=0.4532, auc1=0.6234
Epoch 1/50 [Val]:   loss=0.4128, auc1=0.6891, avg_auc=0.6234
...
Epoch 50/50 [Test]:  loss=0.3512, avg_auc=0.7456, avg_pr_auc=0.4523
```

**Important**: Metrics will be **lower than with old splits** (5-15% decrease) but are **trustworthy** for publication.

---

## 🚀 Next After Training

1. **Save best model**: Automatically saved as `best_model.pth` in output_dir
2. **Evaluate on test set**: Run extract_test_predictions.py
3. **Generate visualizations**: Run plot_roc_by_interval.py
4. **Document results**: Include dataset version and split method in papers
5. **Verify reproducibility**: Run with same seed=42 to get same results

---

## 🆘 Getting Help

1. **Path issues?** → See PATH_REFERENCE.md
2. **Data leakage questions?** → See DATASET_ANALYSIS.md
3. **Code errors?** → Check FIX_DETAILS.md
4. **Script arguments?** → Run `python <script>.py --help`
5. **Installation issues?** → See troubleshooting above

---

## Key Points to Remember

✅ **DO**:
- Regenerate dataset_splits with fixed code
- Use relative paths (already fixed in code)
- Run verify_paths.sh before training
- Document your dataset version in results
- Save your trained model checkpoints

❌ **DON'T**:
- Use old dataset_splits (has data leakage)
- Trust metrics from models trained on old splits
- Use hardcoded cluster paths (already fixed)
- Skip the data regeneration step
- Publish results without mentioning patient-level split fix

---

## Progress Tracker

Track your progress:

- [ ] Install Python packages
- [ ] Set environment variables
- [ ] Run verify_paths.sh (all green)
- [ ] Have metadata CSV and images ready
- [ ] Have pretrained weights
- [ ] Regenerate dataset_splits
- [ ] Verify no data leakage
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Evaluate test set
- [ ] Generate visualizations
- [ ] Document results

---

**You're all set! Questions? Check the documentation files above.** 🚀
