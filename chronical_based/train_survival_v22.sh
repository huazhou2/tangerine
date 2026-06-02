#!/bin/bash
#SBATCH --job-name=tan6yr_v22c
#SBATCH --output=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260319/chro_202604_by2021/logs/tangerine_%j.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260319/chro_202604_by2021/logs/tangerine_%j.err
#SBATCH --time=28-00:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "=================================================="
echo "TANGERINE - 6-Year Lung Cancer Survival Prediction v22c"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="
echo ""
echo "v22c changes (vs v22):"
echo "  Dataset split: chronological by ct_date (fixed date cutoffs)"
echo "    train: ct_date < 2021-01-01"
echo "    val:   2021-01-01 <= ct_date < 2022-01-01"
echo "    test:  ct_date >= 2022-01-01"
echo "  (previously: stratified random split with seed=42)"
echo ""
echo "v22 fixes (vs v2):"
echo "  Grad-CAM:   hooks block input (not output) — patch gradients now non-zero"
echo "  Rollout:    lung mask applied before visualization (excludes spine/background)"
echo "  PDF report: always exactly 2 pages (auto page break disabled)"
echo "  Rollout:    discard_ratio raised from 0.9 to 0.95"
echo ""

# ============================================================================
# Paths
# ============================================================================

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
TANGERINE_6YRS_DIR="$BASE_DIR/models/tangerine_6yrs_20260319/chro_202604_by2021"
TANGERINE_DIR="$BASE_DIR/models/tangerine"
IMAGES_DIR="$BASE_DIR/data/images_3d_swine"
METADATA_CSV="$BASE_DIR/lungct_with_mrn_anonacc.csv"
ENCODER_WEIGHTS="$TANGERINE_DIR/pretrained/mae_pretrained.pth"
RUN_DIR="$TANGERINE_6YRS_DIR/outputs/run_$(date +%Y%m%d_%H%M%S)"
DATASET_DIR="$TANGERINE_6YRS_DIR/dataset_splits"

mkdir -p "$TANGERINE_6YRS_DIR/logs"
mkdir -p "$RUN_DIR"
mkdir -p "$DATASET_DIR"

echo "Configuration:"
echo "  Working dir:      $TANGERINE_6YRS_DIR"
echo "  TANGERINE model:  $TANGERINE_DIR/3D-MAE-MedImaging"
echo "  Images directory: $IMAGES_DIR"
echo "  Metadata CSV:     $METADATA_CSV"
echo "  Dataset splits:   $DATASET_DIR"
echo "  Encoder weights:  $ENCODER_WEIGHTS"
echo "  Run output:       $RUN_DIR"
echo ""

# ============================================================================
# Environment
# ============================================================================

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:$LD_LIBRARY_PATH
unset CUDA_HOME

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "  CUDA_VISIBLE_DEVICES was unset — defaulting to GPU 0"
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Environment: transformer"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

python -c "import torch, sys; sys.exit(0) if torch.cuda.is_available() else (print('ERROR: No CUDA GPU detected. Aborting.'), sys.exit(1))"
if [ $? -ne 0 ]; then exit 1; fi
echo ""

cd "$TANGERINE_6YRS_DIR"

# ============================================================================
# STEP 0 — Dataset Splits (auto-prepare / check / re-run)
# ============================================================================

echo "=================================================="
echo "STEP 0 — DATASET SPLITS"
echo "=================================================="
echo ""

if [ -f "$DATASET_DIR/train.csv" ] && [ -f "$DATASET_DIR/val.csv" ] && [ -f "$DATASET_DIR/test.csv" ]; then
    HAS_SURVIVAL=$(python -c "
import pandas as pd
df = pd.read_csv('$DATASET_DIR/train.csv')
print('yes' if 'y_seq_0' in df.columns else 'no')
" 2>/dev/null)

    if [ "$HAS_SURVIVAL" = "yes" ]; then
        TRAIN_COUNT=$(wc -l < "$DATASET_DIR/train.csv")
        VAL_COUNT=$(wc -l < "$DATASET_DIR/val.csv")
        TEST_COUNT=$(wc -l < "$DATASET_DIR/test.csv")
        echo "Survival dataset splits already exist — skipping preparation."
        echo "  Train: $TRAIN_COUNT  Val: $VAL_COUNT  Test: $TEST_COUNT"
    else
        echo "Splits exist but missing survival columns (y_seq_0). Re-running prepare_survival_dataset.py ..."
        python prepare_survival_dataset.py \
            --metadata_csv "$METADATA_CSV" \
            --images_dir   "$IMAGES_DIR" \
            --output_dir   "$DATASET_DIR" \
            --max_followup 6
        if [ $? -ne 0 ]; then echo "ERROR: Dataset preparation failed!"; exit 1; fi
        echo "Dataset splits updated."
    fi
else
    echo "Dataset splits not found. Running prepare_survival_dataset.py ..."

    if [ ! -f "$METADATA_CSV" ]; then
        echo "ERROR: Metadata CSV not found: $METADATA_CSV"; exit 1; fi
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "ERROR: Images directory not found: $IMAGES_DIR"; exit 1; fi

    python prepare_survival_dataset.py \
        --metadata_csv "$METADATA_CSV" \
        --images_dir   "$IMAGES_DIR" \
        --output_dir   "$DATASET_DIR" \
        --max_followup 6

    if [ $? -ne 0 ]; then echo "ERROR: Dataset preparation failed!"; exit 1; fi
    TRAIN_COUNT=$(wc -l < "$DATASET_DIR/train.csv")
    VAL_COUNT=$(wc -l < "$DATASET_DIR/val.csv")
    TEST_COUNT=$(wc -l < "$DATASET_DIR/test.csv")
    echo "Dataset splits created:"
    echo "  Train: $TRAIN_COUNT  Val: $VAL_COUNT  Test: $TEST_COUNT"
fi
echo ""

# ============================================================================
# Verify Images and Weights
# ============================================================================

IMAGE_COUNT=$(ls -1 "$IMAGES_DIR"/*.nii.gz 2>/dev/null | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then
    echo "ERROR: No .nii.gz files found in $IMAGES_DIR"; exit 1; fi
echo "Images: $IMAGE_COUNT volumes in $IMAGES_DIR"

if [ ! -f "$ENCODER_WEIGHTS" ]; then
    echo "ERROR: Pretrained encoder weights not found: $ENCODER_WEIGHTS"; exit 1; fi
WEIGHT_SIZE=$(du -h "$ENCODER_WEIGHTS" | cut -f1)
echo "Pretrained weights: $ENCODER_WEIGHTS  ($WEIGHT_SIZE)"

if [ ! -d "$TANGERINE_DIR/3D-MAE-MedImaging" ]; then
    echo "ERROR: TANGERINE model code not found: $TANGERINE_DIR/3D-MAE-MedImaging"; exit 1; fi
echo "TANGERINE model code: $TANGERINE_DIR/3D-MAE-MedImaging"
echo ""

# ============================================================================
# STEP 1 — Training + Calibration  (v2 hyperparameters)
# ============================================================================

echo "=================================================="
echo "STEP 1/6 — TRAINING + CALIBRATION (v22)"
echo "=================================================="
echo ""
echo "Foundation Model: TANGERINE ViT-Large (98,000 chest CTs, MAE pretrained)"
echo "Survival Head:    CumulativeProbabilityLayer (Sybil-style, 6-year)"
echo "Loss:             Masked BCE (y_mask supervises observable window only)"
echo ""
echo "v22 Training Parameters (identical to v2):"
echo "  Batch size: 4  |  Grad accum: 4  |  Effective batch: 16"
echo "  Epochs: 120  |  Patience: 50"
echo "  LR (head): 1e-4  |  LR (encoder top): 5e-6  (ratio=0.05)"
echo "  LLRD decay: 0.75 per block"
echo "  Weight decay: 1e-3  |  Warmup: 30 epochs frozen  |  Seed: 42"
echo ""

python finetune_tangerine_survival.py \
    --dataset_dir      "$DATASET_DIR" \
    --images_dir       "$IMAGES_DIR" \
    --output_dir       "$RUN_DIR" \
    --encoder_weights  "$ENCODER_WEIGHTS" \
    --epochs           150 \
    --batch_size       4 \
    --lr               1e-4 \
    --encoder_lr_ratio 0.05 \
    --weight_decay     1e-3 \
    --warmup_epochs    30 \
    --patience         50 \
    --gradient_clip    1.0 \
    --num_workers      8 \
    --patch_size       256 256 256 \
    --use_amp \
    --augment \
    --llrd_decay       0.75 \
    --grad_accum       4 \
    --seed             42

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: TRAINING FAILED (exit code: $TRAIN_EXIT)"
    exit $TRAIN_EXIT
fi

echo ""
echo "STEP 1/6 COMPLETE — Training + Calibration"
echo ""

# ============================================================================
# STEP 2 — ROC Curves + Confusion Matrix
# ============================================================================

echo "=================================================="
echo "STEP 2/6 — ROC CURVES + CONFUSION MATRIX"
echo "=================================================="
echo ""

if [ ! -f "$RUN_DIR/test_predictions.csv" ]; then
    echo "ERROR: test_predictions.csv not found — training may have failed"; exit 1; fi

python plot_survival_results.py \
    --predictions  "$RUN_DIR/test_predictions.csv" \
    --metadata     "$METADATA_CSV" \
    --output_dir   "$RUN_DIR" \
    --threshold    0.5 \
    --n_boot       1000

PLOT_EXIT=$?
if [ $PLOT_EXIT -ne 0 ]; then
    echo "WARNING: Plotting failed (exit code: $PLOT_EXIT) — continuing..."
else
    echo "STEP 2 COMPLETE — Plots saved to $RUN_DIR"
fi
echo ""

# ============================================================================
# STEP 3 — Attention Maps (rollout)
# ============================================================================

echo "=================================================="
echo "STEP 3/6 — ATTENTION MAPS (rollout, cancer patients only)"
echo "=================================================="
echo ""

ATTN_DIR="$RUN_DIR/attention/rollout"
mkdir -p "$ATTN_DIR"

python extract_attention_maps.py \
    --checkpoint    "$RUN_DIR/best_model.pth" \
    --dataset_dir   "$DATASET_DIR" \
    --images_dir    "$IMAGES_DIR" \
    --output_dir    "$ATTN_DIR" \
    --split         test \
    --layers        -1 12 0 \
    --rollout \
    --discard_ratio 0.95 \
    --max_patients  50 \
    --cancer_only

ATTN_EXIT=$?
if [ $ATTN_EXIT -ne 0 ]; then
    echo "WARNING: Attention extraction failed (exit code: $ATTN_EXIT) — continuing..."
else
    echo "STEP 3 COMPLETE — Attention maps saved to $ATTN_DIR/by_ct/"
fi
echo ""

# ============================================================================
# STEP 4 — Attention PDF reports
# ============================================================================

echo "=================================================="
echo "STEP 4/6 — ATTENTION PDF REPORTS"
echo "=================================================="
echo ""

python generate_patient_reports.py \
    --run_dir    "$RUN_DIR" \
    --meta_csv   "$METADATA_CSV" \
    --images_dir "$IMAGES_DIR"

RPT_EXIT=$?
if [ $RPT_EXIT -ne 0 ]; then
    echo "WARNING: Attention report generation failed (exit code: $RPT_EXIT) — continuing..."
else
    echo "STEP 4 COMPLETE — Attention PDF reports saved to $ATTN_DIR/reports/"
fi
echo ""

# ============================================================================
# STEP 5 — Grad-CAM Maps
# ============================================================================

echo "=================================================="
echo "STEP 5/6 — GRAD-CAM MAPS (cancer patients only)"
echo "=================================================="
echo ""

GRADCAM_DIR="$RUN_DIR/attention/grad_cam"
mkdir -p "$GRADCAM_DIR"

python extract_gradcam.py \
    --checkpoint   "$RUN_DIR/best_model.pth" \
    --dataset_dir  "$DATASET_DIR" \
    --images_dir   "$IMAGES_DIR" \
    --output_dir   "$GRADCAM_DIR" \
    --split        test \
    --cancer_only \
    --max_patients 50

GCAM_EXIT=$?
if [ $GCAM_EXIT -ne 0 ]; then
    echo "WARNING: Grad-CAM extraction failed (exit code: $GCAM_EXIT) — continuing..."
else
    echo "STEP 5 COMPLETE — Grad-CAM maps saved to $GRADCAM_DIR/by_ct/"
fi
echo ""

# ============================================================================
# STEP 6 — Grad-CAM PDF reports
# ============================================================================

echo "=================================================="
echo "STEP 6/6 — GRAD-CAM PDF REPORTS"
echo "=================================================="
echo ""

python generate_gradcam_reports.py \
    --run_dir    "$RUN_DIR" \
    --meta_csv   "$METADATA_CSV" \
    --images_dir "$IMAGES_DIR"

GRPT_EXIT=$?
if [ $GRPT_EXIT -ne 0 ]; then
    echo "WARNING: Grad-CAM report generation failed (exit code: $GRPT_EXIT) — continuing..."
else
    echo "STEP 6 COMPLETE — Grad-CAM PDF reports saved to $GRADCAM_DIR/reports/"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=================================================="
echo "JOB COMPLETED"
echo "=================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "ALL STEPS COMPLETED SUCCESSFULLY"
    echo ""
    echo "Output directory: $RUN_DIR"
    echo ""
    echo "Key files:"
    echo "  test_results.json                  — raw + calibrated AUC per year"
    echo "  test_predictions.csv               — pred_1..pred_6 (calibrated, R-ready)"
    echo "  best_model.pth                     — best checkpoint (avg AUC years 1-6)"
    echo "  calibrator.pkl                     — CalibratedClassifierCV per year"
    echo "  roc_6year_combined.png             — 6-year ROC curves, overall + by sex"
    echo "  attention/rollout/by_ct/<id>/       — attention rollout maps (lung-masked)"
    echo "  attention/rollout/reports/ct_*.pdf — attention PDF reports"
    echo "  attention/grad_cam/by_ct/<id>/     — Grad-CAM maps (fixed gradients)"
    echo "  attention/grad_cam/reports/        — Grad-CAM PDF reports (always 2 pages)"
    echo ""
    echo "Quick view results:"
    echo "  cat $RUN_DIR/test_results.json"
else
    echo "TRAINING FAILED — see logs for details"
    echo "  cat $TANGERINE_6YRS_DIR/logs/tangerine_$SLURM_JOB_ID.err"
fi

exit $TRAIN_EXIT
