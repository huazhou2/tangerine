#!/bin/bash
#SBATCH --job-name=tan6yr_20260527
#SBATCH --output=logs/tangerine_%j.out
#SBATCH --error=logs/tangerine_%j.err
#SBATCH --time=28-00:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

mkdir -p logs

echo "=================================================="
echo "TANGERINE - 6-Year Lung Cancer Survival Prediction"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="
echo ""

# ============================================================================
# Paths
# ============================================================================

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
TANGERINE_DIR="$BASE_DIR/models/tangerine"
IMAGES_DIR="$BASE_DIR/data/images_3d_swine"
METADATA_CSV="$BASE_DIR/lungct_with_mrn_anonacc.csv"
ENCODER_WEIGHTS="$TANGERINE_DIR/pretrained/mae_pretrained.pth"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
RUN_DIR="$SCRIPT_DIR/outputs/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$RUN_DIR"
mkdir -p "$DATASET_DIR"

echo "Configuration:"
echo "  Script dir:       $SCRIPT_DIR"
echo "  TANGERINE model:  $TANGERINE_DIR/3D-MAE-MedImaging"
echo "  Images directory: $IMAGES_DIR"
echo "  Metadata CSV:     $METADATA_CSV"
echo "  Dataset splits:   $DATASET_DIR"
echo "  Encoder weights:  $ENCODER_WEIGHTS"
echo "  Run output:       $RUN_DIR"
echo ""
echo "Training parameters (best config from run_20260320):"
echo "  Batch size: 4  |  Grad accum: 4  |  Effective batch: 16"
echo "  Epochs: 120  |  Patience: 50  |  Warmup: 30 epochs frozen"
echo "  LR (head): 1e-4  |  LR (encoder top): 5e-6  (ratio=0.05)"
echo "  LLRD decay: 0.75  |  Weight decay: 1e-3  |  Seed: 42"
echo ""

# ============================================================================
# Environment
# ============================================================================

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:$LD_LIBRARY_PATH
unset CUDA_HOME

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Environment: transformer"
echo ""

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

python -c "import torch, sys; sys.exit(0) if torch.cuda.is_available() else (print('ERROR: No CUDA GPU detected. Aborting.'), sys.exit(1))"
if [ $? -ne 0 ]; then exit 1; fi
echo ""

cd "$SCRIPT_DIR"

# ============================================================================
# STEP 0 — Dataset Splits
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
        echo "Splits exist but missing survival columns. Re-running prepare_survival_dataset.py ..."
        python prepare_survival_dataset.py \
            --metadata_csv "$METADATA_CSV" \
            --images_dir   "$IMAGES_DIR" \
            --output_dir   "$DATASET_DIR" \
            --max_followup 6 \
            --train_ratio  0.7 \
            --val_ratio    0.15 \
            --test_ratio   0.15 \
            --seed         42
        if [ $? -ne 0 ]; then echo "ERROR: Dataset preparation failed!"; exit 1; fi
    fi
else
    echo "Dataset splits not found. Running prepare_survival_dataset.py ..."

    if [ ! -f "$METADATA_CSV" ]; then echo "ERROR: Metadata CSV not found: $METADATA_CSV"; exit 1; fi
    if [ ! -d "$IMAGES_DIR" ];   then echo "ERROR: Images directory not found: $IMAGES_DIR"; exit 1; fi

    python prepare_survival_dataset.py \
        --metadata_csv "$METADATA_CSV" \
        --images_dir   "$IMAGES_DIR" \
        --output_dir   "$DATASET_DIR" \
        --max_followup 6 \
        --train_ratio  0.7 \
        --val_ratio    0.15 \
        --test_ratio   0.15 \
        --seed         42

    if [ $? -ne 0 ]; then echo "ERROR: Dataset preparation failed!"; exit 1; fi
    echo "Dataset splits created:"
    echo "  Train: $(wc -l < $DATASET_DIR/train.csv)  Val: $(wc -l < $DATASET_DIR/val.csv)  Test: $(wc -l < $DATASET_DIR/test.csv)"
fi
echo ""

IMAGE_COUNT=$(ls -1 "$IMAGES_DIR"/*.nii.gz 2>/dev/null | wc -l)
if [ $IMAGE_COUNT -eq 0 ]; then echo "ERROR: No .nii.gz files found in $IMAGES_DIR"; exit 1; fi
echo "Images: $IMAGE_COUNT volumes in $IMAGES_DIR"

if [ ! -f "$ENCODER_WEIGHTS" ]; then echo "ERROR: Encoder weights not found: $ENCODER_WEIGHTS"; exit 1; fi
echo "Pretrained weights: $ENCODER_WEIGHTS  ($(du -h $ENCODER_WEIGHTS | cut -f1))"

if [ ! -d "$TANGERINE_DIR/3D-MAE-MedImaging" ]; then echo "ERROR: TANGERINE model code not found"; exit 1; fi
echo "TANGERINE model code: $TANGERINE_DIR/3D-MAE-MedImaging"
echo ""

# ============================================================================
# STEP 1 — Training + Calibration
# ============================================================================

echo "=================================================="
echo "STEP 1/7 — TRAINING + CALIBRATION"
echo "=================================================="
echo ""

python finetune_tangerine_survival.py \
    --dataset_dir      "$DATASET_DIR" \
    --images_dir       "$IMAGES_DIR" \
    --output_dir       "$RUN_DIR" \
    --encoder_weights  "$ENCODER_WEIGHTS" \
    --epochs           120 \
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
echo "STEP 2/7 — ROC CURVES + CONFUSION MATRIX"
echo "=================================================="
echo ""

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
echo "STEP 3/7 — ATTENTION MAPS (rollout, cancer patients only)"
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
    --cancer_only

ATTN_EXIT=$?
if [ $ATTN_EXIT -ne 0 ]; then
    echo "WARNING: Attention extraction failed (exit code: $ATTN_EXIT) — continuing..."
else
    echo "STEP 3 COMPLETE — Attention maps saved to $ATTN_DIR/by_ct/"
fi
echo ""

# ============================================================================
# STEP 4 — Attention PDF Reports
# ============================================================================

echo "=================================================="
echo "STEP 4/7 — ATTENTION PDF REPORTS"
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
echo "STEP 5/7 — GRAD-CAM MAPS (cancer patients only)"
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
    --cancer_only

GCAM_EXIT=$?
if [ $GCAM_EXIT -ne 0 ]; then
    echo "WARNING: Grad-CAM extraction failed (exit code: $GCAM_EXIT) — continuing..."
else
    echo "STEP 5 COMPLETE — Grad-CAM maps saved to $GRADCAM_DIR/by_ct/"
fi
echo ""

# ============================================================================
# STEP 6 — Grad-CAM PDF Reports
# ============================================================================

echo "=================================================="
echo "STEP 6/7 — GRAD-CAM PDF REPORTS"
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
# STEP 7 — Embedding extraction + UMAP (all splits, coloured by Lung-RADS)
# ============================================================================

echo "=================================================="
echo "STEP 7/7 — EMBEDDINGS + UMAP (Lung-RADS / cancer / pred1)"
echo "=================================================="
echo ""

EMBED_DIR="$RUN_DIR/embeddings"
mkdir -p "$EMBED_DIR"

LRADS_CSV="$SCRIPT_DIR/scan_master_with_lrads_value_v3_with_base.csv"

python extract_embeddings.py \
    --checkpoint  "$RUN_DIR/best_model.pth" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir  "$IMAGES_DIR" \
    --output_dir  "$EMBED_DIR" \
    --lrads_csv   "$LRADS_CSV" \
    --split       all \
    --reduction   umap \
    --batch_size  4

EMBED_EXIT=$?
if [ $EMBED_EXIT -ne 0 ]; then
    echo "WARNING: Embedding extraction failed (exit code: $EMBED_EXIT) — continuing..."
else
    echo "STEP 7 COMPLETE — Embeddings and UMAP plots saved to $EMBED_DIR"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=================================================="
echo "JOB COMPLETED"
echo "=================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""
echo "Output directory: $RUN_DIR"
echo ""
echo "Key files:"
echo "  test_results.json          — raw + calibrated AUC per year"
echo "  test_predictions.csv       — pred_1..pred_6 (calibrated, R-ready)"
echo "  best_model.pth             — best checkpoint"
echo "  calibrator.pkl             — CalibratedClassifierCV per year"
echo "  roc_6year_combined.png     — 6-year ROC curves, overall + by sex"
echo "  embeddings/embeddings.npy        — CLS token embeddings [N x 1024]"
echo "  embeddings/embeddings_meta.csv   — patient_id, lrads, cancer, preds"
echo "  embeddings/umap_combined.png     — UMAP coloured by LR/cancer/pred1"
echo ""
echo "  cat $RUN_DIR/test_results.json"

exit $TRAIN_EXIT
