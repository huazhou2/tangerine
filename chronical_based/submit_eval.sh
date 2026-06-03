#!/bin/bash
#SBATCH --job-name=tan6yr_eval
#SBATCH --output=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260319/logs/eval_%j.out
#SBATCH --error=/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260319/logs/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Usage:
#   sbatch submit_eval.sh <run_dir>
#
# Example:
#   sbatch submit_eval.sh /gpfs/.../chro_202604/outputs/run_20260406_203857

RUN_DIR="$1"

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR argument required."
    echo "Usage: sbatch submit_eval.sh <path/to/outputs/run_YYYYMMDD_HHMMSS>"
    exit 1
fi

echo "=================================================="
echo "TANGERINE — Eval + Downstream Analysis (skip retrain)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# ============================================================================
# Paths
# ============================================================================

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
TANGERINE_DIR="$BASE_DIR/models/tangerine"
IMAGES_DIR="$BASE_DIR/data/images_3d_swine"
METADATA_CSV="$BASE_DIR/lungct_with_mrn_anonacc.csv"
ENCODER_WEIGHTS="$TANGERINE_DIR/pretrained/mae_pretrained.pth"

# DATASET_DIR is the dataset_splits folder sibling to outputs/
DATASET_DIR="$(dirname $(dirname $RUN_DIR))/dataset_splits"
TANGERINE_6YRS_DIR="$(dirname $(dirname $RUN_DIR))"

mkdir -p "$TANGERINE_6YRS_DIR/logs"
mkdir -p /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260319/logs

echo "Configuration:"
echo "  Working dir:    $TANGERINE_6YRS_DIR"
echo "  Dataset splits: $DATASET_DIR"
echo "  Run dir:        $RUN_DIR"
echo ""

# ── Validate ─────────────────────────────────────────────────────────────────
if [ ! -f "$RUN_DIR/best_model.pth" ]; then
    echo "ERROR: best_model.pth not found in $RUN_DIR"
    echo "  Set RUN_DIR to the folder containing best_model.pth"
    exit 1
fi
echo "Found best_model.pth — skipping training."
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
fi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

cd "$TANGERINE_6YRS_DIR"

# ============================================================================
# STEP 1 — Calibration + Test Evaluation (eval_only)
# ============================================================================

echo "=================================================="
echo "STEP 1/5 — CALIBRATION + TEST EVALUATION"
echo "=================================================="
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
    --seed             42 \
    --eval_only

EVAL_EXIT=$?
if [ $EVAL_EXIT -ne 0 ]; then
    echo "ERROR: Evaluation failed (exit code: $EVAL_EXIT)"; exit $EVAL_EXIT; fi
echo "STEP 1 COMPLETE"
echo ""

# ============================================================================
# STEP 2 — ROC Curves
# ============================================================================

echo "=================================================="
echo "STEP 2/5 — ROC CURVES"
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
    echo "STEP 2 COMPLETE"
fi
echo ""

# ============================================================================
# STEP 3 — Attention Maps (rollout)
# ============================================================================

echo "=================================================="
echo "STEP 3/5 — ATTENTION MAPS (rollout)"
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
    echo "STEP 3 COMPLETE"
fi
echo ""

# ============================================================================
# STEP 4 — Attention PDF Reports
# ============================================================================

echo "=================================================="
echo "STEP 4/5 — ATTENTION PDF REPORTS"
echo "=================================================="
echo ""

python generate_patient_reports.py \
    --run_dir    "$RUN_DIR" \
    --meta_csv   "$METADATA_CSV" \
    --images_dir "$IMAGES_DIR"

RPT_EXIT=$?
if [ $RPT_EXIT -ne 0 ]; then
    echo "WARNING: Attention reports failed (exit code: $RPT_EXIT) — continuing..."
else
    echo "STEP 4 COMPLETE"
fi
echo ""

# ============================================================================
# STEP 5 — Grad-CAM Maps + PDF Reports
# ============================================================================

echo "=================================================="
echo "STEP 5/5 — GRAD-CAM MAPS + PDF REPORTS"
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
    echo "  Grad-CAM maps done."
fi

python generate_gradcam_reports.py \
    --run_dir    "$RUN_DIR" \
    --meta_csv   "$METADATA_CSV" \
    --images_dir "$IMAGES_DIR"

GRPT_EXIT=$?
if [ $GRPT_EXIT -ne 0 ]; then
    echo "WARNING: Grad-CAM reports failed (exit code: $GRPT_EXIT) — continuing..."
else
    echo "STEP 5 COMPLETE"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=================================================="
echo "EVAL JOB COMPLETED"
echo "=================================================="
echo "End time: $(date)"
echo "Output directory: $RUN_DIR"
echo ""
echo "Key files:"
echo "  test_results.json                   — raw + calibrated AUC per year"
echo "  test_predictions.csv                — pred_1..pred_6"
echo "  calibrator.pkl                      — per-year CalibratedClassifierCV"
echo "  roc_6year_combined.png              — 6-year ROC curves"
echo "  attention/rollout/by_ct/<id>/       — attention rollout maps"
echo "  attention/rollout/reports/ct_*.pdf  — attention PDF reports"
echo "  attention/grad_cam/by_ct/<id>/      — Grad-CAM maps"
echo "  attention/grad_cam/reports/         — Grad-CAM PDF reports"
echo ""
echo "  cat $RUN_DIR/test_results.json"
