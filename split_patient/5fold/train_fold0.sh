#!/bin/bash
#SBATCH --job-name=tangerine_fold0
#SBATCH --output=logs/tangerine_fold0_%j.out
#SBATCH --error=logs/tangerine_fold0_%j.err
#SBATCH --time=28-00:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

mkdir -p logs

echo "=================================================="
echo "TANGERINE - 5-Fold CV: Fold 0"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=================================================="
echo ""

# ============================================================================
# Setup
# ============================================================================

BASE_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct"
SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260611_full/5fold"
TANGERINE_DIR="$BASE_DIR/models/tangerine"
IMAGES_DIR="$BASE_DIR/data/images_3d_swine"
ENCODER_WEIGHTS="$TANGERINE_DIR/pretrained/mae_pretrained.pth"
METADATA_CSV="$BASE_DIR/lungct_with_mrn_anonacc.csv"

FOLD=0
DATASET_DIR="$SCRIPT_DIR/dataset_splits/fold${FOLD}"
RUN_DIR="$SCRIPT_DIR/outputs/fold${FOLD}_gamma2_pr_auc_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$RUN_DIR"

echo "Configuration:"
echo "  Script dir:       $SCRIPT_DIR"
echo "  Images directory: $IMAGES_DIR"
echo "  Dataset splits:   $DATASET_DIR"
echo "  Encoder weights:  $ENCODER_WEIGHTS"
echo "  Run output:       $RUN_DIR"
echo "  Fold:             $FOLD"
echo ""
echo "Training parameters (Focal Loss + PR-AUC):"
echo "  Batch size: 4  |  Grad accum: 4  |  Effective batch: 16"
echo "  Epochs: 320  |  Patience: 100  |  Warmup: 30 epochs"
echo "  LR (head): 1e-4  |  LR (encoder): 5e-6  (ratio=0.05)"
echo "  LLRD decay: 0.75  |  Weight decay: 1e-3  |  Seed: 42"
echo "  Focal Loss: gamma=2.0  |  alpha=0.25"
echo "  Best model metric: PR-AUC"
echo ""

# ============================================================================
# Check & Regenerate Dataset Splits if Needed
# ============================================================================

if [ ! -d "$DATASET_DIR" ]; then
    echo "=================================================="
    echo "Dataset splits not found - REGENERATING Fold $FOLD"
    echo "=================================================="
    echo ""

    echo "Running: prepare_survival_dataset.py"
    cd "$SCRIPT_DIR"
    python prepare_survival_dataset.py \
        --metadata_csv "$METADATA_CSV" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$DATASET_DIR" \
        --fold_idx $FOLD \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to regenerate dataset splits for fold $FOLD"
        exit 1
    fi

    echo ""
    echo "Verifying no data leakage (checking PatientID overlaps)..."
    python -c "
import pandas as pd
t = set(pd.read_csv('$DATASET_DIR/train.csv')['PatientID'])
v = set(pd.read_csv('$DATASET_DIR/val.csv')['PatientID'])
s = set(pd.read_csv('$DATASET_DIR/test.csv')['PatientID'])
tv, ts, vs = len(t & v), len(t & s), len(v & s)
print(f'Train-Val: {tv}, Train-Test: {ts}, Val-Test: {vs}')
if tv == 0 and ts == 0 and vs == 0:
    print('✓ No data leakage - splits valid!')
else:
    print('✗ Data leakage detected!'); exit(1)
"

    if [ $? -ne 0 ]; then
        echo "ERROR: Data leakage verification failed for fold $FOLD"
        exit 1
    fi
    echo "✓ Splits regenerated and verified for fold $FOLD"
    echo ""
else
    echo "✓ Using existing dataset_splits for fold $FOLD"
    echo ""
fi

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
# Training with Focal Loss + PR-AUC Monitoring
# ============================================================================

echo "=================================================="
echo "TRAINING FOLD $FOLD WITH FOCAL LOSS + PR-AUC"
echo "=================================================="
echo ""

python finetune_tangerine_survival.py \
    --dataset_dir      "$DATASET_DIR" \
    --images_dir       "$IMAGES_DIR" \
    --output_dir       "$RUN_DIR" \
    --encoder_weights  "$ENCODER_WEIGHTS" \
    --epochs           320 \
    --batch_size       4 \
    --lr               1e-4 \
    --encoder_lr_ratio 0.05 \
    --weight_decay     1e-3 \
    --warmup_epochs    30 \
    --patience         100 \
    --gradient_clip    1.0 \
    --num_workers      8 \
    --patch_size       256 256 256 \
    --use_amp \
    --augment \
    --llrd_decay       0.75 \
    --grad_accum       4 \
    --seed             42 \
    --use_focal_loss \
    --focal_gamma      2.0 \
    --focal_alpha      0.25 \
    --best_model_metric pr_auc

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: TRAINING FAILED FOR FOLD $FOLD (exit code: $TRAIN_EXIT)"
    exit $TRAIN_EXIT
fi
echo ""
echo "TRAINING COMPLETE FOR FOLD $FOLD"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=================================================="
echo "FOLD $FOLD COMPLETED"
echo "=================================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""
echo "Output directory: $RUN_DIR"
echo ""
echo "Key files:"
echo "  test_results.json          — raw + calibrated AUC per year"
echo "  test_predictions.csv       — pred_1..pred_6 (calibrated, R-ready)"
echo "  best_model.pth             — best checkpoint (selected by PR-AUC)"
echo "  tensorboard/               — training logs (loss, AUC, PR-AUC)"
echo ""

exit $TRAIN_EXIT
