#!/bin/bash
#SBATCH --job-name=embed_pretrain
#SBATCH --time=2:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Array job: extract pretrained embeddings for layers 0-23
# Each task gets one layer via $SLURM_ARRAY_TASK_ID

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
METADATA_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv"
LRADS_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/scan_master_with_lrads_value_v3_with_base.csv"

cd "$SCRIPT_DIR"

LAYER=$SLURM_ARRAY_TASK_ID

echo "================================================================================"
echo "Extracting PRETRAINED embeddings - Layer $LAYER"
echo "================================================================================"
echo "Task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python extract_embeddings_pretrained.py \
    --checkpoint "$SCRIPT_DIR/pretrained/mae_pretrained.pth" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$SCRIPT_DIR/outputs/pretrained/embeddings" \
    --metadata_csv "$METADATA_CSV" \
    --lrads_csv "$LRADS_CSV" \
    --split all \
    --layer $LAYER \
    --reduction umap

echo ""
echo "✅ Layer $LAYER complete"
