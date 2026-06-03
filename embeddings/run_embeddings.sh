#!/bin/bash
#SBATCH --job-name=tan_embed
#SBATCH --output=logs/embed_%A_%a.out
#SBATCH --error=logs/embed_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Each array task handles one layer for BOTH trained and pretrained models.
# All 24 layers run in parallel — total wall time ~2-3 hours instead of ~40.
#
# After all array tasks finish, run the probe analysis (CPU-only, no GPU needed):
#   python layer_probe_analysis.py --embeddings_dir <trained_embed_dir>  --output_dir ...
#   python layer_probe_analysis.py --embeddings_dir <pretrained_embed_dir> --output_dir ...
# Or submit run_probe_analysis.sh which does this automatically.

mkdir -p logs

LAYER=$SLURM_ARRAY_TASK_ID

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
PRETRAINED_CKPT="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/pretrained/mae_pretrained.pth"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/outputs/run_20260529_101746}"
TRAINED_CKPT="$RUN_DIR/best_model.pth"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
TRAINED_EMBED_DIR="$RUN_DIR/embeddings"
PRETRAINED_EMBED_DIR="$SCRIPT_DIR/outputs/pretrained/embeddings"
LRADS_CSV="$SCRIPT_DIR/scan_master_with_lrads_value_v3_with_base.csv"
METADATA_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv"

mkdir -p "$TRAINED_EMBED_DIR" "$PRETRAINED_EMBED_DIR"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:$LD_LIBRARY_PATH
unset CUDA_HOME
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0

cd "$SCRIPT_DIR"

echo "Array task $SLURM_ARRAY_TASK_ID — Layer $LAYER"
echo "Trained checkpoint:    $TRAINED_CKPT"
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""

# ── Trained model ─────────────────────────────────────────────────────────────
echo "=== Trained model — layer $LAYER ==="
python extract_embeddings.py \
    --checkpoint   "$TRAINED_CKPT" \
    --dataset_dir  "$DATASET_DIR" \
    --images_dir   "$IMAGES_DIR" \
    --output_dir   "$TRAINED_EMBED_DIR" \
    --lrads_csv    "$LRADS_CSV" \
    --metadata_csv "$METADATA_CSV" \
    --split        all \
    --reduction    umap \
    --batch_size   4 \
    --layer        $LAYER
[ $? -ne 0 ] && echo "ERROR: Trained layer $LAYER failed"
echo ""

# ── Pretrained model ──────────────────────────────────────────────────────────
echo "=== Pretrained model — layer $LAYER ==="
python extract_embeddings_pretrained.py \
    --checkpoint   "$PRETRAINED_CKPT" \
    --dataset_dir  "$DATASET_DIR" \
    --images_dir   "$IMAGES_DIR" \
    --output_dir   "$PRETRAINED_EMBED_DIR" \
    --lrads_csv    "$LRADS_CSV" \
    --metadata_csv "$METADATA_CSV" \
    --split        all \
    --reduction    umap \
    --batch_size   4 \
    --layer        $LAYER
[ $? -ne 0 ] && echo "ERROR: Pretrained layer $LAYER failed"

echo ""
echo "Layer $LAYER done."
