#!/bin/bash
#SBATCH --job-name=tan_embed_pretrain
#SBATCH --output=logs/embed_pretrain_%j.out
#SBATCH --error=logs/embed_pretrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

mkdir -p logs

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
CHECKPOINT="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine/pretrained/mae_pretrained.pth"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
OUTPUT_DIR="$SCRIPT_DIR/outputs/pretrained/embeddings"
LRADS_CSV="$SCRIPT_DIR/scan_master_with_lrads_value_v3_with_base.csv"
METADATA_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv"

mkdir -p "$OUTPUT_DIR"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:$LD_LIBRARY_PATH
unset CUDA_HOME
[ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=0

cd "$SCRIPT_DIR"

LAYERS="${LAYERS:-6 12 18 23}"
if [ "$LAYERS" = "all" ]; then
    LAYERS=$(seq 0 23)
fi

for LAYER in $LAYERS; do
    echo "=================================================="
    echo "=== Layer $LAYER ==="
    echo "=================================================="
    python extract_embeddings_pretrained.py \
        --checkpoint   "$CHECKPOINT" \
        --dataset_dir  "$DATASET_DIR" \
        --images_dir   "$IMAGES_DIR" \
        --output_dir   "$OUTPUT_DIR" \
        --lrads_csv    "$LRADS_CSV" \
        --metadata_csv "$METADATA_CSV" \
        --split        all \
        --reduction    umap \
        --batch_size   4 \
        --layer        $LAYER
    [ $? -ne 0 ] && echo "ERROR: Layer $LAYER failed"
    echo ""
done

echo "Running layer probe analysis..."
python layer_probe_analysis.py \
    --embeddings_dir "$OUTPUT_DIR" \
    --output_dir     "$OUTPUT_DIR/layer_probe"

echo "Done. Outputs in $OUTPUT_DIR"
