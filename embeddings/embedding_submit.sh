#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --time=24:00:00
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Full embedding pipeline with corrections applied:
# 1. Extract PRETRAINED embeddings (all 24 layers) → outputs/pretrained/embeddings/pretrain/
# 2. Extract TRAINED embeddings (final layer) → outputs/run_XXX/embeddings/trained/
# 3. Combined plots → embeddings/combined/
# 4. LRADS coherence analysis (both models)
#
# Features:
# - Race labels shortened (American Indian → Am. Indian, etc.)
# - Results organized into trained/pretrain/combined subfolders
# - Includes coherence analysis to find best layer for LRADS

set -e

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="$SCRIPT_DIR/outputs/run_20260529_101746"
DATASET_DIR="$SCRIPT_DIR/dataset_splits"
IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
METADATA_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv"
LRADS_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/scan_master_with_lrads_value_v3_with_base.csv"

# Verify checkpoint
if [ ! -f "$RUN_DIR/best_model.pth" ]; then
    echo "ERROR: Checkpoint not found at $RUN_DIR/best_model.pth"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "================================================================================"
echo "TANGERINE EMBEDDINGS - FULL PIPELINE WITH CORRECTIONS"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Script dir:   $SCRIPT_DIR"
echo "  Run dir:      $RUN_DIR"
echo "  Dataset:      $DATASET_DIR"
echo "  Images:       $IMAGES_DIR"
echo ""

# Step 1: Extract PRETRAINED embeddings
echo "[1/4] Extracting PRETRAINED embeddings (all 24 layers)..."
echo ""
for layer in {0..23}; do
    python extract_embeddings_pretrained.py \
        --checkpoint "$SCRIPT_DIR/pretrained/mae_pretrained.pth" \
        --dataset_dir "$DATASET_DIR" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$SCRIPT_DIR/outputs/pretrained/embeddings" \
        --metadata_csv "$METADATA_CSV" \
        --lrads_csv "$LRADS_CSV" \
        --split all \
        --layer $layer \
        --reduction umap
done

echo ""
echo "[2/4] Extracting TRAINED embeddings (final layer)..."
echo ""
python extract_embeddings.py \
    --checkpoint "$RUN_DIR/best_model.pth" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "$RUN_DIR/embeddings" \
    --metadata_csv "$METADATA_CSV" \
    --lrads_csv "$LRADS_CSV" \
    --split all \
    --layer -1 \
    --reduction umap

echo ""
echo "[3/4] Analyzing LRADS cluster coherence..."
echo ""

# Analyze PRETRAINED
echo "  [3a] Pretrained model..."
python lrads_coherence_analysis.py \
    --embeddings_dir "$SCRIPT_DIR/outputs/pretrained/embeddings/pretrain" \
    --model_type pretrain \
    --output_dir "$SCRIPT_DIR/outputs/pretrained/embeddings/analysis"

echo ""

# Analyze TRAINED
echo "  [3b] Trained model..."
python lrads_coherence_analysis.py \
    --embeddings_dir "$RUN_DIR/embeddings/trained" \
    --model_type trained \
    --output_dir "$RUN_DIR/embeddings/analysis"

echo ""
echo "================================================================================"
echo "✅ PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Output structure:"
echo ""
echo "PRETRAINED:"
echo "  $SCRIPT_DIR/outputs/pretrained/embeddings/"
echo "  ├── pretrain/          - Embeddings for layers 0-23"
echo "  ├── combined/          - Combined visualization plots"
echo "  └── analysis/          - Coherence analysis results"
echo ""
echo "TRAINED:"
echo "  $RUN_DIR/embeddings/"
echo "  ├── trained/          - Final layer embeddings"
echo "  ├── combined/          - Combined visualization plots"
echo "  └── analysis/          - Coherence analysis results"
echo ""
echo "All race labels shortened (American Indian → Am. Indian, etc.)"
echo ""
