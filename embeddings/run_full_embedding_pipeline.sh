#!/bin/bash
# Full embedding pipeline with coherence analysis and reorganization

set -e

echo "================================================================================"
echo "TANGERINE EMBEDDINGS - FULL PIPELINE"
echo "================================================================================"
echo ""

WORK_DIR="${1:-.}"
RUN_DIR="${2:-run_20260529_101746}"
CHECKPOINT="${WORK_DIR}/outputs/${RUN_DIR}/best_model.pth"
DATASET_DIR="${WORK_DIR}/dataset_splits"
IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"

echo "Configuration:"
echo "  Work dir:        $WORK_DIR"
echo "  Run dir:         $RUN_DIR"
echo "  Checkpoint:      $CHECKPOINT"
echo "  Dataset:         $DATASET_DIR"
echo ""

# Step 1: Extract pretrained embeddings
echo "[1/5] Extracting PRETRAINED embeddings..."
python extract_embeddings_pretrained.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings"

echo ""

# Step 2: Extract fine-tuned embeddings
echo "[2/5] Extracting TRAINED embeddings..."
python extract_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --dataset_dir "$DATASET_DIR" \
    --images_dir "$IMAGES_DIR" \
    --output_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings"

echo ""

# Step 3: Reorganize into subfolders
echo "[3/5] Reorganizing embeddings into folders..."
python reorganize_embeddings.py \
    --embeddings_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings"

echo ""

# Step 4: Generate visualizations
echo "[4/5] Generating UMAP visualizations..."
python replot_embeddings.py \
    --embeddings_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings" \
    --metadata_csv "$DATASET_DIR/train.csv"

echo ""

# Step 5: Analyze LRADS coherence
echo "[5/5] Analyzing LRADS cluster coherence..."
echo ""
echo "  [5a] Analyzing PRETRAINED model..."
python lrads_coherence_analysis.py \
    --embeddings_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings/pretrain" \
    --model_type pretrain \
    --output_dir "${WORK_DIR}/outputs/${RUN_DIR}/analysis"

echo ""
echo "  [5b] Analyzing TRAINED model..."
python lrads_coherence_analysis.py \
    --embeddings_dir "${WORK_DIR}/outputs/${RUN_DIR}/embeddings/trained" \
    --model_type trained \
    --output_dir "${WORK_DIR}/outputs/${RUN_DIR}/analysis"

echo ""
echo "================================================================================"
echo "✅ PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Output structure:"
echo "  embeddings/"
echo "  ├── trained/          - Fine-tuned model embeddings"
echo "  ├── pretrain/         - Pretrained model embeddings"
echo "  └── combined/         - Combined visualization plots"
echo ""
echo "  analysis/"
echo "  ├── lrads_coherence_results.json       (pretrain)"
echo "  ├── lrads_coherence_summary.png        (pretrain)"
echo "  ├── lrads_coherence_results.json       (trained)"
echo "  └── lrads_coherence_summary.png        (trained)"
echo ""
