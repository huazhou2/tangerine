#!/bin/bash
#
# 5-Fold Cross-Validation Wrapper
# Generates splits and trains models for all 5 folds
#
# Usage:
#   bash run_5fold_cv.sh /path/to/metadata.csv /path/to/images_dir [output_base_dir]
#

set -e

METADATA_CSV="${1:?Error: metadata CSV path required}"
IMAGES_DIR="${2:?Error: images directory path required}"
OUTPUT_BASE="${3:-.}"

if [ ! -f "$METADATA_CSV" ]; then
    echo "✗ ERROR: Metadata CSV not found: $METADATA_CSV"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "✗ ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    5-FOLD CROSS-VALIDATION SETUP                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Metadata CSV:    $METADATA_CSV"
echo "  Images directory: $IMAGES_DIR"
echo "  Output base:     $OUTPUT_BASE"
echo ""
echo "This will generate splits and training configs for all 5 folds"
echo ""

# Create output directories
for FOLD in 0 1 2 3 4; do
    FOLD_DIR="$OUTPUT_BASE/dataset_splits_fold${FOLD}"

    echo "════════════════════════════════════════════════════════════════════════════"
    echo "FOLD $FOLD: Generating splits (test=20%, val=10%, train=70%)"
    echo "════════════════════════════════════════════════════════════════════════════"

    python prepare_survival_dataset.py \
        --metadata_csv "$METADATA_CSV" \
        --images_dir "$IMAGES_DIR" \
        --output_dir "$FOLD_DIR" \
        --fold_idx $FOLD \
        --seed 42

    echo ""
    echo "✓ Fold $FOLD splits saved to: $FOLD_DIR"
    echo ""
done

echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ All 5 folds generated successfully!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Submit training jobs for each fold:"
echo ""
for FOLD in 0 1 2 3 4; do
    echo "       sbatch train_fold${FOLD}.sh"
done
echo ""
echo "  2. OR run all folds in sequence:"
echo "       bash submit_all_folds.sh"
echo ""
echo "  3. Collect predictions from all 5 folds:"
echo "       python collect_5fold_predictions.py"
echo ""
