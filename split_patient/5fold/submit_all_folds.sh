#!/bin/bash
#
# 🚀 COMPLETE 5-FOLD CV MASTER SUBMISSION SCRIPT
#
# This script does EVERYTHING in one command:
#   1. Generates splits for all 5 folds
#   2. Generates training scripts
#   3. Submits all 5 fold training jobs (parallel or queued)
#   4. Submits automatic result aggregation job (depends on all 5 completing)
#   5. Combines all predictions into final CSV
#
# Usage:
#   bash submit_all_folds.sh [metadata_csv] [images_dir]
#
# Examples:
#   # Interactive (uses cluster defaults)
#   bash submit_all_folds.sh
#
#   # With arguments
#   bash submit_all_folds.sh /path/to/metadata.csv /path/to/images
#

set -e

# ════════════════════════════════════════════════════════════════════════════════
# Setup & Paths
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              🚀 5-FOLD CV - COMPLETE AUTOMATED WORKFLOW                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Working directory: $SCRIPT_DIR"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# Get paths from arguments or use defaults
# ════════════════════════════════════════════════════════════════════════════════

METADATA_CSV="${1:-}"
IMAGES_DIR="${2:-}"

# If not provided as arguments, use cluster defaults
if [ -z "$METADATA_CSV" ]; then
    METADATA_CSV="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv"
fi

if [ -z "$IMAGES_DIR" ]; then
    IMAGES_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
fi

echo "Configuration:"
echo "  Metadata CSV: $METADATA_CSV"
echo "  Images dir:   $IMAGES_DIR"
echo ""

# Verify paths exist
if [ ! -f "$METADATA_CSV" ]; then
    echo "✗ ERROR: Metadata CSV not found: $METADATA_CSV"
    echo ""
    echo "Usage: bash submit_all_folds.sh [metadata_csv_path] [images_dir_path]"
    echo ""
    echo "Example:"
    echo "  bash submit_all_folds.sh \\"
    echo "    /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/lungct_with_mrn_anonacc.csv \\"
    echo "    /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "✗ ERROR: Images directory not found: $IMAGES_DIR"
    exit 1
fi

echo "✓ Paths verified"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: Generate all 5 folds' splits
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "STEP 1: GENERATING DATASET SPLITS FOR ALL 5 FOLDS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

for FOLD in 0 1 2 3 4; do
    FOLD_DIR="./dataset_splits/fold${FOLD}"

    if [ -d "$FOLD_DIR" ]; then
        echo "Fold $FOLD: Splits already exist. Skipping..."
    else
        echo "Fold $FOLD: Generating splits..."
        mkdir -p ./dataset_splits
        python prepare_survival_dataset.py \
            --metadata_csv "$METADATA_CSV" \
            --images_dir "$IMAGES_DIR" \
            --output_dir "$FOLD_DIR" \
            --fold_idx $FOLD \
            --seed 42 > /dev/null 2>&1

        if [ $? -eq 0 ]; then
            echo "  ✓ Fold $FOLD splits generated"
        else
            echo "  ✗ Fold $FOLD failed!"
            exit 1
        fi
    fi
done

echo ""
echo "✓ All 5 fold splits ready!"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: Generate training scripts for all 5 folds
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "STEP 2: GENERATING TRAINING SCRIPTS FOR ALL 5 FOLDS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

if [ ! -f "train_fold0.sh" ]; then
    echo "✗ ERROR: train_fold0.sh template not found!"
    exit 1
fi

python generate_fold_scripts.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Training scripts generated (fold1-4.sh created from template)"
else
    echo "✗ Failed to generate training scripts!"
    exit 1
fi

echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: Submit all 5 fold training jobs
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "STEP 3: SUBMITTING 5 FOLD TRAINING JOBS"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p logs

JOB_IDS=()
for i in {0..4}; do
    if [ ! -f "train_fold${i}.sh" ]; then
        echo "✗ ERROR: train_fold${i}.sh not found!"
        exit 1
    fi

    echo "Submitting fold $i..."
    JOB_ID=$(sbatch train_fold${i}.sh | awk '{print $4}')
    JOB_IDS+=($JOB_ID)
    echo "  ✓ Fold $i: Job ID $JOB_ID"
done

echo ""
echo "✓ All 5 fold jobs submitted!"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: Submit automatic result aggregation job
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "STEP 4: SUBMITTING AUTOMATIC RESULT AGGREGATION JOB"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Create dependency string: afterok:job1:job2:job3:job4:job5
DEPENDENCY="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${JID}"
done

echo "Creating aggregation job..."
echo "Dependency: Will run after ALL 5 folds complete"
echo ""

AGGREGATE_JOB=$(sbatch --dependency=$DEPENDENCY aggregate_fold_results.sh | awk '{print $4}')

echo "✓ Aggregation job submitted!"
echo "  Job ID: $AGGREGATE_JOB"
echo ""

# ════════════════════════════════════════════════════════════════════════════════
# Summary & Instructions
# ════════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ COMPLETE WORKFLOW SUBMITTED!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 JOB SUMMARY"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Training Jobs (run in parallel or queued):"
for i in {0..4}; do
    echo "  Fold $i: ${JOB_IDS[$i]}"
done
echo ""
echo "Aggregation Job (runs after ALL 5 complete):"
echo "  Job ID: $AGGREGATE_JOB"
echo "  Purpose: Combines 5 test_predictions.csv into final output"
echo ""

echo "⏱️ EXPECTED TIMELINE"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "  Data preparation:         Complete ✓"
echo "  Training scripts:          Complete ✓"
echo "  Job submission:            Complete ✓"
echo ""
echo "  5 fold training:           ~2-4 hours (parallel) or ~12-20 hours (sequential)"
echo "  Automatic aggregation:     ~1 minute (after all folds complete)"
echo ""
echo "  📊 FINAL OUTPUT: 5fold_combined_test_predictions.csv (100% of patients!)"
echo ""

echo "🔍 MONITORING"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "Check all jobs:"
echo "  squeue -u \$USER | grep -E 'tangerine_fold|aggregate'"
echo ""
echo "Watch fold 0 training:"
echo "  tail -f logs/tangerine_fold0_${JOB_IDS[0]}.out"
echo ""
echo "When aggregation starts (automatically after all 5):"
echo "  tail -f logs/aggregate_5fold_${AGGREGATE_JOB}.out"
echo ""
echo "After completion (in this directory):"
echo "  ls -lh 5fold_combined_test_predictions.csv"
echo "  cat 5fold_summary.json"
echo ""

echo "📚 WHAT HAPPENS AUTOMATICALLY"
echo "───────────────────────────────────────────────────────────────────────────"
echo ""
echo "✓ 5 independent models trained (one per fold)"
echo "✓ Each fold has different class weights (based on training data)"
echo "✓ Each fold finds its own best model (by validation PR-AUC)"
echo "✓ Each fold predicts on its 20% test set"
echo "✓ All predictions combined into final CSV"
echo "✓ Summary statistics generated"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "🎯 No more manual steps! Everything is automated from here. 🚀"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
