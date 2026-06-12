#!/bin/bash
#
# Master submission script for 5-fold CV
# Submits all 5 fold training jobs + automatic result aggregation
# Uses SLURM job dependencies to chain them together
#
# Usage:
#   bash submit_all_folds.sh
#

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║            5-FOLD CV - MASTER JOB SUBMISSION WITH AUTO-AGGREGATION         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if fold scripts exist
for i in {0..4}; do
    if [ ! -f "train_fold${i}.sh" ]; then
        echo "✗ ERROR: train_fold${i}.sh not found!"
        echo ""
        echo "First run: python generate_fold_scripts.py"
        exit 1
    fi
done

echo "Submitting 5-fold training jobs..."
echo ""

# Submit all 5 fold jobs and capture job IDs
JOB_IDS=()
for i in {0..4}; do
    echo "Submitting fold $i..."
    JOB_ID=$(sbatch train_fold${i}.sh | awk '{print $4}')
    JOB_IDS+=($JOB_ID)
    echo "  ✓ Fold $i: Job ID $JOB_ID"
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ All 5 fold jobs submitted!"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Job IDs:"
for i in {0..4}; do
    echo "  Fold $i: ${JOB_IDS[$i]}"
done

# Create the aggregation job dependency string
# Format: afterok:job1:job2:job3:job4:job5
DEPENDENCY="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${JID}"
done

echo ""
echo "Creating aggregation job (will run after all 5 folds complete)..."
echo "Dependency: $DEPENDENCY"
echo ""

# Submit the aggregation job that depends on all 5 folds
AGGREGATE_JOB=$(sbatch --dependency=$DEPENDENCY aggregate_fold_results.sh | awk '{print $4}')

echo "✅ Aggregation job submitted!"
echo "  Job ID: $AGGREGATE_JOB"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 SUBMISSION SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Training jobs (parallel or queued):"
for i in {0..4}; do
    echo "  Fold $i: ${JOB_IDS[$i]}"
done
echo ""
echo "Aggregation job (runs after ALL 5 complete):"
echo "  Job ID: $AGGREGATE_JOB"
echo ""
echo "Expected timeline:"
echo "  • All 5 folds: ~2-4 hours (if parallel) or ~12-20 hours (if sequential)"
echo "  • Aggregation: ~1 minute (after all 5 complete)"
echo "  • Final output: 5fold_combined_test_predictions.csv"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "🔍 MONITORING"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Check all jobs:"
echo "  squeue -u \$USER | grep tangerine_fold"
echo ""
echo "Watch fold 0 training:"
echo "  tail -f logs/tangerine_fold0_${JOB_IDS[0]}.out"
echo ""
echo "When aggregation starts (automatically):"
echo "  squeue -u \$USER | grep aggregate_fold"
echo ""
echo "After everything completes:"
echo "  ls -lh 5fold_combined_test_predictions.csv"
echo "  cat 5fold_summary.json"
echo ""

echo "════════════════════════════════════════════════════════════════════════════"
echo "✨ Everything is automated! You can monitor progress and results will"
echo "   automatically be combined when all folds complete. 🚀"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""
