#!/bin/bash
#SBATCH --job-name=aggregate_5fold
#SBATCH --output=logs/aggregate_5fold_%j.out
#SBATCH --error=logs/aggregate_5fold_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

echo "════════════════════════════════════════════════════════════════════════════"
echo "AGGREGATING 5-FOLD RESULTS"
echo "════════════════════════════════════════════════════════════════════════════"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

# Load environment
module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Running: collect_5fold_predictions.py"
echo ""

# Run the collection script
python collect_5fold_predictions.py

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ ERROR: Aggregation failed!"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "✅ AGGREGATION COMPLETE"
echo "════════════════════════════════════════════════════════════════════════════"
echo "End: $(date)"
echo ""
echo "Output files:"
echo "  • 5fold_combined_test_predictions.csv (ready for analysis!)"
echo "  • 5fold_summary.json"
echo ""
echo "Next steps:"
echo "  1. Download results:"
echo "       rsync -avz 5fold_combined_test_predictions.csv ."
echo "       rsync -avz 5fold_summary.json ."
echo ""
echo "  2. Load in R:"
echo "       df <- read_csv('5fold_combined_test_predictions.csv')"
echo "       library(pROC)"
echo "       auc <- roc(df\$cancer, df\$pred_6)"
echo ""

exit 0
