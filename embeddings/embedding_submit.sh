#!/bin/bash
# Submit embedding array job + probe analysis for run_20260529_101746.
# Usage: ./embedding_submit.sh

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="$SCRIPT_DIR/outputs/run_20260529_101746"

# Verify checkpoint exists before submitting
if [ ! -f "$RUN_DIR/best_model.pth" ]; then
    echo "ERROR: checkpoint not found at $RUN_DIR/best_model.pth"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "Submitting embedding array job for: $RUN_DIR"
ARRAY_JOB=$(sbatch --parsable run_embeddings.sh)
echo "  Array job ID: $ARRAY_JOB  (24 tasks, layers 0-23)"

echo "Submitting probe analysis (runs after all array tasks complete)..."
PROBE_JOB=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB run_probe_analysis.sh)
echo "  Probe job ID: $PROBE_JOB"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $SCRIPT_DIR/logs/embed_${ARRAY_JOB}_0.out"
echo ""
echo "Results will be in:"
echo "  Trained:    $RUN_DIR/embeddings/"
echo "  Pretrained: $SCRIPT_DIR/outputs/pretrained/embeddings/"
echo "  Layer probe (trained):    $RUN_DIR/embeddings/layer_probe/"
echo "  Layer probe (pretrained): $SCRIPT_DIR/outputs/pretrained/embeddings/layer_probe/"
