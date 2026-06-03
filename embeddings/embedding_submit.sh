#!/bin/bash
# Submit parallel embedding extraction jobs with dependencies
#
# Job Structure:
#   JOB 1: Array job - extract pretrained layers 0-23 (24 parallel tasks)
#   JOB 2: Extract trained final layer (depends on JOB 1)
#   JOB 3: LRADS coherence analysis (depends on JOB 2)
#
# Usage: sbatch embedding_submit.sh

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="$SCRIPT_DIR/outputs/run_20260529_101746"

# Verify checkpoint
if [ ! -f "$RUN_DIR/best_model.pth" ]; then
    echo "ERROR: Checkpoint not found at $RUN_DIR/best_model.pth"
    exit 1
fi

cd "$SCRIPT_DIR"

echo "================================================================================"
echo "TANGERINE EMBEDDINGS - SUBMITTING PARALLEL JOBS"
echo "================================================================================"
echo ""

# JOB 1: Extract PRETRAINED embeddings (24 parallel tasks, one per layer)
echo "[1/3] Submitting array job: Extract PRETRAINED embeddings (layers 0-23)..."
PRETRAIN_JOB=$(sbatch --parsable \
    --job-name=embed_pretrain \
    --array=0-23 \
    run_embeddings_pretrained_array.sh)
echo "  Job ID: $PRETRAIN_JOB (24 parallel tasks)"
echo ""

# JOB 2: Extract TRAINED embeddings (depends on pretrain job)
echo "[2/3] Submitting trained extraction (depends on pretrain job $PRETRAIN_JOB)..."
TRAINED_JOB=$(sbatch --parsable \
    --job-name=embed_trained \
    --dependency=afterok:$PRETRAIN_JOB \
    run_embeddings_trained.sh)
echo "  Job ID: $TRAINED_JOB"
echo ""

# JOB 3: LRADS coherence analysis (depends on trained job)
echo "[3/3] Submitting coherence analysis (depends on trained job $TRAINED_JOB)..."
ANALYSIS_JOB=$(sbatch --parsable \
    --job-name=embed_analysis \
    --dependency=afterok:$TRAINED_JOB \
    run_coherence_analysis.sh)
echo "  Job ID: $ANALYSIS_JOB"
echo ""

echo "================================================================================"
echo "✅ ALL JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Job Chain:"
echo "  $PRETRAIN_JOB   (pretrain, 24 parallel)  →"
echo "  $TRAINED_JOB    (trained, 1 job)         →"
echo "  $ANALYSIS_JOB   (analysis, 1 job)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -j $PRETRAIN_JOB"
echo ""
echo "Results will be in:"
echo "  Pretrained: $SCRIPT_DIR/outputs/pretrained/embeddings/pretrain/"
echo "  Trained:    $RUN_DIR/embeddings/trained/"
echo "  Combined:   embeddings/combined/"
echo "  Analysis:   embeddings/analysis/"
echo ""
