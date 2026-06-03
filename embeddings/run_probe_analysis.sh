#!/bin/bash
#SBATCH --job-name=tan_probe
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# CPU-only job: runs layer probe analysis after all embedding array tasks complete.
# Submit with dependency:
#   ARRAY_JOB=$(sbatch --parsable run_embeddings.sh)
#   sbatch --dependency=afterok:$ARRAY_JOB run_probe_analysis.sh

mkdir -p logs

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/outputs/run_20260529_101746}"
TRAINED_EMBED_DIR="$RUN_DIR/embeddings"
PRETRAINED_EMBED_DIR="$SCRIPT_DIR/outputs/pretrained/embeddings"

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/new/etc/profile.d/conda.sh
conda activate transformer

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/gpfs/home/zhouh05/.conda/envs/transformer/lib:/gpfs/share/apps/anaconda3/gpu/new/envs/transformer/lib:$LD_LIBRARY_PATH

cd "$SCRIPT_DIR"

echo "=== Layer probe analysis: trained model ==="
python layer_probe_analysis.py \
    --embeddings_dir "$TRAINED_EMBED_DIR" \
    --output_dir     "$TRAINED_EMBED_DIR/layer_probe"
echo ""

echo "=== Layer probe analysis: pretrained model ==="
python layer_probe_analysis.py \
    --embeddings_dir "$PRETRAINED_EMBED_DIR" \
    --output_dir     "$PRETRAINED_EMBED_DIR/layer_probe"

echo ""
echo "Probe analysis done."
echo "  Trained:    $TRAINED_EMBED_DIR/layer_probe/"
echo "  Pretrained: $PRETRAINED_EMBED_DIR/layer_probe/"
