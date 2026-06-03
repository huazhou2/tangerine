#!/bin/bash
# Submit 6 parallel jobs for comprehensive embedding analysis
# All results go to: outputs/run_20260529_101746/embeddings/{pretrain,trained,combined}/

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="$SCRIPT_DIR/outputs/run_20260529_101746"
EMBEDDINGS_DIR="$RUN_DIR/embeddings"

cd "$SCRIPT_DIR"

echo "================================================================================"
echo "TANGERINE EMBEDDINGS - 6-JOB PARALLEL PIPELINE"
echo "================================================================================"
echo ""
echo "All results → $EMBEDDINGS_DIR/"
echo "  ├── pretrain/"
echo "  ├── trained/"
echo "  ├── combined/"
echo "  └── analysis/"
echo ""

# JOB 1: Pretrained layers 0-8 + pre/post norm
echo "[1/6] Submitting Job 1: Pretrained layers 0-8 + pre/post norm..."
JOB1=$(sbatch --parsable \
    --job-name=embed_p1_layers0-8 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=4 \
    --partition=a100_long \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_p1_layers0-8
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_long

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python extract_embeddings_extended.py \
    --checkpoint pretrained/mae_pretrained.pth \
    --dataset_dir dataset_splits \
    --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
    --output_dir outputs/run_20260529_101746/embeddings \
    --layers 0-8 \
    --representation_types full,pre_norm,post_norm
JOBEOF
)
echo "  Job ID: $JOB1"

# JOB 2: Pretrained layers 9-16 + attention heads
echo "[2/6] Submitting Job 2: Pretrained layers 9-16 + attention heads..."
JOB2=$(sbatch --parsable \
    --job-name=embed_p2_layers9-16 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=4 \
    --partition=a100_long \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_p2_layers9-16
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_long

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python extract_embeddings_extended.py \
    --checkpoint pretrained/mae_pretrained.pth \
    --dataset_dir dataset_splits \
    --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
    --output_dir outputs/run_20260529_101746/embeddings \
    --layers 9-16 \
    --representation_types full,attention_heads
JOBEOF
)
echo "  Job ID: $JOB2"

# JOB 3: Pretrained layers 17-23 + pooling
echo "[3/6] Submitting Job 3: Pretrained layers 17-23 + pooling..."
JOB3=$(sbatch --parsable \
    --job-name=embed_p3_layers17-23 \
    --time=04:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=4 \
    --partition=a100_long \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_p3_layers17-23
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_long

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python extract_embeddings_extended.py \
    --checkpoint pretrained/mae_pretrained.pth \
    --dataset_dir dataset_splits \
    --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
    --output_dir outputs/run_20260529_101746/embeddings \
    --layers 17-23 \
    --representation_types full,mean_pool,max_pool
JOBEOF
)
echo "  Job ID: $JOB3"

# JOB 4: Trained final layer
echo "[4/6] Submitting Job 4: Trained final layer..."
JOB4=$(sbatch --parsable \
    --job-name=embed_trained_final \
    --time=02:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    --cpus-per-task=4 \
    --partition=a100_long \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_trained_final
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100_long

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python extract_embeddings.py \
    --checkpoint outputs/run_20260529_101746/best_model.pth \
    --dataset_dir dataset_splits \
    --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
    --output_dir outputs/run_20260529_101746/embeddings \
    --layer -1 \
    --reduction umap
JOBEOF
)
echo "  Job ID: $JOB4"

# JOB 5: Coherence analysis pretrained
echo "[5/6] Submitting Job 5: Coherence analysis (pretrained) [depends on 1-3]..."
JOB5=$(sbatch --parsable \
    --job-name=embed_analysis_pretrain \
    --time=02:00:00 \
    --dependency=afterok:$JOB1:$JOB2:$JOB3 \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_analysis_pretrain
#SBATCH --time=02:00:00

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python lrads_coherence_analysis.py \
    --embeddings_dir outputs/run_20260529_101746/embeddings/pretrain \
    --model_type pretrain \
    --output_dir outputs/run_20260529_101746/embeddings/analysis
JOBEOF
)
echo "  Job ID: $JOB5"

# JOB 6: Coherence analysis trained
echo "[6/6] Submitting Job 6: Coherence analysis (trained) [depends on 4]..."
JOB6=$(sbatch --parsable \
    --job-name=embed_analysis_trained \
    --time=02:00:00 \
    --dependency=afterok:$JOB4 \
    << 'JOBEOF'
#!/bin/bash
#SBATCH --job-name=embed_analysis_trained
#SBATCH --time=02:00:00

cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
python lrads_coherence_analysis.py \
    --embeddings_dir outputs/run_20260529_101746/embeddings/trained \
    --model_type trained \
    --output_dir outputs/run_20260529_101746/embeddings/analysis
JOBEOF
)
echo "  Job ID: $JOB6"

echo ""
echo "================================================================================"
echo "✅ 6 JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Job Dependencies:"
echo "  Extraction (parallel, ~4 hrs):"
echo "    $JOB1 (layers 0-8, norm)"
echo "    $JOB2 (layers 9-16, heads)"
echo "    $JOB3 (layers 17-23, pool)"
echo "    $JOB4 (trained final)"
echo ""
echo "  Analysis (parallel, ~2 hrs, depends on extraction):"
echo "    $JOB5 (pretrain analysis)"
echo "    $JOB6 (trained analysis)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER | grep embed'"
echo ""
echo "Results in:"
echo "  $EMBEDDINGS_DIR/"
echo "  ├── pretrain/       (embeddings + UMAPs)"
echo "  ├── trained/        (embeddings + UMAPs)"
echo "  ├── combined/       (combined plots)"
echo "  └── analysis/       (coherence results)"
echo ""
