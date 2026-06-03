#!/bin/bash
# Submit 8 parallel jobs for comprehensive embedding analysis
#
# Job allocation:
#   Jobs 1-3: Pretrained extraction (layers split by representation type)
#   Job 4:    Trained extraction  
#   Jobs 5-6: Coherence analysis (pretrained + trained)
#   Job 7:    Generate text/table reports
#   Job 8:    Final summary
#
# Usage: sbatch embedding_submit_parallel_8jobs.sh

SCRIPT_DIR="/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527"
RUN_DIR="$SCRIPT_DIR/outputs/run_20260529_101746"

cd "$SCRIPT_DIR"

echo "================================================================================"
echo "TANGERINE EMBEDDINGS - 8-JOB PARALLEL PIPELINE"
echo "================================================================================"
echo ""

# JOB 1: Pretrained layers 0-8 + pre/post norm
echo "[1/8] Pretrained layers 0-8 + pre/post norm..."
JOB1=$(sbatch --parsable \
    --job-name=embed_p1_layers0-8 \
    --time=4:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    -c 4 \
    -p a100_long \
    -c "cd $SCRIPT_DIR && python extract_embeddings_extended.py \
        --checkpoint $SCRIPT_DIR/pretrained/mae_pretrained.pth \
        --dataset_dir $SCRIPT_DIR/dataset_splits \
        --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
        --output_dir $SCRIPT_DIR/outputs/pretrained/embeddings \
        --layers 0-8 \
        --representation_types full,pre_norm,post_norm")
echo "  Job ID: $JOB1"

# JOB 2: Pretrained layers 9-16 + attention heads
echo "[2/8] Pretrained layers 9-16 + attention heads..."
JOB2=$(sbatch --parsable \
    --job-name=embed_p2_layers9-16 \
    --time=4:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    -c 4 \
    -p a100_long \
    -c "cd $SCRIPT_DIR && python extract_embeddings_extended.py \
        --checkpoint $SCRIPT_DIR/pretrained/mae_pretrained.pth \
        --dataset_dir $SCRIPT_DIR/dataset_splits \
        --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
        --output_dir $SCRIPT_DIR/outputs/pretrained/embeddings \
        --layers 9-16 \
        --representation_types full,attention_heads")
echo "  Job ID: $JOB2"

# JOB 3: Pretrained layers 17-23 + pooling strategies
echo "[3/8] Pretrained layers 17-23 + pooling strategies..."
JOB3=$(sbatch --parsable \
    --job-name=embed_p3_layers17-23 \
    --time=4:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    -c 4 \
    -p a100_long \
    -c "cd $SCRIPT_DIR && python extract_embeddings_extended.py \
        --checkpoint $SCRIPT_DIR/pretrained/mae_pretrained.pth \
        --dataset_dir $SCRIPT_DIR/dataset_splits \
        --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
        --output_dir $SCRIPT_DIR/outputs/pretrained/embeddings \
        --layers 17-23 \
        --representation_types full,mean_pool,max_pool")
echo "  Job ID: $JOB3"

# JOB 4: Trained final layer + all representation types
echo "[4/8] Trained final layer + all representations..."
JOB4=$(sbatch --parsable \
    --job-name=embed_trained_final \
    --time=2:00:00 \
    --gres=gpu:1 \
    --mem=32G \
    -c 4 \
    -p a100_long \
    -c "cd $SCRIPT_DIR && python extract_embeddings.py \
        --checkpoint $RUN_DIR/best_model.pth \
        --dataset_dir $SCRIPT_DIR/dataset_splits \
        --images_dir /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/data/images_3d_swine \
        --output_dir $RUN_DIR/embeddings \
        --layer -1 \
        --reduction umap")
echo "  Job ID: $JOB4"

# JOB 5: Coherence analysis on pretrained (depends on 1-3)
echo "[5/8] Coherence analysis (pretrained) [depends on jobs 1-3]..."
JOB5=$(sbatch --parsable \
    --job-name=embed_analysis_pretrain \
    --time=2:00:00 \
    --dependency=afterok:$JOB1:$JOB2:$JOB3 \
    -c "cd $SCRIPT_DIR && python lrads_coherence_analysis.py \
        --embeddings_dir $SCRIPT_DIR/outputs/pretrained/embeddings/pretrain \
        --model_type pretrain \
        --output_dir $SCRIPT_DIR/outputs/pretrained/embeddings/analysis")
echo "  Job ID: $JOB5"

# JOB 6: Coherence analysis on trained (depends on 4)
echo "[6/8] Coherence analysis (trained) [depends on job 4]..."
JOB6=$(sbatch --parsable \
    --job-name=embed_analysis_trained \
    --time=2:00:00 \
    --dependency=afterok:$JOB4 \
    -c "cd $SCRIPT_DIR && python lrads_coherence_analysis.py \
        --embeddings_dir $RUN_DIR/embeddings/trained \
        --model_type trained \
        --output_dir $RUN_DIR/embeddings/analysis")
echo "  Job ID: $JOB6"

# JOB 7: Generate text/table reports (depends on 5-6)
echo "[7/8] Generate text/table reports [depends on jobs 5-6]..."
JOB7=$(sbatch --parsable \
    --job-name=embed_reports \
    --time=1:00:00 \
    --dependency=afterok:$JOB5:$JOB6 \
    -c "cd $SCRIPT_DIR && python generate_coherence_report.py \
        --pretrain_results $SCRIPT_DIR/outputs/pretrained/embeddings/analysis/lrads_coherence_results.json \
        --trained_results $RUN_DIR/embeddings/analysis/lrads_coherence_results.json \
        --output_dir $SCRIPT_DIR/outputs/embeddings/reports")
echo "  Job ID: $JOB7"

# JOB 8: Final summary (depends on 7)
echo "[8/8] Final summary synthesis [depends on job 7]..."
JOB8=$(sbatch --parsable \
    --job-name=embed_summary \
    --time=30:00 \
    --dependency=afterok:$JOB7 \
    -c "cd $SCRIPT_DIR && python generate_final_summary.py \
        --reports_dir $SCRIPT_DIR/outputs/embeddings/reports \
        --output_file $SCRIPT_DIR/outputs/embeddings/COHERENCE_SUMMARY.md")
echo "  Job ID: $JOB8"

echo ""
echo "================================================================================"
echo "✅ ALL 8 JOBS SUBMITTED"
echo "================================================================================"
echo ""
echo "Job Dependencies:"
echo "  ┌─ $JOB1 (p1: layers 0-8, norm)     ──┐"
echo "  ├─ $JOB2 (p2: layers 9-16, heads)  ──┼─ $JOB5 (analysis) ──┐"
echo "  └─ $JOB3 (p3: layers 17-23, pool) ──┘                      │"
echo "                                                                ├─ $JOB7 (reports) ─ $JOB8 (summary)"
echo "  $JOB4 (trained final, all)  ─── $JOB6 (analysis) ──┘"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER | grep embed'"
echo ""
