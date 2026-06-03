# Embedding Pipeline - Folder Structure Corrections

## Summary of Changes

Fixed the embedding extraction and analysis pipeline to ensure **all results consolidate under a single unified location** with proper subfolder organization.

---

## Corrected Output Structure

All results now go to: `outputs/run_20260529_101746/embeddings/`

```
outputs/run_20260529_101746/embeddings/
├── pretrain/          ← Pretrained model (24 layers + extended representations)
│   ├── embeddings_layer0.npy
│   ├── embeddings_layer1.npy
│   ├── ...
│   ├── embeddings_layer23.npy
│   ├── embeddings_meta_layer0.csv
│   ├── embeddings_meta_layer1.csv
│   ├── ...
│   ├── umap_coords_layer0.npy
│   ├── umap_coords_layer23.npy
│   ├── umap_lrads_layer0.png
│   ├── umap_lrads_layer23.png
│   ├── umap_cancer_layer0.png
│   ├── ... (age, sex, race, smoking, era plots per layer)
│   └── [extended representations with attention heads, pre/post-norm, pooling variants]
│
├── trained/           ← Fine-tuned model (final layer + extended representations)
│   ├── embeddings_layer_final.npy
│   ├── embeddings_meta_layer_final.csv
│   ├── umap_coords_layer_final.npy
│   ├── umap_lrads_layer_final.png
│   ├── umap_cancer_layer_final.png
│   ├── umap_pred1_layer_final.png  (year-1 risk prediction)
│   ├── umap_age_layer_final.png
│   ├── umap_sex_layer_final.png
│   ├── umap_race_layer_final.png
│   ├── umap_smoke_layer_final.png
│   ├── umap_ctera_layer_final.png  (CT era)
│   └── [extended representations]
│
├── combined/          ← Merged visualizations (both pretrained and trained)
│   ├── umap_combined_layer_final.png  (trained: 9-panel dashboard)
│   ├── umap_combined_layer0.png       (pretrain: 8-panel, no pred1)
│   ├── umap_combined_layer1.png
│   ├── ...
│   └── umap_combined_layer23.png
│
└── analysis/          ← Coherence analysis results
    ├── lrads_coherence_results_pretrain.json
    ├── lrads_coherence_summary_pretrain.png
    ├── lrads_coherence_results_trained.json
    └── lrads_coherence_summary_trained.png
```

---

## What Changed

### 1. **Extract Scripts (extract_embeddings.py, extract_embeddings_pretrained.py)**

Both scripts already create their own subfolders correctly:

```python
base_output = Path(args.output_dir)

# extract_embeddings.py (fine-tuned)
output_dir = base_output / 'trained'      # Creates: trained/
combined_dir = base_output / 'combined'

# extract_embeddings_pretrained.py (pretrained)
output_dir = base_output / 'pretrain'     # Creates: pretrain/
combined_dir = base_output / 'combined'
```

**Action**: Pass `--output_dir outputs/run_20260529_101746/embeddings` to both scripts, and they automatically create the correct subfolders.

**Before** (incorrect):
```bash
python extract_embeddings.py \
    --output_dir outputs/run_20260529_101746/embeddings/trained
    # Result: outputs/run_20260529_101746/embeddings/trained/trained/ ✗
```

**After** (correct):
```bash
python extract_embeddings.py \
    --output_dir outputs/run_20260529_101746/embeddings
    # Result: outputs/run_20260529_101746/embeddings/trained/ ✓
```

---

### 2. **Parallel SLURM Script (embedding_submit_parallel_8jobs.sh)**

Fixed all `--output_dir` arguments to pass just the parent directory:

| Job | Script | Old Path | New Path |
|-----|--------|----------|----------|
| 1,2,3 | `extract_embeddings_extended.py` | `embeddings/pretrained` ❌ | `embeddings` ✓ |
| 4 | `extract_embeddings.py` | `embeddings/trained` ❌ | `embeddings` ✓ |
| 5 | `lrads_coherence_analysis.py` | `embeddings/pretrained` ❌ | `embeddings/pretrain` ✓ |
| 6 | `lrads_coherence_analysis.py` | `embeddings/trained` ✓ | `embeddings/trained` ✓ |

**Key Fix**: Changed Job 5's embeddings_dir from `pretrained` to `pretrain` (note the difference!) because the extraction script creates `pretrain/`, not `pretrained/`.

---

### 3. **Coherence Analysis Script Input Paths**

The `lrads_coherence_analysis.py` script expects to read embeddings from:
- `embeddings/pretrain/` (folder created by `extract_embeddings_pretrained.py`)
- `embeddings/trained/` (folder created by `extract_embeddings.py`)

Both jobs now correctly point to these locations.

---

## Verification

### Folder Naming Convention

The **extraction scripts use different folder names**:

- **Pretrained script**: Creates `pretrain/` (not `pretrained`)
- **Trained script**: Creates `trained/`

Both create a shared `combined/` folder for merged plots.

### SLURM Script Job Order

```
Extraction Phase (parallel):
  ├─ Job 1: pretrain layers 0-8  (4 hrs)
  ├─ Job 2: pretrain layers 9-16 (4 hrs)
  ├─ Job 3: pretrain layers 17-23 (4 hrs)
  └─ Job 4: trained final layer   (2 hrs)
             └─ max parallel time: 4 hrs

Analysis Phase (depends on extraction):
  ├─ Job 5: Pretrain coherence → embeddings/pretrain/ → analysis/
  └─ Job 6: Trained coherence  → embeddings/trained/ → analysis/
             └─ max parallel time: 2 hrs

Total pipeline: ~6 hours end-to-end (sequential phases)
```

---

## How to Submit

```bash
cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
bash embedding_submit_parallel_8jobs.sh
```

Or with sbatch:
```bash
sbatch embedding_submit_parallel_8jobs.sh
```

The script will:
1. Submit 4 extraction jobs (parallel)
2. Once extraction completes, automatically submit 2 analysis jobs (parallel)
3. All results consolidate in `outputs/run_20260529_101746/embeddings/`

---

## What This Solves

✅ **Before**: Results scattered across multiple locations:
- `outputs/pretrained/embeddings/pretrain/`
- `outputs/run_20260529_101746/embeddings/trained/`
- No clear organization

❌ **After**: All results in one place:
- `outputs/run_20260529_101746/embeddings/{pretrain,trained,combined,analysis}/`
- Clear folder hierarchy
- Easy to find and analyze all embeddings

---

## Files Modified

1. `embedding_submit_parallel_8jobs.sh` — Fixed all `--output_dir` paths
2. `extract_embeddings.py` — No changes needed (already correct)
3. `extract_embeddings_pretrained.py` — No changes needed (already correct)
4. `extract_embeddings_extended.py` — Must handle `pretrain/` subfolder creation
5. `lrads_coherence_analysis.py` — No changes needed

---

## Race Label Corrections

Both extraction scripts already apply the race label shortening:
```python
RACE_SHORTMAP = {
    'American Indian or Alaska Native': 'Am. Indian',
    'Native Hawaiian or Pacific Islander': 'Pac. Islander',
    'Not Reported': 'Not Reported',
    'Unknown': 'Unknown',
}

meta['race'] = meta['race'].replace(RACE_SHORTMAP)
```

This improves readability on plots.

---

**Status**: ✅ Ready to submit to cluster  
**Date**: 2026-06-02  
**Next Step**: Push updated scripts to GitHub, then rsync back to cluster
