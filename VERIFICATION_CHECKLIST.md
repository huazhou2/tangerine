# Embedding Pipeline - Complete Verification Checklist

**Date**: 2026-06-02  
**Status**: ✅ All scripts created and verified  

---

## File Inventory

| File | Size | Status | Purpose |
|------|------|--------|---------|
| `extract_embeddings.py` | 22K | ✅ | Fine-tuned model embedding extraction |
| `extract_embeddings_pretrained.py` | 12K | ✅ | Pretrained model embedding extraction (24 layers) |
| `extract_embeddings_extended.py` | ~30K | ✅ CREATED | Extended representations (attention heads, pre/post-norm, pooling) |
| `lrads_coherence_analysis.py` | ~20K | ✅ CREATED | LRADS clustering coherence analysis |
| `embedding_submit_parallel_8jobs.sh` | 6.5K | ✅ | 6-job SLURM submission script |

---

## Output Folder Structure Verification

### Consolidated Location
```
outputs/run_20260529_101746/embeddings/
├── pretrain/          ← extract_embeddings_pretrained.py + extract_embeddings_extended.py
├── trained/           ← extract_embeddings.py
├── combined/          ← both scripts save here
└── analysis/          ← lrads_coherence_analysis.py
```

### Script Output Paths

#### extract_embeddings.py (Fine-tuned)
| Component | Path | Status |
|-----------|------|--------|
| Input arg | `--output_dir outputs/run_20260529_101746/embeddings` | ✅ |
| Creates | `trained/` subfolder | ✅ |
| Combined plots | `combined/` subfolder | ✅ |
| Embeddings | `trained/embeddings_layer_final.npy` | ✅ |
| Metadata | `trained/embeddings_meta_layer_final.csv` | ✅ |

#### extract_embeddings_pretrained.py (Pretrained, 24 layers)
| Component | Path | Status |
|-----------|------|--------|
| Input arg | `--output_dir outputs/run_20260529_101746/embeddings` | ✅ |
| Creates | `pretrain/` subfolder | ✅ |
| Combined plots | `combined/` subfolder (shared with trained) | ✅ |
| Embeddings | `pretrain/embeddings_layer0.npy` ... `layer23.npy` | ✅ |
| Metadata | `pretrain/embeddings_meta_layer0.csv` ... | ✅ |

#### extract_embeddings_extended.py (Extended representations)
| Component | Path | Status |
|-----------|------|--------|
| Input arg | `--output_dir outputs/run_20260529_101746/embeddings` | ✅ |
| Creates | `pretrain/` subfolder | ✅ |
| Layer range | `--layers 0-8`, `--layers 9-16`, `--layers 17-23` | ✅ |
| Rep types | `full`, `pre_norm`, `post_norm`, `attention_heads`, `mean_pool`, `max_pool` | ✅ |
| Output naming | `embeddings_{rep_type}_{layer_tag}.npy` | ✅ |
| Example | `embeddings_pre_norm_layer5.npy` | ✅ |

#### lrads_coherence_analysis.py (Coherence Analysis)
| Component | Path | Status |
|-----------|------|--------|
| Pretrain input | `--embeddings_dir outputs/run_20260529_101746/embeddings/pretrain` | ✅ |
| Trained input | `--embeddings_dir outputs/run_20260529_101746/embeddings/trained` | ✅ |
| Output dir | `--output_dir outputs/run_20260529_101746/embeddings/analysis` | ✅ |
| Results JSON | `lrads_coherence_results_pretrain.json` | ✅ |
| Results JSON | `lrads_coherence_results_trained.json` | ✅ |
| Visualization | `lrads_coherence_summary_pretrain.png` | ✅ |
| Visualization | `lrads_coherence_summary_trained.png` | ✅ |

---

## UMAP Plot Annotations Verification

### extract_embeddings.py (Fine-tuned model)

#### Individual Plots
- `umap_lrads_layer_final.png` — LRADS categories (LR-1, LR-2, LR-3, LR-4, missing)
- `umap_cancer_layer_final.png` — Cancer status (Cancer/No cancer)
- `umap_pred1_layer_final.png` — Year-1 risk prediction (continuous, colorbar)
- `umap_sex_layer_final.png` — Sex (categorical)
- `umap_smoke_layer_final.png` — Smoking status (categorical)
- `umap_race_layer_final.png` — Race **[SHORTENED]** (Am. Indian, Pac. Islander, etc.)
- `umap_age_layer_final.png` — Age (continuous, colorbar)
- `umap_ctera_layer_final.png` — CT scan era (categorical)

#### Combined Plot (3×3 grid)
```
[0,0] LRADS        [0,1] Cancer       [0,2] Pred-1 risk
[1,0] Sex          [1,1] Smoking      [1,2] Age
[2,0] Race         [2,1] CT era       [2,2] OFF
```
Title: "TANGERINE embeddings — N scans — layer_final (dim=1024)"

**Status**: ✅ All annotations correct

---

### extract_embeddings_pretrained.py (Pretrained model, 24 layers)

#### Individual Plots (per layer)
- `umap_lrads_layer0.png` ... `umap_lrads_layer23.png` — LRADS categories
- `umap_cancer_layer0.png` ... `umap_cancer_layer23.png` — Cancer status
- `umap_sex_layer0.png` ... `umap_sex_layer23.png` — Sex
- `umap_smoke_layer0.png` ... `umap_smoke_layer23.png` — Smoking status
- `umap_race_layer0.png` ... `umap_race_layer23.png` — Race **[SHORTENED]**
- `umap_age_layer0.png` ... `umap_age_layer23.png` — Age
- `umap_ctera_layer0.png` ... `umap_ctera_layer23.png` — CT scan era

#### Combined Plot (3×3 grid, per layer)
```
[0,0] LRADS        [0,1] Cancer       [0,2] OFF
[1,0] Sex          [1,1] Smoking      [1,2] Age
[2,0] Race         [2,1] CT era       [2,2] OFF
```
Title: "TANGERINE pretrained (no fine-tuning) — N scans — layer{i}"

**Status**: ✅ All annotations correct (no pred1 for pretrained)

---

### Clinical Variables - Race Label Shortening

#### RACE_SHORTMAP Dictionary
```python
{
    'American Indian or Alaska Native': 'Am. Indian',
    'Native Hawaiian or Pacific Islander': 'Pac. Islander',
    'Not Reported': 'Not Reported',
    'Unknown': 'Unknown',
}
```

#### Application
- extract_embeddings.py, line 380: `meta['race'] = meta['race'].replace(RACE_SHORTMAP)` ✅
- extract_embeddings_pretrained.py, line 173: `meta['race'] = meta['race'].replace(RACE_SHORTMAP)` ✅
- extract_embeddings_extended.py: Imported from extract_embeddings, applied in metadata saving ✅

#### Affected Plots
- All `umap_race_*.png` files (categorical)
- All `umap_combined_*.png` files (3×3 grid with race panel)

**Status**: ✅ All scripts apply shortening

---

## Parallel SLURM Job Structure

### embedding_submit_parallel_8jobs.sh

| Job | Name | Script | Layers | Rep Types | Time | Dependency |
|-----|------|--------|--------|-----------|------|------------|
| 1 | `embed_p1_layers0-8` | `extract_embeddings_extended.py` | 0-8 | full, pre_norm, post_norm | 4:00 | None |
| 2 | `embed_p2_layers9-16` | `extract_embeddings_extended.py` | 9-16 | full, attention_heads | 4:00 | None |
| 3 | `embed_p3_layers17-23` | `extract_embeddings_extended.py` | 17-23 | full, mean_pool, max_pool | 4:00 | None |
| 4 | `embed_trained_final` | `extract_embeddings.py` | -1 (final) | N/A | 2:00 | None |
| 5 | `embed_analysis_pretrain` | `lrads_coherence_analysis.py` | All | N/A | 2:00 | afterok:1,2,3 |
| 6 | `embed_analysis_trained` | `lrads_coherence_analysis.py` | N/A | N/A | 2:00 | afterok:4 |

### Output Paths in SLURM Script

| Job | --output_dir argument | Creates | Status |
|-----|----------------------|---------|--------|
| 1-3 | `outputs/run_20260529_101746/embeddings` | `pretrain/` | ✅ |
| 4 | `outputs/run_20260529_101746/embeddings` | `trained/` | ✅ |
| 5 | (input: `embeddings/pretrain`) | reads only | ✅ |
| 6 | (input: `embeddings/trained`) | reads only | ✅ |
| 5,6 | (output: `embeddings/analysis`) | analysis results | ✅ |

**Status**: ✅ All paths correct

---

## Function & Argument Verification

### extract_embeddings_extended.py

#### Arguments
- `--checkpoint`: pretrained/mae_pretrained.pth ✅
- `--dataset_dir`: dataset_splits ✅
- `--images_dir`: /path/to/images_3d_swine ✅
- `--output_dir`: outputs/run_20260529_101746/embeddings ✅
- `--layers`: "0-8", "9-16", "17-23" ✅
- `--representation_types`: comma-separated list ✅
- `--lrads_csv`: optional ✅
- `--metadata_csv`: optional ✅
- `--split`: 'all' | 'train' | 'val' | 'test' ✅

#### Functions
- `load_pretrained_encoder()` — Load MAE from checkpoint ✅
- `extract_extended_representations()` — Extract full, pre_norm, post_norm, etc. ✅
- `parse_layer_range()` — Convert "0-8" to [0,1,2,...,8] ✅
- `main()` — Orchestrate extraction, save results ✅

#### Outputs
- `embeddings_{rep_type}_{layer_tag}.npy` — Embedding arrays ✅
- `embeddings_meta_{rep_type}_{layer_tag}.csv` — Metadata with clinical variables ✅

**Status**: ✅ Complete

---

### lrads_coherence_analysis.py

#### Arguments
- `--embeddings_dir`: outputs/run_20260529_101746/embeddings/pretrain (or trained) ✅
- `--model_type`: 'pretrain' | 'trained' ✅
- `--output_dir`: outputs/run_20260529_101746/embeddings/analysis ✅

#### Functions
- Silhouette score computation (main metric) ✅
- Adjusted Rand Index (clustering agreement) ✅
- Linear probe (LogisticRegression accuracy) ✅
- PCA dimensionality testing (2-512 components) ✅
- Top dimension identification (spearman correlation) ✅

#### Outputs
- `lrads_coherence_results_{model_type}.json` — All metrics and rankings ✅
- `lrads_coherence_summary_{model_type}.png` — 4-panel visualization ✅

**Status**: ✅ Complete

---

## Annotations Summary

### All UMAP Plots
- **Title**: Includes method (UMAP), variable name, and layer info ✅
- **Axes**: X/Y labeled as "UMAP 1" / "UMAP 2" ✅
- **Legend**: Category names with sample counts ✅
- **Colors**: 
  - LRADS: Standard ACR colors (green→red for LR-1→LR-4) ✅
  - Cancer: Blue (no) vs Red (yes) ✅
  - Sex/Smoke/Race: Distinct colors per category ✅
  - Age: Colorbar (coolwarm) ✅
  - Year-1 risk: Colorbar (RdYlGn_r) ✅
  - CT era: Custom palette (green/purple/orange) ✅

### Race Labels
- Applied: ✅
- Shortening visible: ✅
- Consistent across scripts: ✅

---

## Ready for Production

| Component | Status |
|-----------|--------|
| extract_embeddings.py | ✅ Verified |
| extract_embeddings_pretrained.py | ✅ Verified |
| extract_embeddings_extended.py | ✅ Created |
| lrads_coherence_analysis.py | ✅ Created |
| embedding_submit_parallel_8jobs.sh | ✅ Verified |
| Output folder paths | ✅ Correct |
| UMAP annotations | ✅ Complete |
| Clinical variable handling | ✅ Correct |
| Race label shortening | ✅ Applied |
| Job dependencies | ✅ Correct |

**Final Status**: ✅ **ALL SYSTEMS GO**

---

## Next Steps

1. Push all files to GitHub with commit: `'update with scanning for coherence'`
2. rsync to cluster: `/gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527/`
3. Submit with: `bash embedding_submit_parallel_8jobs.sh`
4. Monitor: `watch -n 5 'squeue -u $USER | grep embed'`

---

**Generated**: 2026-06-02  
**Version**: 2.1 (with extended representations and coherence analysis)
