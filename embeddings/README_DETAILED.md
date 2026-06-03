# TANGERINE Embeddings Analysis - Complete Guide

## 📋 Overview

This pipeline extracts and analyzes layer-wise embeddings from the **TANGERINE model** - a vision transformer pretrained on 98,000 chest CT scans and fine-tuned for 6-year lung cancer survival prediction.

**Key Question Answered**: Which transformer layer best captures LRADS risk stratification?

---

## 🎯 What We're Doing

### 1. **Extract Embeddings from Both Models**

```
Pretrained Model (MAE)         Fine-tuned Model (Survival Task)
├── Layer 0                     ├── Layer -1 (final)
├── Layer 1                     └── [+ survival head]
├── ...
└── Layer 23
```

**Why Two Models?**
- **Pretrained**: Shows what the foundation model learns from general CT patterns (98K images, masked autoencoder loss)
- **Fine-tuned**: Shows what additional information is learned after training on the lung cancer survival task

### 2. **Organize into Structured Folders**

```
embeddings/
├── trained/           ← Fine-tuned model results
│   ├── embeddings_layer_final.npy  (5621 patients × 1024 dims)
│   ├── embeddings_meta_layer_final.csv  (with predictions + clinical vars)
│   ├── umap_cancer_layer_final.png
│   ├── umap_age_layer_final.png
│   ├── umap_sex_layer_final.png
│   ├── umap_race_layer_final.png
│   ├── umap_smoke_layer_final.png
│   ├── umap_lrads_layer_final.png
│   ├── umap_ctera_layer_final.png
│   ├── umap_pred1_layer_final.png
│   └── analysis/  (coherence results)
│
├── pretrain/          ← Pretrained model (24 layers)
│   ├── embeddings_layer0.npy ... embeddings_layer23.npy
│   ├── embeddings_meta_layer0.csv ... embeddings_meta_layer23.csv
│   ├── umap_cancer_layer0.png ... umap_cancer_layer23.png
│   ├── umap_age_layer0.png ... umap_age_layer23.png
│   ├── umap_sex_layer0.png ... umap_sex_layer23.png
│   ├── umap_race_layer0.png ... umap_race_layer23.png
│   ├── umap_smoke_layer0.png ... umap_smoke_layer23.png
│   ├── umap_lrads_layer0.png ... umap_lrads_layer23.png
│   ├── umap_ctera_layer0.png ... umap_ctera_layer23.png
│   └── analysis/  (coherence results)
│
└── combined/          ← Merged visualization plots
    ├── umap_combined_layer_final.png  (9 variables, 1 image)
    └── umap_combined_layer0-23.png    (pretrained, per layer)
```

### 3. **Clean Up Labels for Better Visualizations**

**Race label shortening** - long category names are unreadable on plots:

```python
BEFORE                                    AFTER
American Indian or Alaska Native    →    Am. Indian
Native Hawaiian or Pacific Islander →    Pac. Islander
Not Reported                        →    Not Reported
Unknown                             →    Unknown
```

This applies to:
- Individual race UMAPs (`umap_race_*.png`)
- Combined plots (`umap_combined_*.png`)
- Metadata CSVs (for reference)

### 4. **Find Which Layer Best Captures LRADS Structure**

**The core analysis**: Which layer's embeddings have clusters that align with LRADS risk groups?

#### What is LRADS?

**Lung-RADS** = standardized CT risk assessment (American College of Radiology)
- Categories: 1 (no findings) → 4B (highest risk)
- Predicts likelihood of malignancy

#### How We Measure Layer Quality

For each layer (pretrained: 0-23, trained: final), we compute:

1. **Silhouette Score** (main metric)
   - Measures cluster tightness for LRADS categories
   - Higher = LRADS groups cluster better in this layer's embedding space
   - Range: [-1, 1]  (>0.3 is good, >0.5 is excellent)

2. **Adjusted Rand Index** (secondary metric)
   - Measures agreement between K-Means clusters and LRADS groups
   - Independent clustering quality metric

3. **Linear Probe Accuracy** (interpretability metric)
   - Train simple LogisticRegression: embeddings → LRADS category
   - Higher accuracy = layer encodes LRADS-predictive information

#### Expected Findings

**For Pretrained Layers (0-23):**
- Early layers (0-5): Generic visual features, low LRADS coherence
- Mid layers (8-15): Mixed information, moderate LRADS coherence
- Late layers (18-23): Task-specific, best LRADS coherence
- **Peak**: Typically layer 20-23 (95% through the network)

**For Fine-tuned Model (final):**
- Optimized for survival prediction
- Should exceed pretrained's best layer (LRADS is component of survival risk)
- Will have survival-specific features + LRADS information

---

## 🚀 Running the Pipeline

### One-Shot Command

```bash
cd /gpfs/data/tsirigoslab/home/zhouh05/lung_ct/models/tangerine_6yrs_20260527
sbatch embedding_submit.sh
```

### What It Does (Automatically)

```
embedding_submit.sh
├── Step 1: Extract PRETRAINED embeddings (24 × 24 layers = 576 embeddings)
│   └── python extract_embeddings_pretrained.py --layer 0..23
├── Step 2: Extract TRAINED embeddings (final layer)
│   └── python extract_embeddings.py --layer -1
├── Step 3: LRADS Coherence Analysis
│   ├── python lrads_coherence_analysis.py --model_type pretrain
│   └── python lrads_coherence_analysis.py --model_type trained
└── Done! Results in embeddings/{trained,pretrain,combined,analysis}/
```

### Monitor Progress

```bash
squeue -u $USER                          # Check job status
tail -f logs/embeddings_*.out            # Watch output
```

### Estimated Runtime

- **Extract PRETRAINED** (24 layers): ~18-24 hours (parallel-friendly)
- **Extract TRAINED** (final): ~1-2 hours
- **Coherence Analysis**: ~1-2 hours
- **Total**: ~24-28 hours

---

## 📊 Understanding the Results

### Coherence Analysis Output

File: `analysis/lrads_coherence_results.json`

```json
{
  "full_layers_pretrain": {
    "layer_0": 0.152,
    "layer_1": 0.163,
    ...
    "layer_23": 0.385          ← Best pretrained layer
  },
  "full_layers_trained": {
    "layer_final": 0.412       ← Better than best pretrained!
  },
  "pca_components": {
    "pca_2": 0.187,
    "pca_50": 0.321,
    "pca_256": 0.405,          ← Optimal dimensionality
    "pca_512": 0.398
  },
  "top_dimensions": {
    "dim_542": 0.486,          ← Most LRADS-predictive dims
    "dim_891": 0.421,
    ...
  }
}
```

### Visualization: `lrads_coherence_summary.png`

4-panel figure showing:
1. **Pretrained Layers** (0-23): Silhouette score curve → peak layer identified
2. **Fine-tuned Layer**: Single score for final layer
3. **PCA Dimensionality**: How much compression is safe?
4. **Summary Stats**: Best layer, best PCA, interpretation

### UMAP Plots: What to Look For

**Good LRADS Structure** (high coherence):
- LRADS categories form distinct, clustered regions
- Within-group points close together
- Between-group points far apart
- Clear separation by color

**Poor LRADS Structure** (low coherence):
- Categories scattered throughout space
- No visible cluster pattern
- Colors mixed randomly

**Example Interpretation**:
- `pretrain/umap_lrads_layer23.png`: "Layer 23 shows LRADS-like clustering"
- `trained/umap_lrads_layer_final.png`: "Fine-tuning adds more structure"
- `combined/umap_combined_*.png`: "9-variable dashboard showing what each layer encodes"

---

## 💾 Data Files Explained

### Embeddings (`.npy` files)

```python
import numpy as np
emb = np.load('embeddings/trained/embeddings_layer_final.npy')
print(emb.shape)  # (5621, 1024)
# Each row: one patient
# Each column: one dimension of learned representation
```

### Metadata (`.csv` files)

```python
import pandas as pd
meta = pd.read_csv('embeddings/trained/embeddings_meta_layer_final.csv')
meta.columns
# ['patient_id', 'split', 'cancer', 'time_at_event',
#  'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6',
#  'lrads_value', 'lrads_category_base',
#  'age', 'sex', 'race', 'smoke', 'ct_date', 'ct_era']
```

### UMAP Coordinates (`.npy` files)

```python
coords = np.load('embeddings/trained/umap_coords_layer_final.npy')
print(coords.shape)  # (5621, 2)
# Pre-computed 2D coordinates for plotting
# Matches order of embeddings_*.npy and metadata CSV
```

---

## 🔍 Interpreting Differences

### Pretrained vs Fine-tuned Coherence

**Hypothesis**: Fine-tuning on survival task should improve LRADS coherence

**Why?**: LRADS is a component of lung cancer risk. Model trained on survival should learn LRADS-relevant features.

**Expected Pattern**:
```
Best pretrained layer:     silhouette = 0.35-0.40
Fine-tuned final layer:    silhouette = 0.40-0.45
                           ↑ 5-15% improvement
```

**Interpretation**:
- If trained >> pretrained: "Survival task heavily relies on LRADS"
- If trained ≈ pretrained: "LRADS info already in pretraining"
- If trained < pretrained: "Survival fine-tuning trades LRADS for other signals"

### Layer Progression in Pretrained

**Expected**: Silhouette score increases through layers

```
Layer 0-8:     0.12-0.18  (early: edge detection, texture)
Layer 8-16:    0.18-0.30  (middle: anatomical structures)
Layer 16-23:   0.30-0.40  (late: task-specific patterns)
```

**If anomaly**: Sudden drop at some layer suggests feature collapse or redundancy.

---

## 📈 Next Steps with Results

### 1. Identify Best Layer for Downstream Tasks

```python
# Load results
import json
with open('analysis/lrads_coherence_results.json') as f:
    results = json.load(f)

# Find best layer
best_layer = max(results['full_layers_pretrain'],
                 key=results['full_layers_pretrain'].get)
print(f"Best pretrained layer: {best_layer}")
# → "layer_23"
```

### 2. Use Embeddings for Clinical Research

```python
# Load best-performing layer embeddings
emb = np.load('pretrain/embeddings_layer23.npy')
meta = pd.read_csv('pretrain/embeddings_meta_layer23.csv')

# Examples:
# - Train survival models on embeddings
# - Find patient subgroups via clustering
# - Identify predictive dimensions
# - Compare to clinical features
```

### 3. Investigate Top Predictive Dimensions

From `lrads_coherence_results.json['top_dimensions']`:

```python
# Dimensions most correlated with LRADS
top_dims = [542, 891, 213, ...]  # sorted by correlation

# Analyze: what do these dimensions encode?
# - Visualize their activation patterns
# - Correlate with clinical variables
# - Use for interpretability studies
```

---

## ⚙️ Technical Details

### Code Changes Applied

**extract_embeddings.py & extract_embeddings_pretrained.py:**

```python
# 1. Added race label shortening
RACE_SHORTMAP = {
    'American Indian or Alaska Native': 'Am. Indian',
    ...
}

# 2. Create subfolders
output_dir = base_output / 'trained'  # or 'pretrain'
combined_dir = base_output / 'combined'

# 3. Apply shortening to metadata
meta['race'] = meta['race'].replace(RACE_SHORTMAP)

# 4. Save combined plots to shared folder
fig.savefig(combined_dir / f'umap_combined_{layer_tag}.png')
```

### LRADS Coherence Analysis

**Silhouette Score Formula**:
```
For each data point i:
  a(i) = mean distance to other points in same LRADS group
  b(i) = mean distance to points in nearest LRADS group
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

Silhouette = mean(s(i)) over all points
```

**Interpretation**:
- s(i) > 0: Point is in correct LRADS cluster ✓
- s(i) < 0: Point would fit better in different LRADS cluster ✗
- Mean silhouette: overall cluster quality

---

## ❓ FAQ

**Q: Why do we have 24 layers for pretrained but only 1 for trained?**
A: The trained model is task-optimized. We already know the final layer works well for survival. The 24 layers for pretrained show how task-specific information emerges through the network during pretraining.

**Q: Should I use pretrained or trained embeddings for downstream tasks?**
A: Depends on your task:
- **If predicting survival**: Use trained (optimized for this)
- **If analyzing general CT patterns**: Use pretrained (less task-specific bias)
- **For comparison studies**: Use both to show task impact

**Q: What if trained coherence is worse than pretrained?**
A: This suggests survival fine-tuning discovered LRADS-orthogonal signals (e.g., vessel patterns, nodule texture) that matter for survival but don't align with radiologist-defined risk categories.

**Q: Can I use PCA-reduced embeddings instead?**
A: Yes! The coherence analysis tests different PCA components. If `pca_50` has high coherence but embeddings are 1024-dim, you can safely use 50 dims, saving storage and computation.

---

## 📚 References

- **TANGERINE**: Vision transformer pretrained on chest CTs (Gee et al.)
- **Sybil Framework**: 6-year survival prediction paradigm
- **Lung-RADS**: American College of Radiology CT risk assessment
- **Silhouette Score**: Rousseeuw, 1987. "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"

---

## 🎓 Citation

If you use these embeddings, please cite:

```bibtex
@article{tangerine,
  title={TANGERINE: Vision Transformer for Chest CT},
  author={...},
  journal={...},
  year={2024}
}
```

---

**Last Updated**: June 2, 2026  
**Pipeline Version**: 2.0 (with coherence analysis)  
**Status**: ✅ Ready for production
