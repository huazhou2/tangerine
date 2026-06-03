# Code Modifications for Next Run

## Changes to Apply to extract_embeddings.py and extract_embeddings_pretrained.py

### 1. Add race label shortening (top of file)

```python
# Race label shortening for better plot display
RACE_SHORTMAP = {
    'American Indian or Alaska Native': 'Am. Indian',
    'Native Hawaiian or Pacific Islander': 'Pac. Islander',
    'Not Reported': 'Not Reported',
    'Unknown': 'Unknown',
}
```

### 2. Modify output_dir setup (around line 265)

**BEFORE:**
```python
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
```

**AFTER:**
```python
# Determine subfolder based on extraction type
model_type = 'trained' if not args.pretrained else 'pretrain'
output_dir = Path(args.output_dir) / model_type
output_dir.mkdir(parents=True, exist_ok=True)
combined_dir = Path(args.output_dir) / 'combined'
combined_dir.mkdir(parents=True, exist_ok=True)
```

### 3. After loading metadata (around line 365)

**ADD THIS:**
```python
# Shorten race labels for better plot display
meta['lrads_category'] = meta['lrads_category'].replace(RACE_SHORTMAP)
```

### 4. When saving combined plots (around line 437)

**CHANGE:**
```python
fig.savefig(output_dir / f'umap_combined_{layer_tag}.png', dpi=150)
```

**TO:**
```python
# Save combined plots to shared 'combined' folder
fig.savefig(combined_dir / f'umap_combined_{layer_tag}.png', dpi=150)
```

## Result

After modifications, embeddings will automatically save to:

```
embeddings/
├── trained/
│   ├── embeddings_layer*.npy
│   ├── embeddings_meta_layer*.csv
│   └── umap_*_layer_final.png
├── pretrain/
│   ├── embeddings_layer0-23.npy
│   ├── embeddings_meta_layer0-23.csv
│   └── umap_*_layer0-23.png
└── combined/
    ├── umap_combined_layer_final.png
    └── umap_combined_layer0-23.png
```

All race labels automatically shortened (American Indian → Am. Indian, etc.)
