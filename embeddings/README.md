# TANGERINE Embeddings Analysis

Layer-wise embedding extraction and analysis for TANGERINE 6-year lung cancer survival model.

## Files

- `extract_embeddings.py` - Extract embeddings from fine-tuned model (all 24 layers)
- `extract_embeddings_pretrained.py` - Extract embeddings from pretrained model (all 24 layers)
- `lrads_layer_analysis.py` - Analyze which layer best captures LRADS risk stratification
- `layer_probe_analysis.py` - Probe what clinical variables are encoded in each layer
- `replot_embeddings.py` - Generate UMAP visualizations
- `embedding_submit.sh` - SLURM job submission script
- `run_embeddings.sh`, `run_embeddings_pretrained.sh` - Local run scripts

## Output Structure

```
embeddings/
├── trained/      - Fine-tuned model embeddings + UMAPs
├── pretrain/     - Pretrained model embeddings + UMAPs  
└── combined/     - Combined visualization plots
```
