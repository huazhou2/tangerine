"""
Analyze which layers best capture LRADS (Lung-RADS) risk stratification.

Measures: silhouette score of LRADS categories in embedding space.
Higher score = layer embeddings cluster LRADS categories better.

Reads embeddings from pretrain/ or trained/ folders and identifies:
- Best single layer for LRADS representation
- Top predictive dimensions
- Optimal PCA dimensionality

Usage:
    python lrads_coherence_analysis.py \
        --embeddings_dir outputs/run_20260529_101746/embeddings/pretrain \
        --model_type pretrain \
        --output_dir outputs/run_20260529_101746/embeddings/analysis
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm


def main(args):
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing coherence of LRADS in {embeddings_dir}")
    print(f"Model type: {args.model_type}")

    # Collect all embedding files and their metadata
    emb_files = sorted(embeddings_dir.glob('embeddings_*.npy'))
    meta_files = sorted(embeddings_dir.glob('embeddings_meta_*.csv'))

    if not emb_files:
        print(f"ERROR: No embeddings found in {embeddings_dir}")
        return

    print(f"Found {len(emb_files)} embedding files")

    results = defaultdict(dict)

    # Process each embedding file
    for emb_file in tqdm(emb_files, desc='Processing embeddings'):
        # Extract layer and representation type from filename
        # Format: embeddings_{rep_type}_{layer_tag}.npy OR embeddings_{layer_tag}.npy
        fname_parts = emb_file.stem.replace('embeddings_', '').split('_')

        # Try to parse: embeddings_rep_type_layerN.npy
        if len(fname_parts) >= 2 and fname_parts[-1].startswith('layer'):
            rep_type = '_'.join(fname_parts[:-1])
            layer_tag = fname_parts[-1]
        else:
            # Fallback: no explicit rep_type, just layer
            rep_type = 'full'
            layer_tag = '_'.join(fname_parts)

        # Load embeddings
        emb = np.load(emb_file)
        if emb.shape[0] == 0:
            continue

        # Find matching metadata file
        meta_file = emb_file.parent / f'embeddings_meta_{emb_file.name[11:]}'  # Remove 'embeddings_' prefix
        if not meta_file.exists():
            print(f"  Warning: no metadata for {emb_file.name}")
            continue

        meta = pd.read_csv(meta_file)

        # Get LRADS categories
        if 'lrads_category_base' not in meta.columns:
            print(f"  Warning: no lrads_category_base in {meta_file.name}")
            continue

        lrads = meta['lrads_category_base'].values
        valid_idx = ~np.isnan(lrads)

        if valid_idx.sum() < 10:
            print(f"  Skipping {layer_tag}: only {valid_idx.sum()} valid LRADS samples")
            continue

        emb_valid = emb[valid_idx]
        lrads_valid = lrads[valid_idx].astype(int)

        # Standardize embeddings
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_valid)

        # 1. Silhouette score (main metric)
        try:
            sil_score = silhouette_score(emb_scaled, lrads_valid, metric='cosine', sample_size=min(5000, len(emb_valid)))
        except:
            sil_score = np.nan

        # 2. Adjusted Rand Index (K-means agreement)
        try:
            n_clusters = len(np.unique(lrads_valid))
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=100)
            clusters = kmeans.fit_predict(emb_scaled)
            ari_score = adjusted_rand_index(clusters, lrads_valid)
        except:
            ari_score = np.nan

        # 3. Linear probe (logistic regression)
        try:
            lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
            lr.fit(emb_scaled, lrads_valid)
            linear_acc = lr.score(emb_scaled, lrads_valid)
        except:
            linear_acc = np.nan

        # Store results
        layer_key = f'{rep_type}_{layer_tag}' if rep_type != 'full' else layer_tag
        results[layer_key] = {
            'silhouette': float(sil_score) if not np.isnan(sil_score) else None,
            'adjusted_rand_index': float(ari_score) if not np.isnan(ari_score) else None,
            'linear_probe_accuracy': float(linear_acc) if not np.isnan(linear_acc) else None,
            'n_samples': int(valid_idx.sum()),
        }

        print(f"  {layer_key}: sil={sil_score:.3f}, ari={ari_score:.3f}, acc={linear_acc:.3f}")

    # Test PCA dimensionality reduction
    print("\nTesting PCA dimensionality...")
    pca_results = {}

    # Load best full-layer embedding for PCA testing
    best_file = None
    best_score = -np.inf
    for emb_file in emb_files:
        if 'full_' not in emb_file.name and not emb_file.name.endswith('full.npy'):
            continue
        meta_file = emb_file.parent / f'embeddings_meta_{emb_file.name[11:]}'
        if not meta_file.exists():
            continue
        meta = pd.read_csv(meta_file)
        if 'lrads_category_base' not in meta.columns:
            continue
        lrads = meta['lrads_category_base'].values
        valid_idx = ~np.isnan(lrads)
        if valid_idx.sum() > best_score:
            best_score = valid_idx.sum()
            best_file = emb_file

    if best_file is not None:
        emb = np.load(best_file)
        meta_file = best_file.parent / f'embeddings_meta_{best_file.name[11:]}'
        meta = pd.read_csv(meta_file)
        lrads = meta['lrads_category_base'].values
        valid_idx = ~np.isnan(lrads)

        emb_valid = emb[valid_idx]
        lrads_valid = lrads[valid_idx].astype(int)
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb_valid)

        for n_components in [2, 5, 10, 25, 50, 100, 256, 512]:
            if n_components >= emb_scaled.shape[1]:
                continue
            pca = PCA(n_components=n_components)
            emb_pca = pca.fit_transform(emb_scaled)
            try:
                sil = silhouette_score(emb_pca, lrads_valid, metric='cosine', sample_size=min(5000, len(emb_pca)))
                pca_results[f'pca_{n_components}'] = float(sil)
            except:
                pass

    # Identify top dimensions for original embeddings
    top_dims = {}
    if best_file is not None:
        # Compute correlation of each dimension with LRADS
        from scipy.stats import spearmanr
        corrs = []
        for dim in range(min(1024, emb_valid.shape[1])):
            rho, _ = spearmanr(emb_valid[:, dim], lrads_valid)
            corrs.append(abs(rho))

        top_dim_indices = np.argsort(corrs)[-10:][::-1]  # Top 10
        for i, dim_idx in enumerate(top_dim_indices[:5]):  # Top 5 only
            top_dims[f'dim_{dim_idx}'] = float(corrs[dim_idx])

    # Save comprehensive results
    all_results = {
        'model_type': args.model_type,
        'layers': results,
        'pca_components': pca_results,
        'top_dimensions': top_dims,
    }

    results_file = output_dir / f'lrads_coherence_results_{args.model_type}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate summary visualization
    layer_names = sorted(results.keys())
    layer_scores = [results[ln]['silhouette'] or 0 for ln in layer_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Layer scores
    ax = axes[0, 0]
    colors = ['green' if s > 0.3 else 'orange' if s > 0.2 else 'red' for s in layer_scores]
    ax.bar(range(len(layer_names)), layer_scores, color=colors, alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'LRADS Coherence by Layer ({args.model_type})')
    ax.set_xticks(range(0, len(layer_names), max(1, len(layer_names)//10)))
    ax.set_xticklabels([layer_names[i] for i in range(0, len(layer_names), max(1, len(layer_names)//10))],
                       rotation=45, ha='right', fontsize=8)
    ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Good (0.3)')
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Fair (0.2)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: PCA dimensionality
    ax = axes[0, 1]
    if pca_results:
        pca_names = sorted(pca_results.keys(), key=lambda x: int(x.split('_')[1]))
        pca_scores = [pca_results[pn] for pn in pca_names]
        ax.plot(range(len(pca_names)), pca_scores, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('PCA Components')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('PCA Dimensionality Analysis')
        ax.set_xticks(range(len(pca_names)))
        ax.set_xticklabels([pn.split('_')[1] for pn in pca_names], rotation=45)
        ax.grid(True, alpha=0.3)

    # Panel 3: Summary stats
    ax = axes[1, 0]
    ax.axis('off')

    best_layer = max(results.items(), key=lambda x: x[1]['silhouette'] or -np.inf)[0]
    best_score = results[best_layer]['silhouette']

    best_pca = max(pca_results.items(), key=lambda x: x[1])[0] if pca_results else 'N/A'
    best_pca_score = pca_results.get(best_pca, 0) if pca_results else 0

    summary_text = f"""
SUMMARY

Model: {args.model_type}
Total layers analyzed: {len(results)}

Best layer: {best_layer}
  Silhouette: {best_score:.3f}

Interpretation:
  > 0.5: Excellent LRADS coherence
  0.3–0.5: Good coherence
  0.2–0.3: Fair coherence
  < 0.2: Weak coherence

Best PCA: {best_pca}
  Score: {best_pca_score:.3f}

Top dimensions (most LRADS-correlated):
{json.dumps(top_dims, indent=2) if top_dims else 'N/A'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 4: Distribution of scores
    ax = axes[1, 1]
    ax.hist([s for s in layer_scores if s is not None], bins=20, alpha=0.7, color='steelblue')
    ax.axvline(x=np.mean(layer_scores), color='r', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Silhouette Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Layer Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    viz_file = output_dir / f'lrads_coherence_summary_{args.model_type}.png'
    fig.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {viz_file}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"BEST LAYER FOR LRADS COHERENCE ({args.model_type})")
    print(f"{'='*70}")
    print(f"  {best_layer}: silhouette = {best_score:.3f}")
    print(f"\nTop 5 layers:")
    top_5 = sorted(results.items(), key=lambda x: x[1]['silhouette'] or -np.inf, reverse=True)[:5]
    for name, metrics in top_5:
        print(f"    {name}: {metrics['silhouette']:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings_dir', required=True,
                   help='Path to pretrain/ or trained/ folder with embeddings')
    p.add_argument('--model_type', required=True,
                   choices=['pretrain', 'trained'],
                   help='pretrain or trained')
    p.add_argument('--output_dir', required=True,
                   help='Where to save analysis results')
    main(p.parse_args())
