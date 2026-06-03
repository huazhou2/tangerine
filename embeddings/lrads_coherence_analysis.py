"""
LRADS Cluster Coherence Analysis

Find which layer/representation best captures LRADS risk stratification through
cluster coherence metrics. Analyzes:
  - All 24 transformer layers (pretrained + trained)
  - Attention head outputs (12 heads × 24 layers)
  - PCA components at different dimensionalities
  - Individual embedding dimensions ranked by correlation

Metrics:
  - Silhouette score (cluster tightness for LRADS groups)
  - Adjusted Rand Index (agreement between embedding clusters and LRADS)
  - Linear probe accuracy (can we classify LRADS from embeddings?)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score, adjusted_rand_index
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import json
import argparse
from tqdm import tqdm

def compute_lrads_coherence(embeddings, lrads_labels, method='silhouette'):
    """
    Measure how well LRADS groups cluster in embedding space.

    Args:
        embeddings: (N, D) array
        lrads_labels: (N,) categorical labels
        method: 'silhouette' | 'rand_index' | 'linear_probe'

    Returns:
        Score (higher = better LRADS clustering)
    """
    # Encode categorical labels
    lrads_encoded = pd.factorize(lrads_labels)[0]

    if method == 'silhouette':
        return silhouette_score(embeddings, lrads_encoded)

    elif method == 'rand_index':
        # ARI: compare embedding-based clustering vs LRADS groups
        from sklearn.cluster import KMeans
        n_clusters = len(np.unique(lrads_encoded))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred_clusters = kmeans.fit_predict(embeddings)
        return adjusted_rand_index(lrads_encoded, pred_clusters)

    elif method == 'linear_probe':
        # Train simple classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(embeddings, lrads_encoded)
        return clf.score(embeddings, lrads_encoded)


def analyze_full_layers(embeddings_dir, model_type='pretrain', output_dir=None):
    """
    Analyze all 24 transformer layers.

    Returns:
        dict: layer → coherence score
    """
    results = {}

    if model_type == 'pretrain':
        n_layers = 24
        meta_file = f'{embeddings_dir}/embeddings_meta_layer0.csv'
    else:
        n_layers = 24
        meta_file = f'{embeddings_dir}/embeddings_meta_layer0.csv'

    meta = pd.read_csv(meta_file)
    lrads = meta['lrads_category'].values

    for layer in tqdm(range(n_layers), desc=f"Analyzing {model_type} layers"):
        emb_file = f'{embeddings_dir}/embeddings_layer{layer}.npy'
        if not Path(emb_file).exists():
            continue

        emb = np.load(emb_file)
        score = compute_lrads_coherence(emb, lrads, method='silhouette')
        results[f'layer_{layer}'] = score

    return results


def analyze_attention_heads(embeddings_dir, model_type='pretrain', output_dir=None):
    """
    Extract and analyze individual attention head outputs.

    Note: Requires extracting attention heads from model checkpoint.
    This is a placeholder for head-level analysis.
    """
    # TODO: Implement if attention weights are extracted
    return {}


def analyze_pca_components(embeddings_dir, model_type='pretrain', n_components_list=None):
    """
    Analyze PCA-reduced embeddings at different dimensionalities.

    This finds if dimensionality reduction reveals LRADS structure better.
    """
    if n_components_list is None:
        n_components_list = [2, 5, 10, 25, 50, 100, 256, 512]

    results = {}
    meta_file = f'{embeddings_dir}/embeddings_meta_layer0.csv'
    meta = pd.read_csv(meta_file)
    lrads = meta['lrads_category'].values

    # Use final layer or layer 23
    if model_type == 'trained':
        emb_file = f'{embeddings_dir}/embeddings_layer_final.npy'
    else:
        emb_file = f'{embeddings_dir}/embeddings_layer23.npy'

    emb = np.load(emb_file)

    for n_comp in tqdm(n_components_list, desc="PCA analysis"):
        if n_comp >= emb.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        emb_reduced = pca.fit_transform(emb)
        score = compute_lrads_coherence(emb_reduced, lrads, method='silhouette')
        results[f'pca_{n_comp}'] = score

    return results


def analyze_top_dimensions(embeddings_dir, model_type='pretrain', top_k=50):
    """
    Find embedding dimensions that most correlate with LRADS.

    Returns top-K most predictive individual dimensions.
    """
    meta_file = f'{embeddings_dir}/embeddings_meta_layer0.csv'
    meta = pd.read_csv(meta_file)
    lrads = pd.factorize(meta['lrads_category'])[0].astype(float)

    if model_type == 'trained':
        emb_file = f'{embeddings_dir}/embeddings_layer_final.npy'
    else:
        emb_file = f'{embeddings_dir}/embeddings_layer23.npy'

    emb = np.load(emb_file)

    # Compute correlation of each dimension with LRADS
    correlations = []
    for dim in range(emb.shape[1]):
        corr, _ = spearmanr(emb[:, dim], lrads)
        correlations.append(abs(corr))

    top_dims = np.argsort(correlations)[-top_k:][::-1]
    top_corrs = [correlations[d] for d in top_dims]

    return {f'dim_{d}': float(top_corrs[i]) for i, d in enumerate(top_dims)}


def create_summary_plot(results_dict, output_file=None):
    """
    Create comparison plot of all analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full layers
    if 'full_layers_pretrain' in results_dict:
        layers = sorted(results_dict['full_layers_pretrain'].keys())
        scores = [results_dict['full_layers_pretrain'][l] for l in layers]
        axes[0, 0].plot(layers, scores, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Pretrained Layers - Silhouette Score')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Full layers trained
    if 'full_layers_trained' in results_dict:
        layers = sorted(results_dict['full_layers_trained'].keys())
        scores = [results_dict['full_layers_trained'][l] for l in layers]
        axes[0, 1].plot(layers, scores, 'o-', color='orange', linewidth=2, markersize=6)
        axes[0, 1].set_title('Fine-tuned Layers - Silhouette Score')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: PCA analysis
    if 'pca_components' in results_dict:
        n_comps = sorted([int(k.split('_')[1]) for k in results_dict['pca_components'].keys()])
        scores = [results_dict['pca_components'][f'pca_{n}'] for n in n_comps]
        axes[1, 0].semilogx(n_comps, scores, 'o-', color='green', linewidth=2, markersize=6)
        axes[1, 0].set_title('PCA Dimensionality - Silhouette Score')
        axes[1, 0].set_xlabel('PCA Components')
        axes[1, 0].set_ylabel('Silhouette Score')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary stats
    axes[1, 1].axis('off')
    summary_text = "Analysis Summary:\n\n"

    if 'full_layers_pretrain' in results_dict:
        best_layer = max(results_dict['full_layers_pretrain'],
                        key=results_dict['full_layers_pretrain'].get)
        best_score = results_dict['full_layers_pretrain'][best_layer]
        summary_text += f"Best Pretrained: {best_layer} (score={best_score:.3f})\n"

    if 'full_layers_trained' in results_dict:
        best_layer = max(results_dict['full_layers_trained'],
                        key=results_dict['full_layers_trained'].get)
        best_score = results_dict['full_layers_trained'][best_layer]
        summary_text += f"Best Trained: {best_layer} (score={best_score:.3f})\n"

    if 'pca_components' in results_dict:
        best_pca = max(results_dict['pca_components'],
                      key=results_dict['pca_components'].get)
        best_score = results_dict['pca_components'][best_pca]
        summary_text += f"Best PCA: {best_pca} (score={best_score:.3f})\n"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to {output_file}")
    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze which layer captures LRADS coherence')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Path to embeddings directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--model_type', choices=['pretrain', 'trained'], default='pretrain')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.embeddings_dir).parent / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("LRADS CLUSTER COHERENCE ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Model: {args.model_type}")
    print(f"Input: {args.embeddings_dir}")
    print(f"Output: {output_dir}\n")

    results = {}

    # 1. Analyze all 24 layers
    print("\n[1/4] Analyzing 24 transformer layers...")
    results[f'full_layers_{args.model_type}'] = analyze_full_layers(
        args.embeddings_dir, args.model_type, str(output_dir)
    )

    # 2. Analyze PCA components
    print("\n[2/4] Analyzing PCA-reduced embeddings...")
    results['pca_components'] = analyze_pca_components(args.embeddings_dir, args.model_type)

    # 3. Find top predictive dimensions
    print("\n[3/4] Finding top predictive dimensions...")
    results['top_dimensions'] = analyze_top_dimensions(args.embeddings_dir, args.model_type, top_k=50)

    # 4. Create summary visualization
    print("\n[4/4] Creating summary plots...")
    create_summary_plot(results, str(output_dir / 'lrads_coherence_summary.png'))

    # Save detailed results
    results_json = {k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()}
    with open(output_dir / 'lrads_coherence_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - lrads_coherence_results.json")
    print(f"  - lrads_coherence_summary.png")

    # Print best layer
    best_layer_dict = results[f'full_layers_{args.model_type}']
    if best_layer_dict:
        best_layer = max(best_layer_dict, key=best_layer_dict.get)
        best_score = best_layer_dict[best_layer]
        print(f"\n🏆 Best layer: {best_layer} (silhouette = {best_score:.4f})")


if __name__ == '__main__':
    main()
