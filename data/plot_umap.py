#!/usr/bin/env python
# scripts/plot_umap.py
"""
Generate UMAP/t-SNE visualization comparing encoder embeddings across models.
Includes quantitative metrics: silhouette score and linear probe accuracy.

Usage:
    python data/plot_umap.py --output figures/umap_view_comparison.pdf
    
Examples:
    # Default UMAP
    python data/plot_umap.py
    
    # Supervised UMAP (uses labels to guide projection)
    python data/plot_umap.py --supervised --output figures/umap_supervised.pdf
    
    # t-SNE instead of UMAP
    python data/plot_umap.py --method tsne --output figures/tsne_comparison.pdf
    
    # With PCA preprocessing
    python data/plot_umap.py --pca 50 --output figures/umap_pca50.pdf
    
    # Just compute metrics, no plot
    python data/plot_umap.py --metrics_only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Class map from your dataset
VIEW_CLASSES = {
    0: 'A2C',
    1: 'A3C',
    2: 'A4C',
    3: 'A5C',
    4: 'Other',
    5: 'PLAX',
    6: 'PSAX-AP',
    7: 'PSAX-AV',
    8: 'PSAX-MV',
    9: 'PSAX-PM',
    10: 'SSN',
    11: 'Subcostal',
    12: 'TEE',
}

# Color palette - distinct colors for each view
# Get the 'rainbow' colormap
cmap = plt.get_cmap('rainbow')

# Create a list of keys (0, 1, 2... 12)
keys = sorted(VIEW_CLASSES.keys())

# Generate evenly spaced values between 0 and 1
# This ensures distinct colors for each class across the full spectrum
vals = np.linspace(0, 1, len(keys))

# Re-create the COLORS dictionary dynamically
# This maps class_id -> RGBA color tuple
COLORS = {
    k: cmap(val) 
    for k, val in zip(keys, vals)
}


def load_embeddings(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load embeddings from .npz file."""
    data = np.load(path)
    return data['embeddings'], data['labels'], data.get('paths', None)


def compute_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    sample_size: int = 5000,
) -> Dict[str, float]:
    """Compute silhouette score and linear probe accuracy."""
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    metrics = {}
    
    # Subsample for silhouette (expensive for large datasets)
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb_sample = embeddings[idx]
        labels_sample = labels[idx]
    else:
        emb_sample = embeddings
        labels_sample = labels
    
    # Silhouette score (higher = better separated clusters)
    try:
        sil_cosine = silhouette_score(emb_sample, labels_sample, metric='cosine')
        metrics['silhouette_cosine'] = sil_cosine
    except Exception as e:
        print(f"  Warning: silhouette (cosine) failed: {e}")
        metrics['silhouette_cosine'] = np.nan
    
    try:
        sil_euclidean = silhouette_score(emb_sample, labels_sample, metric='euclidean')
        metrics['silhouette_euclidean'] = sil_euclidean
    except Exception as e:
        print(f"  Warning: silhouette (euclidean) failed: {e}")
        metrics['silhouette_euclidean'] = np.nan
    
    # Linear probe accuracy (3-fold CV for speed)
    try:
        # Normalize embeddings for linear probe
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(embeddings)
        
        clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
        acc = cross_val_score(clf, emb_scaled, labels, cv=3, n_jobs=-1).mean()
        metrics['linear_acc'] = acc
    except Exception as e:
        print(f"  Warning: linear probe failed: {e}")
        metrics['linear_acc'] = np.nan
    
    return metrics


def apply_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    from sklearn.decomposition import PCA
    
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def compute_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'umap',
    supervised: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: str = 'cosine',
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute 2D projection using UMAP or t-SNE."""
    
    if method == 'umap':
        from umap import UMAP
        
        if supervised:
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                metric=metric,
                random_state=random_state,
                target_metric='categorical',
            )
            proj = reducer.fit_transform(embeddings, y=labels)
        else:
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                metric=metric,
                random_state=random_state,
            )
            proj = reducer.fit_transform(embeddings)
    
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        
        # t-SNE doesn't support cosine directly, use precomputed if needed
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_distances
            # Subsample if too large (t-SNE is slow)
            if len(embeddings) > 10000:
                print(f"    Subsampling to 10000 for t-SNE...")
                idx = np.random.choice(len(embeddings), 10000, replace=False)
                embeddings = embeddings[idx]
                labels = labels[idx]
            
            distances = cosine_distances(embeddings)
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric='precomputed',
                random_state=random_state,
                n_jobs=-1,
            )
            proj = reducer.fit_transform(distances)
        else:
            if len(embeddings) > 10000:
                print(f"    Subsampling to 10000 for t-SNE...")
                idx = np.random.choice(len(embeddings), 10000, replace=False)
                embeddings = embeddings[idx]
            
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                metric=metric,
                random_state=random_state,
                n_jobs=-1,
            )
            proj = reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return proj, labels  # Return labels too (may be subsampled for t-SNE)


def plot_comparison(
    model_paths: Dict[str, str],
    output_path: str,
    method: str = 'umap',
    supervised: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: str = 'cosine',
    perplexity: float = 30.0,
    random_state: int = 42,
    exclude_classes: List[int] = None,
    pca_components: int = None,
    point_size: float = 5,
    alpha: float = 0.7,
    figsize: tuple = None,
    normalize_axes: bool = True,
    compute_metrics_flag: bool = True,
    show_metrics: bool = True,
):
    """Generate side-by-side projection plots with optional metrics."""
    
    n_models = len(model_paths)
    
    if figsize is None:
        figsize = (4.5 * n_models, 5.0 if show_metrics else 4.5)
    
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    # Store results
    all_projections = []
    all_labels_list = []
    all_metrics = {}
    model_names = list(model_paths.keys())
    
    # First pass: compute projections and metrics
    for model_name, emb_path in model_paths.items():
        print(f"Processing {model_name}...")
        
        embeddings, labels, _ = load_embeddings(emb_path)
        print(f"  Loaded {len(embeddings)} samples, shape: {embeddings.shape}")
        
        # Exclude classes if specified
        if exclude_classes:
            mask = ~np.isin(labels, exclude_classes)
            embeddings = embeddings[mask]
            labels = labels[mask]
            print(f"  After excluding {exclude_classes}: {len(embeddings)} samples")
        
        # Compute metrics on original embeddings
        if compute_metrics_flag:
            print(f"  Computing metrics...")
            metrics = compute_metrics(embeddings, labels, model_name)
            all_metrics[model_name] = metrics
            print(f"    Silhouette (cosine): {metrics['silhouette_cosine']:.3f}")
            print(f"    Silhouette (euclidean): {metrics['silhouette_euclidean']:.3f}")
            print(f"    Linear probe acc: {metrics['linear_acc']:.1%}")
        
        # Apply PCA if requested
        if pca_components:
            print(f"  Applying PCA ({pca_components} components)...")
            embeddings = apply_pca(embeddings, pca_components)
        
        # Compute projection
        method_name = f"{'Supervised ' if supervised else ''}{method.upper()}"
        print(f"  Computing {method_name} projection...")
        proj, labels = compute_projection(
            embeddings, labels,
            method=method,
            supervised=supervised,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            metric=metric,
            perplexity=perplexity,
            random_state=random_state,
        )
        
        all_projections.append(proj)
        all_labels_list.append(labels)
        
        print(f"  Done. Range: x=[{proj[:,0].min():.1f}, {proj[:,0].max():.1f}], y=[{proj[:,1].min():.1f}, {proj[:,1].max():.1f}]")
    
    # Compute consistent axis limits
    if normalize_axes:
        max_range = 0
        centers = []
        for proj in all_projections:
            x_range = proj[:, 0].max() - proj[:, 0].min()
            y_range = proj[:, 1].max() - proj[:, 1].min()
            max_range = max(max_range, x_range, y_range)
            centers.append([proj[:, 0].mean(), proj[:, 1].mean()])
        max_range = max_range * 1.1
    else:
        centers = [None] * n_models
    
    # Second pass: plot
    for i, (ax, proj, labels, model_name) in enumerate(zip(axes, all_projections, all_labels_list, model_names)):
        
        for class_idx in sorted(np.unique(labels)):
            class_mask = labels == class_idx
            ax.scatter(
                proj[class_mask, 0],
                proj[class_mask, 1],
                c=COLORS.get(class_idx, '#333333'),
                s=point_size,
                alpha=alpha,
                label=VIEW_CLASSES.get(class_idx, f'Class {class_idx}'),
                rasterized=True,
                edgecolors='none',
            )
        
        # Title with optional metrics
        title = model_name
        if show_metrics and model_name in all_metrics:
            m = all_metrics[model_name]
            title += f"\nSil={m['silhouette_cosine']:.2f}, Acc={m['linear_acc']:.0%}"
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        
        if normalize_axes and centers[i] is not None:
            center = centers[i]
            ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        
        ax.set_aspect('equal')
        
        # Remove borders (spines) and axis ticks completely
        ax.axis('off')
    
    # Shared legend
    all_present_classes = set()
    for labels in all_labels_list:
        all_present_classes.update(np.unique(labels))
    
    handles = [
        mpatches.Patch(color=COLORS[i], label=VIEW_CLASSES[i])
        for i in sorted(VIEW_CLASSES.keys())
        if i in all_present_classes and (exclude_classes is None or i not in exclude_classes)
    ]
    
    fig.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        frameon=False, # Removed border from legend as well
        title='View Class',
        title_fontsize=10,
    )
    
    # Parameter annotation
    if method == 'umap':
        param_text = f"{'Supervised ' if supervised else ''}UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    else:
        param_text = f"t-SNE: perplexity={perplexity}, metric={metric}"
    
    if pca_components:
        param_text += f", PCA={pca_components}"
    
    fig.text(0.5, 0.02, param_text, ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88, bottom=0.08)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {png_path}")
    
    plt.close()
    
    return all_metrics


def print_metrics_table(metrics: Dict[str, Dict[str, float]]):
    """Print metrics as a formatted table."""
    print("\n" + "=" * 65)
    print("QUANTITATIVE METRICS")
    print("=" * 65)
    print(f"{'Model':<15} {'Sil (cos)':>12} {'Sil (euc)':>12} {'Linear Acc':>12}")
    print("-" * 65)
    
    for model_name, m in metrics.items():
        sil_cos = f"{m['silhouette_cosine']:.3f}" if not np.isnan(m['silhouette_cosine']) else "N/A"
        sil_euc = f"{m['silhouette_euclidean']:.3f}" if not np.isnan(m['silhouette_euclidean']) else "N/A"
        lin_acc = f"{m['linear_acc']:.1%}" if not np.isnan(m['linear_acc']) else "N/A"
        print(f"{model_name:<15} {sil_cos:>12} {sil_euc:>12} {lin_acc:>12}")
    
    print("=" * 65)
    print("Silhouette: higher = better separated clusters (-1 to 1)")
    print("Linear Acc: accuracy of logistic regression on raw embeddings")
    print("=" * 65 + "\n")


def compute_metrics_only(model_paths: Dict[str, str], exclude_classes: List[int] = None):
    """Compute and print metrics without generating plots."""
    all_metrics = {}
    
    for model_name, emb_path in model_paths.items():
        print(f"Processing {model_name}...")
        
        embeddings, labels, _ = load_embeddings(emb_path)
        print(f"  Loaded {len(embeddings)} samples, shape: {embeddings.shape}")
        
        if exclude_classes:
            mask = ~np.isin(labels, exclude_classes)
            embeddings = embeddings[mask]
            labels = labels[mask]
        
        metrics = compute_metrics(embeddings, labels, model_name)
        all_metrics[model_name] = metrics
    
    print_metrics_table(all_metrics)
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate UMAP/t-SNE comparison with metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Default UMAP
  python data/plot_umap.py
  
  # Supervised UMAP (labels guide the projection)
  python data/plot_umap.py --supervised --output figures/umap_supervised.pdf
  
  # t-SNE comparison
  python data/plot_umap.py --method tsne --output figures/tsne_comparison.pdf
  
  # With PCA preprocessing (50 dims)
  python data/plot_umap.py --pca 50 --output figures/umap_pca50.pdf
  
  # Tight clusters
  python data/plot_umap.py --n_neighbors 10 --min_dist 0.0 --output figures/umap_tight.pdf
  
  # Global structure
  python data/plot_umap.py --n_neighbors 50 --min_dist 0.25 --output figures/umap_global.pdf
  
  # Metrics only (no plot)
  python data/plot_umap.py --metrics_only
  
  # Hide metrics from plot titles
  python data/plot_umap.py --no_show_metrics
        """
    )
    
    # Output
    parser.add_argument("--output", default="figures/umap_view_comparison.pdf")
    parser.add_argument("--embeddings_dir", default="embeddings")
    
    # Method selection
    parser.add_argument("--method", choices=['umap', 'tsne'], default='umap',
                        help="Dimensionality reduction method")
    parser.add_argument("--supervised", action="store_true",
                        help="Use supervised UMAP (labels guide projection)")
    parser.add_argument("--pca", type=int, default=None, metavar='N',
                        help="Apply PCA to N dimensions before projection")
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors (5-50)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist (0.0-1.0)")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="UMAP spread")
    parser.add_argument("--metric", choices=['cosine', 'euclidean'], default='cosine')
    
    # t-SNE parameters
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity (5-50)")
    
    # Common parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_class", type=int, nargs='+', default=None)
    
    # Visual parameters
    parser.add_argument("--point_size", type=float, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--no_normalize", action="store_true",
                        help="Don't normalize axis ranges")
    
    # Metrics
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only compute metrics, don't generate plot")
    parser.add_argument("--no_metrics", action="store_true",
                        help="Skip metrics computation")
    parser.add_argument("--no_show_metrics", action="store_true",
                        help="Don't show metrics in plot titles")
    
    args = parser.parse_args()
    
    # Define model paths with requested order
    emb_dir = args.embeddings_dir
    model_paths = {
        'PanEcho': f'{emb_dir}/panecho_embeddings.npz',
        'EchoPrime': f'{emb_dir}/echoprime_embeddings.npz',
        'EchoMAE-L': f'{emb_dir}/echomae_l_embeddings.npz',
        'EchoJEPA-L': f'{emb_dir}/echojepa_l_embeddings.npz',
        'EchoJEPA-G': f'{emb_dir}/echojepa_g_embeddings.npz',
    }
    
    # Filter to available models
    available_models = {k: v for k, v in model_paths.items() if Path(v).exists()}
    
    if not available_models:
        print("Error: No embedding files found!")
        for k, v in model_paths.items():
            print(f"  {k}: {v} - {'EXISTS' if Path(v).exists() else 'MISSING'}")
        return
    
    print(f"Found {len(available_models)} models: {list(available_models.keys())}")
    
    # Metrics only mode
    if args.metrics_only:
        compute_metrics_only(available_models, args.exclude_class)
        return
    
    # Full visualization
    print(f"\nSettings:")
    print(f"  Method: {'Supervised ' if args.supervised else ''}{args.method.upper()}")
    if args.method == 'umap':
        print(f"  n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, spread={args.spread}")
    else:
        print(f"  perplexity={args.perplexity}")
    print(f"  metric={args.metric}")
    if args.pca:
        print(f"  PCA preprocessing: {args.pca} components")
    print()
    
    metrics = plot_comparison(
        model_paths=available_models,
        output_path=args.output,
        method=args.method,
        supervised=args.supervised,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        metric=args.metric,
        perplexity=args.perplexity,
        random_state=args.seed,
        exclude_classes=args.exclude_class,
        pca_components=args.pca,
        point_size=args.point_size,
        alpha=args.alpha,
        normalize_axes=not args.no_normalize,
        compute_metrics_flag=not args.no_metrics,
        show_metrics=not args.no_show_metrics,
    )
    
    # Print metrics table
    if metrics and not args.no_metrics:
        print_metrics_table(metrics)


if __name__ == "__main__":
    main()