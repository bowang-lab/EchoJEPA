#!/usr/bin/env python
# scripts/plot_umap_lvef.py
"""
Generate UMAP visualization for LVEF embeddings with continuous heatmap coloring.
Shows how well embeddings capture cardiac function (ejection fraction) structure.

Usage:
    python scripts/plot_umap_lvef.py --output figures/umap_lvef_comparison.pdf
    
Examples:
    # Default UMAP with viridis colormap
    python scripts/plot_umap_lvef.py
    
    # Custom colormap (plasma, magma, inferno, coolwarm, RdYlGn)
    python scripts/plot_umap_lvef.py --cmap RdYlGn_r --output figures/umap_lvef_rdylgn.pdf
    
    # Supervised UMAP (LVEF values guide projection)
    python scripts/plot_umap_lvef.py --supervised --output figures/umap_lvef_supervised.pdf
    
    # Bin LVEF for metrics (silhouette requires discrete labels)
    python scripts/plot_umap_lvef.py --lvef_bins 5 --output figures/umap_lvef_binned.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Clinical LVEF categories for reference
LVEF_CATEGORIES = {
    'HFrEF (≤40%)': (0, 40),
    'HFmrEF (41-49%)': (41, 49),
    'HFpEF (≥50%)': (50, 100),
}


def load_embeddings(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load embeddings from .npz file."""
    data = np.load(path)
    embeddings = data['embeddings']
    
    # Handle both 'lvef' and 'labels' keys
    if 'lvef' in data:
        lvef = data['lvef']
    elif 'labels' in data:
        lvef = data['labels']
    else:
        raise KeyError(f"No 'lvef' or 'labels' found in {path}")
    
    paths = data.get('paths', None)
    return embeddings, lvef, paths


def compute_metrics(
    embeddings: np.ndarray,
    lvef: np.ndarray,
    model_name: str,
    n_bins: int = 5,
    sample_size: int = 5000,
) -> Dict[str, float]:
    """
    Compute metrics for continuous LVEF values.
    - Silhouette: Computed on binned LVEF values
    - Linear probe: R² score for regression
    - Spearman correlation: Between embedding distances and LVEF differences
    """
    from sklearn.metrics import silhouette_score, r2_score
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    
    metrics = {}
    
    # Subsample for expensive metrics
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb_sample = embeddings[idx]
        lvef_sample = lvef[idx]
    else:
        emb_sample = embeddings
        lvef_sample = lvef
    
    # Bin LVEF for silhouette score
    lvef_min, lvef_max = lvef_sample.min(), lvef_sample.max()
    bins = np.linspace(lvef_min, lvef_max, n_bins + 1)
    lvef_binned = np.digitize(lvef_sample, bins[:-1]) - 1
    
    # Silhouette score on binned values
    try:
        # Only compute if we have multiple samples per bin
        unique_bins, counts = np.unique(lvef_binned, return_counts=True)
        if len(unique_bins) > 1 and counts.min() > 1:
            sil_cosine = silhouette_score(emb_sample, lvef_binned, metric='cosine')
            metrics['silhouette_cosine'] = sil_cosine
        else:
            metrics['silhouette_cosine'] = np.nan
    except Exception as e:
        print(f"  Warning: silhouette failed: {e}")
        metrics['silhouette_cosine'] = np.nan
    
    # Linear probe R² (regression)
    try:
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(embeddings)
        
        reg = Ridge(alpha=1.0)
        r2_scores = cross_val_score(reg, emb_scaled, lvef, cv=3, scoring='r2', n_jobs=-1)
        metrics['r2_score'] = r2_scores.mean()
    except Exception as e:
        print(f"  Warning: R² computation failed: {e}")
        metrics['r2_score'] = np.nan
    
    # Spearman correlation between embedding similarity and LVEF similarity
    try:
        # Sample pairs for correlation
        n_pairs = min(10000, len(emb_sample) * (len(emb_sample) - 1) // 2)
        idx1 = np.random.randint(0, len(emb_sample), n_pairs)
        idx2 = np.random.randint(0, len(emb_sample), n_pairs)
        
        # Embedding distances (cosine)
        from sklearn.metrics.pairwise import cosine_similarity
        emb_sim = np.array([
            cosine_similarity(emb_sample[i1:i1+1], emb_sample[i2:i2+1])[0, 0]
            for i1, i2 in zip(idx1, idx2)
        ])
        
        # LVEF differences (negative so higher similarity = closer LVEF)
        lvef_diff = -np.abs(lvef_sample[idx1] - lvef_sample[idx2])
        
        corr, pval = spearmanr(emb_sim, lvef_diff)
        metrics['spearman_corr'] = corr
    except Exception as e:
        print(f"  Warning: Spearman correlation failed: {e}")
        metrics['spearman_corr'] = np.nan
    
    return metrics


def apply_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Apply PCA dimensionality reduction."""
    from sklearn.decomposition import PCA
    
    n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def compute_projection(
    embeddings: np.ndarray,
    lvef: np.ndarray,
    method: str = 'umap',
    supervised: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    spread: float = 1.0,
    metric: str = 'cosine',
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D projection using UMAP or t-SNE."""
    
    # Subsample if needed
    if max_samples and len(embeddings) > max_samples:
        print(f"    Subsampling to {max_samples} for projection...")
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        lvef = lvef[idx]
    
    if method == 'umap':
        from umap import UMAP
        
        if supervised:
            # Use LVEF as continuous target
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                spread=spread,
                metric=metric,
                random_state=random_state,
                target_metric='l2',  # L2 distance for continuous targets
            )
            proj = reducer.fit_transform(embeddings, y=lvef)
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
        
        if len(embeddings) > 10000:
            print(f"    Subsampling to 10000 for t-SNE...")
            idx = np.random.choice(len(embeddings), 10000, replace=False)
            embeddings = embeddings[idx]
            lvef = lvef[idx]
        
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric='cosine' if metric == 'cosine' else 'euclidean',
            random_state=random_state,
            n_jobs=-1,
        )
        proj = reducer.fit_transform(embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return proj, lvef


def create_custom_colormap(name: str = 'cardiac'):
    """Create custom colormaps for LVEF visualization."""
    
    if name == 'cardiac':
        # Red (low EF) -> Yellow -> Green (high EF)
        colors = ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']
        return mcolors.LinearSegmentedColormap.from_list('cardiac', colors)
    
    elif name == 'cardiac_diverging':
        # Blue (low) -> White (normal ~55%) -> Red (high)
        colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']
        return mcolors.LinearSegmentedColormap.from_list('cardiac_div', colors)
    
    else:
        # Use matplotlib built-in
        return plt.get_cmap(name)


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
    pca_components: int = None,
    point_size: float = 3,
    alpha: float = 0.6,
    figsize: tuple = None,
    normalize_axes: bool = True,
    compute_metrics_flag: bool = True,
    show_metrics: bool = True,
    cmap: str = 'RdYlGn',
    vmin: float = None,
    vmax: float = None,
    max_samples: int = 20000,
    lvef_bins: int = 5,
):
    """Generate side-by-side LVEF UMAP plots with continuous colormap."""
    
    n_models = len(model_paths)
    
    if figsize is None:
        figsize = (4.5 * n_models + 0.8, 5.0 if show_metrics else 4.5)
    
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    
    # Get colormap
    if cmap in ['cardiac', 'cardiac_diverging']:
        colormap = create_custom_colormap(cmap)
    else:
        colormap = plt.get_cmap(cmap)
    
    # Store results
    all_projections = []
    all_lvef_list = []
    all_metrics = {}
    model_names = list(model_paths.keys())
    
    # Determine global LVEF range for consistent coloring
    global_lvef_min = float('inf')
    global_lvef_max = float('-inf')
    
    # First pass: load data and compute range
    print("Loading embeddings...")
    loaded_data = {}
    for model_name, emb_path in model_paths.items():
        embeddings, lvef, _ = load_embeddings(emb_path)
        loaded_data[model_name] = (embeddings, lvef)
        global_lvef_min = min(global_lvef_min, lvef.min())
        global_lvef_max = max(global_lvef_max, lvef.max())
    
    # Use provided vmin/vmax or global range
    if vmin is None:
        vmin = global_lvef_min
    if vmax is None:
        vmax = global_lvef_max
    
    print(f"LVEF range for colormap: [{vmin:.1f}, {vmax:.1f}]")
    
    # Second pass: compute projections and metrics
    for model_name, (embeddings, lvef) in loaded_data.items():
        print(f"\nProcessing {model_name}...")
        print(f"  Loaded {len(embeddings)} samples, shape: {embeddings.shape}")
        print(f"  LVEF: [{lvef.min():.1f}, {lvef.max():.1f}], mean: {lvef.mean():.1f}")
        
        # Compute metrics on original embeddings
        if compute_metrics_flag:
            print(f"  Computing metrics...")
            metrics = compute_metrics(embeddings, lvef, model_name, n_bins=lvef_bins)
            all_metrics[model_name] = metrics
            print(f"    Silhouette ({lvef_bins}-bin): {metrics['silhouette_cosine']:.3f}")
            print(f"    R² score: {metrics['r2_score']:.3f}")
            print(f"    Spearman corr: {metrics['spearman_corr']:.3f}")
        
        # Apply PCA if requested
        if pca_components:
            print(f"  Applying PCA ({pca_components} components)...")
            embeddings = apply_pca(embeddings, pca_components)
        
        # Compute projection
        method_name = f"{'Supervised ' if supervised else ''}{method.upper()}"
        print(f"  Computing {method_name} projection...")
        proj, lvef_proj = compute_projection(
            embeddings, lvef,
            method=method,
            supervised=supervised,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            metric=metric,
            perplexity=perplexity,
            random_state=random_state,
            max_samples=max_samples,
        )
        
        all_projections.append(proj)
        all_lvef_list.append(lvef_proj)
    
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
    
    # Normalize colors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Third pass: plot
    for i, (ax, proj, lvef_vals, model_name) in enumerate(zip(axes, all_projections, all_lvef_list, model_names)):
        
        # Sort by LVEF so high values are plotted on top
        sort_idx = np.argsort(lvef_vals)
        proj_sorted = proj[sort_idx]
        lvef_sorted = lvef_vals[sort_idx]
        
        scatter = ax.scatter(
            proj_sorted[:, 0],
            proj_sorted[:, 1],
            c=lvef_sorted,
            cmap=colormap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            rasterized=True,
            edgecolors='none',
        )
        
        # Title with optional metrics
        title = model_name
        if show_metrics and model_name in all_metrics:
            m = all_metrics[model_name]
            title += f"\nR²={m['r2_score']:.2f}, ρ={m['spearman_corr']:.2f}"
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if normalize_axes and centers[i] is not None:
            center = centers[i]
            ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
            ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        
        ax.set_aspect('equal')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('LVEF (%)', fontsize=10)
    
    # Add clinical category annotations
    cbar.ax.axhline(y=40, color='black', linewidth=1, linestyle='--', alpha=0.5)
    cbar.ax.axhline(y=50, color='black', linewidth=1, linestyle='--', alpha=0.5)
    
    # Parameter annotation
    if method == 'umap':
        param_text = f"{'Supervised ' if supervised else ''}UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}"
    else:
        param_text = f"t-SNE: perplexity={perplexity}"
    
    if pca_components:
        param_text += f", PCA={pca_components}"
    
    fig.text(0.45, 0.02, param_text, ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.90, bottom=0.08)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {png_path}")
    
    plt.close()
    
    return all_metrics


def print_metrics_table(metrics: Dict[str, Dict[str, float]], n_bins: int = 5):
    """Print metrics as a formatted table."""
    print("\n" + "=" * 75)
    print("QUANTITATIVE METRICS (LVEF)")
    print("=" * 75)
    print(f"{'Model':<15} {'Sil (cos)':<12} {'R² Score':<12} {'Spearman ρ':<12}")
    print(f"{'':15} {f'({n_bins}-bin)':<12} {'(regression)':<12} {'(corr)':<12}")
    print("-" * 75)
    
    for model_name, m in metrics.items():
        sil = f"{m['silhouette_cosine']:.3f}" if not np.isnan(m['silhouette_cosine']) else "N/A"
        r2 = f"{m['r2_score']:.3f}" if not np.isnan(m['r2_score']) else "N/A"
        rho = f"{m['spearman_corr']:.3f}" if not np.isnan(m['spearman_corr']) else "N/A"
        print(f"{model_name:<15} {sil:<12} {r2:<12} {rho:<12}")
    
    print("=" * 75)
    print("Silhouette: cluster separation when LVEF is binned (-1 to 1, higher=better)")
    print("R² Score: linear regression predictability of LVEF from embeddings")
    print("Spearman ρ: correlation between embedding similarity and LVEF similarity")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LVEF UMAP visualization with continuous colormap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Default UMAP with RdYlGn colormap
  python scripts/plot_umap_lvef.py
  
  # Custom colormap
  python scripts/plot_umap_lvef.py --cmap plasma --output figures/umap_lvef_plasma.pdf
  
  # Cardiac-specific colormap (red=low EF, green=high EF)
  python scripts/plot_umap_lvef.py --cmap cardiac --output figures/umap_lvef_cardiac.pdf
  
  # Supervised UMAP (LVEF guides projection)
  python scripts/plot_umap_lvef.py --supervised --output figures/umap_lvef_supervised.pdf
  
  # Fixed LVEF range for clinical interpretation
  python scripts/plot_umap_lvef.py --vmin 20 --vmax 75 --output figures/umap_lvef_clinical.pdf
        """
    )
    
    # Output
    parser.add_argument("--output", default="figures/umap_lvef_comparison.pdf")
    parser.add_argument("--embeddings_dir", default="embeddings/lvef")
    
    # Method selection
    parser.add_argument("--method", choices=['umap', 'tsne'], default='umap')
    parser.add_argument("--supervised", action="store_true",
                        help="Use supervised UMAP (LVEF guides projection)")
    parser.add_argument("--pca", type=int, default=None, metavar='N')
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--metric", choices=['cosine', 'euclidean'], default='cosine')
    
    # t-SNE parameters
    parser.add_argument("--perplexity", type=float, default=30.0)
    
    # Colormap
    parser.add_argument("--cmap", default="RdYlGn",
                        help="Colormap name (viridis, plasma, RdYlGn, cardiac, etc.)")
    parser.add_argument("--vmin", type=float, default=None, help="Min LVEF for colormap")
    parser.add_argument("--vmax", type=float, default=None, help="Max LVEF for colormap")
    
    # Common parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Max samples per model for projection")
    parser.add_argument("--lvef_bins", type=int, default=5,
                        help="Number of bins for silhouette score")
    
    # Visual parameters
    parser.add_argument("--point_size", type=float, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no_normalize", action="store_true")
    
    # Metrics
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--no_show_metrics", action="store_true")
    
    args = parser.parse_args()
    
    # Define model paths
    emb_dir = args.embeddings_dir
    model_paths = {
        'EchoJEPA-G': f'{emb_dir}/echojepa_g_embeddings.npz',
        'EchoJEPA-L': f'{emb_dir}/echojepa_l_embeddings.npz',
        'EchoMAE-L': f'{emb_dir}/echomae_l_embeddings.npz',
        'EchoPrime': f'{emb_dir}/echoprime_embeddings.npz',
        'PanEcho': f'{emb_dir}/panecho_embeddings.npz',
    }
    
    # Filter to available models
    available_models = {k: v for k, v in model_paths.items() if Path(v).exists()}
    
    if not available_models:
        print("Error: No embedding files found!")
        print(f"Expected directory: {emb_dir}")
        for k, v in model_paths.items():
            print(f"  {k}: {v} - {'EXISTS' if Path(v).exists() else 'MISSING'}")
        return
    
    print(f"Found {len(available_models)} models: {list(available_models.keys())}")
    
    # Full visualization
    print(f"\nSettings:")
    print(f"  Method: {'Supervised ' if args.supervised else ''}{args.method.upper()}")
    print(f"  Colormap: {args.cmap}")
    if args.method == 'umap':
        print(f"  n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
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
        pca_components=args.pca,
        point_size=args.point_size,
        alpha=args.alpha,
        normalize_axes=not args.no_normalize,
        compute_metrics_flag=not args.no_metrics,
        show_metrics=not args.no_show_metrics,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        max_samples=args.max_samples,
        lvef_bins=args.lvef_bins,
    )
    
    # Print metrics table
    if metrics and not args.no_metrics:
        print_metrics_table(metrics, args.lvef_bins)


if __name__ == "__main__":
    main()