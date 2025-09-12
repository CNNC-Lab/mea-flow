"""
Clustering analysis for feature space data.

This module provides clustering algorithms for analyzing processed datasets where features
are measurements and samples are individual experimental conditions or observations.
Complements the manifold learning capabilities with unsupervised clustering methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import (
        silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
        calinski_harabasz_score, davies_bouldin_score
    )
    from sklearn.impute import SimpleImputer
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Clustering analysis will be disabled.")

from .feature_space import load_feature_data, apply_feature_embedding


def apply_dbscan_clustering(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = 'euclidean',
    **kwargs
) -> Dict[str, Any]:
    """
    Apply DBSCAN (Density-Based Spatial Clustering) to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (n_samples x n_features)
    eps : float
        Maximum distance between two samples for them to be considered neighbors
    min_samples : int
        Minimum number of samples in a neighborhood for a point to be core
    metric : str
        Distance metric to use
    **kwargs
        Additional DBSCAN parameters
        
    Returns
    -------
    dict
        Dictionary containing clustering results and metadata
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for clustering analysis")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        **kwargs
    )
    
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Calculate metrics (excluding noise points for silhouette score)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'noise_ratio': n_noise / len(cluster_labels)
    }
    
    # Calculate silhouette score if we have valid clusters
    if n_clusters > 1 and n_noise < len(cluster_labels):
        valid_mask = cluster_labels != -1
        if np.sum(valid_mask) > 1:
            metrics['silhouette_score'] = silhouette_score(
                X_scaled[valid_mask], cluster_labels[valid_mask]
            )
    
    result = {
        'cluster_labels': cluster_labels,
        'model': dbscan,
        'scaler': scaler,
        'method': 'DBSCAN',
        'parameters': {'eps': eps, 'min_samples': min_samples, 'metric': metric},
        'metrics': metrics,
        'original_shape': X.shape
    }
    
    return result


def apply_gmm_clustering(
    X: np.ndarray,
    n_components: int = 3,
    covariance_type: str = 'full',
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Gaussian Mixture Model clustering to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (n_samples x n_features)
    n_components : int
        Number of mixture components (clusters)
    covariance_type : str
        Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
    random_state : int
        Random state for reproducibility
    **kwargs
        Additional GMM parameters
        
    Returns
    -------
    dict
        Dictionary containing clustering results and metadata
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for clustering analysis")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        **kwargs
    )
    
    cluster_labels = gmm.fit_predict(X_scaled)
    cluster_probs = gmm.predict_proba(X_scaled)
    
    # Calculate metrics
    metrics = {
        'n_clusters': n_components,
        'aic': gmm.aic(X_scaled),
        'bic': gmm.bic(X_scaled),
        'log_likelihood': gmm.score(X_scaled),
        'converged': gmm.converged_
    }
    
    if n_components > 1:
        metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
    
    result = {
        'cluster_labels': cluster_labels,
        'cluster_probabilities': cluster_probs,
        'model': gmm,
        'scaler': scaler,
        'method': 'GMM',
        'parameters': {'n_components': n_components, 'covariance_type': covariance_type},
        'metrics': metrics,
        'original_shape': X.shape
    }
    
    return result


def apply_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply K-means clustering to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (n_samples x n_features)
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    n_init : int
        Number of random initializations
    **kwargs
        Additional K-means parameters
        
    Returns
    -------
    dict
        Dictionary containing clustering results and metadata
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for clustering analysis")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        **kwargs
    )
    
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = {
        'n_clusters': n_clusters,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_
    }
    
    if n_clusters > 1:
        metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
    
    result = {
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'model': kmeans,
        'scaler': scaler,
        'method': 'K-means',
        'parameters': {'n_clusters': n_clusters, 'n_init': n_init},
        'metrics': metrics,
        'original_shape': X.shape
    }
    
    return result


def apply_hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = 3,
    linkage_method: str = 'ward',
    distance_threshold: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply Agglomerative Hierarchical clustering to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (n_samples x n_features)
    n_clusters : int
        Number of clusters (ignored if distance_threshold is set)
    linkage_method : str
        Linkage criterion ('ward', 'complete', 'average', 'single')
    distance_threshold : float, optional
        Distance threshold for automatic cluster determination
    **kwargs
        Additional hierarchical clustering parameters
        
    Returns
    -------
    dict
        Dictionary containing clustering results and metadata
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for clustering analysis")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters if distance_threshold is None else None,
        linkage=linkage_method,
        distance_threshold=distance_threshold,
        **kwargs
    )
    
    cluster_labels = hierarchical.fit_predict(X_scaled)
    actual_n_clusters = len(np.unique(cluster_labels))
    
    # Calculate linkage matrix for dendrogram
    linkage_matrix = linkage(X_scaled, method=linkage_method)
    
    # Calculate metrics
    metrics = {
        'n_clusters': actual_n_clusters,
        'n_leaves': hierarchical.n_leaves_,
        'n_connected_components': hierarchical.n_connected_components_
    }
    
    if actual_n_clusters > 1:
        metrics['silhouette_score'] = silhouette_score(X_scaled, cluster_labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, cluster_labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, cluster_labels)
    
    result = {
        'cluster_labels': cluster_labels,
        'linkage_matrix': linkage_matrix,
        'model': hierarchical,
        'scaler': scaler,
        'method': 'Hierarchical',
        'parameters': {'n_clusters': n_clusters, 'linkage': linkage_method, 
                      'distance_threshold': distance_threshold},
        'metrics': metrics,
        'original_shape': X.shape
    }
    
    return result


def apply_multiple_clustering(
    X: np.ndarray,
    methods: List[str] = ['KMEANS', 'GMM', 'DBSCAN', 'HIERARCHICAL'],
    n_clusters: int = 3,
    random_state: int = 42,
    **method_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple clustering methods to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix
    methods : list of str
        List of clustering methods to apply
    n_clusters : int
        Number of clusters for methods that require it
    random_state : int
        Random state for reproducibility
    **method_kwargs
        Method-specific parameters (e.g., dbscan_eps=0.5)
        
    Returns
    -------
    dict
        Dictionary mapping method names to clustering results
    """
    clustering_results = {}
    
    for method in methods:
        print(f"Applying {method}...")
        
        try:
            method_upper = method.upper()
            
            if method_upper == 'DBSCAN':
                eps = method_kwargs.get('dbscan_eps', 0.5)
                min_samples = method_kwargs.get('dbscan_min_samples', 5)
                result = apply_dbscan_clustering(
                    X, eps=eps, min_samples=min_samples
                )
                
            elif method_upper == 'GMM':
                covariance_type = method_kwargs.get('gmm_covariance_type', 'full')
                result = apply_gmm_clustering(
                    X, n_components=n_clusters, 
                    covariance_type=covariance_type, 
                    random_state=random_state
                )
                
            elif method_upper in ['KMEANS', 'K-MEANS']:
                n_init = method_kwargs.get('kmeans_n_init', 10)
                result = apply_kmeans_clustering(
                    X, n_clusters=n_clusters, 
                    random_state=random_state, 
                    n_init=n_init
                )
                
            elif method_upper in ['HIERARCHICAL', 'AGGLOMERATIVE']:
                linkage_method = method_kwargs.get('hierarchical_linkage', 'ward')
                distance_threshold = method_kwargs.get('hierarchical_distance_threshold', None)
                result = apply_hierarchical_clustering(
                    X, n_clusters=n_clusters, 
                    linkage_method=linkage_method,
                    distance_threshold=distance_threshold
                )
                
            else:
                warnings.warn(f"Unknown clustering method: {method}")
                continue
                
            clustering_results[method] = result
            
        except Exception as e:
            warnings.warn(f"Failed to apply {method}: {e}")
            continue
    
    return clustering_results


def plot_clustering_results(
    clustering_results: Dict[str, Dict[str, Any]],
    X: np.ndarray,
    labels: np.ndarray,
    embedding_method: str = 'PCA',
    condition_names: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize clustering results using dimensionality reduction.
    
    Parameters
    ----------
    clustering_results : dict
        Dictionary of clustering results from apply_multiple_clustering
    X : np.ndarray
        Original feature matrix
    labels : np.ndarray
        True condition labels for comparison
    embedding_method : str
        Dimensionality reduction method for visualization
    condition_names : dict, optional
        Mapping of label values to readable names
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    n_methods = len(clustering_results)
    if n_methods == 0:
        warnings.warn("No clustering results to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No clustering results available', 
               transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Get 2D embedding for visualization
    embedding_result = apply_feature_embedding(X, method=embedding_method, n_components=2)
    X_embedded = embedding_result['embedding']
    
    # Create subplot layout (including true labels)
    n_cols = min(3, n_methods + 1)
    n_rows = int(np.ceil((n_methods + 1) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if (n_methods + 1) == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot true labels first
    ax = axes[0]
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))
    
    for label in unique_labels:
        mask = labels == label
        label_name = condition_names.get(label, str(label)) if condition_names else str(label)
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                  c=[color_map[label]], label=label_name, s=100, alpha=0.8, 
                  edgecolors='black', linewidth=1)
    
    ax.set_xlabel(f'{embedding_method} Component 1')
    ax.set_ylabel(f'{embedding_method} Component 2')
    ax.set_title('True Labels')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot clustering results
    for i, (method_name, result) in enumerate(clustering_results.items(), 1):
        ax = axes[i]
        cluster_labels = result['cluster_labels']
        
        # Handle noise points in DBSCAN
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Create colors for clusters
        if -1 in unique_clusters:  # DBSCAN noise
            cluster_colors = sns.color_palette("husl", n_clusters - 1)
            cluster_color_map = {cluster: color for cluster, color in 
                               zip([c for c in unique_clusters if c != -1], cluster_colors)}
            cluster_color_map[-1] = 'gray'  # Noise points
        else:
            cluster_colors = sns.color_palette("husl", n_clusters)
            cluster_color_map = dict(zip(unique_clusters, cluster_colors))
        
        # Plot clusters
        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            cluster_name = 'Noise' if cluster == -1 else f'Cluster {cluster}'
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                      c=[cluster_color_map[cluster]], label=cluster_name, 
                      s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        ax.set_xlabel(f'{embedding_method} Component 1')
        ax.set_ylabel(f'{embedding_method} Component 2')
        ax.set_title(f'{method_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add clustering metrics
        metrics = result['metrics']
        info_text = f"Clusters: {metrics.get('n_clusters', 'N/A')}"
        if 'silhouette_score' in metrics:
            info_text += f"\nSilhouette: {metrics['silhouette_score']:.3f}"
        if 'noise_ratio' in metrics:
            info_text += f"\nNoise: {metrics['noise_ratio']:.1%}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    # Remove extra subplots
    for i in range(n_methods + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Clustering Analysis Results ({embedding_method} Visualization)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dendrogram(
    hierarchical_result: Dict[str, Any],
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot dendrogram for hierarchical clustering results.
    
    Parameters
    ----------
    hierarchical_result : dict
        Result dictionary from apply_hierarchical_clustering
    labels : np.ndarray, optional
        Sample labels for dendrogram
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if hierarchical_result['method'] != 'Hierarchical':
        raise ValueError("Input must be hierarchical clustering result")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    linkage_matrix = hierarchical_result['linkage_matrix']
    
    # Create dendrogram
    dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10
    )
    
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('Sample Index or Label')
    ax.set_ylabel('Distance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_clustering_methods(
    clustering_results: Dict[str, Dict[str, Any]],
    true_labels: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compare clustering methods using various metrics.
    
    Parameters
    ----------
    clustering_results : dict
        Dictionary of clustering results
    true_labels : np.ndarray, optional
        True labels for external validation metrics
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each method
    """
    comparison_data = []
    
    for method_name, result in clustering_results.items():
        metrics = result['metrics'].copy()
        metrics['method'] = method_name
        
        # Add external validation metrics if true labels provided
        if true_labels is not None:
            cluster_labels = result['cluster_labels']
            
            # Only calculate if we have valid clusters
            if len(np.unique(cluster_labels)) > 1:
                try:
                    metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_labels)
                    metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, cluster_labels)
                except:
                    pass
        
        comparison_data.append(metrics)
    
    return pd.DataFrame(comparison_data)


def analyze_clustering(
    csv_path: str,
    condition_column: str = 'condition',
    exclude_columns: Optional[List[str]] = None,
    methods: List[str] = ['KMEANS', 'GMM', 'DBSCAN', 'HIERARCHICAL'],
    n_clusters: int = 3,
    embedding_method: str = 'PCA',
    condition_names: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    **method_kwargs
) -> Dict[str, Any]:
    """
    Complete pipeline for clustering analysis of feature space data.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing feature data
    condition_column : str
        Column containing condition labels
    exclude_columns : list of str, optional
        Columns to exclude from analysis
    methods : list of str
        Clustering methods to apply
    n_clusters : int
        Number of clusters for methods that require it
    embedding_method : str
        Dimensionality reduction method for visualization
    condition_names : dict, optional
        Mapping of condition labels to readable names
    figsize : tuple
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
    **method_kwargs
        Method-specific parameters
        
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    print("=== Feature Space Clustering Analysis ===")
    
    # Load and prepare data
    X, labels, feature_names = load_feature_data(csv_path, condition_column, exclude_columns)
    
    # Apply clustering methods
    clustering_results = apply_multiple_clustering(X, methods, n_clusters, **method_kwargs)
    
    # Create visualization
    fig = plot_clustering_results(
        clustering_results, X, labels, embedding_method, 
        condition_names, figsize, save_path
    )
    
    # Compare methods
    comparison_df = compare_clustering_methods(clustering_results, labels)
    
    # Print summary statistics
    print("\n=== Clustering Analysis Summary ===")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(labels)}")
    print(f"True conditions: {np.unique(labels)}")
    print(f"Target clusters: {n_clusters}")
    
    print("\n=== Method Comparison ===")
    print(comparison_df.to_string(index=False))
    
    results = {
        'data': X,
        'labels': labels,
        'feature_names': feature_names,
        'clustering_results': clustering_results,
        'comparison': comparison_df,
        'figure': fig
    }
    
    return results
