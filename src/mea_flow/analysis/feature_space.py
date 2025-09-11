"""
Feature space analysis using manifold learning techniques.

This module provides functions for analyzing processed datasets where features
are measurements and samples are individual experimental conditions or observations.
Adapts MEA-Flow's manifold learning capabilities for summary statistics analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE
    from sklearn.metrics import pairwise_distances
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Feature space analysis will be disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_feature_data(
    csv_path: str,
    condition_column: str = 'condition',
    exclude_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load CSV data and prepare for feature space analysis.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing feature data
    condition_column : str
        Column name containing condition labels
    exclude_columns : list of str, optional
        Columns to exclude from analysis (e.g., ['file', 'well'])
    
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples x n_features)
    labels : np.ndarray
        Condition labels for each sample
    feature_names : list of str
        Names of features used in analysis
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for feature space analysis")
    
    df = pd.read_csv(csv_path)
    
    # Get condition labels
    if condition_column in df.columns:
        labels = df[condition_column].values
    else:
        # If no condition column, create generic labels
        labels = np.arange(len(df))
        warnings.warn(f"'{condition_column}' not found. Using row indices as labels.")
    
    # Exclude specified columns
    exclude_cols = [condition_column] if condition_column in df.columns else []
    if exclude_columns:
        exclude_cols.extend(exclude_columns)
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    feature_cols = [col for col in numeric_df.columns if col not in exclude_cols]
    
    X = numeric_df[feature_cols].values
    
    # Handle missing values
    if np.any(np.isnan(X)):
        warnings.warn("Found NaN values. Filling with column means.")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    
    print(f"Data shape: {X.shape} ({len(labels)} samples, {len(feature_cols)} features)")
    print(f"Conditions: {np.unique(labels)}")
    
    return X, labels, feature_cols


def apply_feature_embedding(
    X: np.ndarray,
    method: str = 'PCA',
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply a single manifold learning method to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (n_samples x n_features)
    method : str
        Embedding method ('PCA', 'MDS', 'TSNE', 'UMAP')
    n_components : int
        Number of components for embedding
    random_state : int
        Random state for reproducibility
    **kwargs
        Additional method-specific parameters
        
    Returns
    -------
    dict
        Dictionary containing embedding results and metadata
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for feature embedding")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    method_upper = method.upper()
    
    if method_upper == 'PCA':
        model = PCA(n_components=n_components, random_state=random_state)
        embedding = model.fit_transform(X_scaled)
        result = {
            'embedding': embedding,
            'explained_variance_ratio': model.explained_variance_ratio_,
            'explained_variance': model.explained_variance_,
            'components': model.components_,
            'model': model
        }
        
    elif method_upper == 'MDS':
        model = MDS(
            n_components=n_components,
            random_state=random_state,
            dissimilarity='euclidean',
            max_iter=kwargs.get('max_iter', 300)
        )
        embedding = model.fit_transform(X_scaled)
        result = {
            'embedding': embedding,
            'stress': model.stress_,
            'model': model
        }
        
    elif method_upper in ['TSNE', 'T-SNE']:
        # Adjust perplexity based on sample size
        perplexity = min(kwargs.get('perplexity', 30), (X_scaled.shape[0] - 1) / 3)
        model = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            max_iter=kwargs.get('max_iter', 1000)
        )
        embedding = model.fit_transform(X_scaled)
        result = {
            'embedding': embedding,
            'kl_divergence': model.kl_divergence_,
            'model': model
        }
        
    elif method_upper == 'UMAP':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        # Adjust n_neighbors based on sample size
        n_neighbors = min(kwargs.get('n_neighbors', 15), X_scaled.shape[0] - 1)
        model = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=kwargs.get('min_dist', 0.1)
        )
        embedding = model.fit_transform(X_scaled)
        result = {
            'embedding': embedding,
            'model': model
        }
        
    elif method_upper == 'SPECTRAL':
        result = _apply_spectral(X_scaled, n_components, config=None, **kwargs)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add common metadata
    result.update({
        'method': method,
        'n_components': n_components,
        'original_shape': X.shape,
        'scaler': scaler
    })
    
    return result


def apply_multiple_embeddings(
    X: np.ndarray,
    methods: List[str] = ['PCA', 'MDS', 'TSNE'],
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], StandardScaler]:
    """
    Apply multiple manifold learning methods to feature data.
    
    Parameters
    ----------
    X : np.ndarray
        Input feature matrix
    methods : list of str
        List of embedding methods to apply
    n_components : int
        Number of components for each method
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    embeddings : dict
        Dictionary mapping method names to embedding results
    scaler : StandardScaler
        Fitted scaler object
    """
    embeddings = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for method in methods:
        print(f"Applying {method}...")
        
        try:
            result = apply_feature_embedding(
                X, method=method, n_components=n_components, random_state=random_state
            )
            embeddings[method] = result
            
        except Exception as e:
            warnings.warn(f"Failed to apply {method}: {e}")
            continue
    
    return embeddings, scaler


def plot_feature_embeddings(
    embeddings: Dict[str, Dict[str, Any]],
    labels: np.ndarray,
    condition_names: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison plot of multiple embedding methods.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary of embedding results from apply_multiple_embeddings
    labels : np.ndarray
        Condition labels for color coding
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
    n_methods = len(embeddings)
    if n_methods == 0:
        warnings.warn("No embeddings to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No embeddings available', 
               transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Create color mapping for conditions
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(unique_labels, colors))
    
    # Determine subplot layout
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each embedding
    for i, (method_name, result) in enumerate(embeddings.items()):
        ax = axes[i]
        embedding_data = result['embedding']
        
        # Plot points colored by condition
        for label in unique_labels:
            mask = labels == label
            label_name = condition_names.get(label, str(label)) if condition_names else str(label)
            ax.scatter(embedding_data[mask, 0], embedding_data[mask, 1], 
                      c=[color_map[label]], label=label_name, s=100, alpha=0.8, 
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'{method_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add method-specific information
        if method_name.upper() == 'PCA' and 'explained_variance_ratio' in result:
            var_ratio = result['explained_variance_ratio']
            info_text = f'Explained variance:\nPC1: {var_ratio[0]:.1%}'
            if len(var_ratio) > 1:
                info_text += f'\nPC2: {var_ratio[1]:.1%}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8))
        
        elif 'stress' in result:
            ax.text(0.02, 0.98, f'Stress: {result["stress"]:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove extra subplots
    for i in range(n_methods, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Feature Space Manifold Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_feature_space(
    csv_path: str,
    condition_column: str = 'condition',
    exclude_columns: Optional[List[str]] = None,
    methods: List[str] = ['PCA', 'MDS', 'TSNE'],
    condition_names: Optional[Dict[Any, str]] = None,
    n_components: int = 2,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete pipeline for feature space analysis using manifold learning.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing feature data
    condition_column : str
        Column containing condition labels
    exclude_columns : list of str, optional
        Columns to exclude from analysis
    methods : list of str
        Manifold learning methods to apply
    condition_names : dict, optional
        Mapping of condition labels to readable names
    n_components : int
        Number of components for embedding
    figsize : tuple
        Figure size for visualization
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    dict
        Dictionary containing all analysis results
    """
    print("=== Feature Space Manifold Analysis ===")
    
    # Load and prepare data
    X, labels, feature_names = load_feature_data(csv_path, condition_column, exclude_columns)
    
    # Apply manifold methods
    embeddings, scaler = apply_multiple_embeddings(X, methods, n_components)
    
    # Create visualization
    fig = plot_feature_embeddings(embeddings, labels, condition_names, figsize, save_path)
    
    # Calculate pairwise distances in original space
    X_scaled = scaler.transform(X)
    distances = pairwise_distances(X_scaled)
    
    # Print summary statistics
    print("\n=== Analysis Summary ===")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of samples: {len(labels)}")
    print(f"Conditions: {np.unique(labels)}")
    
    for method, result in embeddings.items():
        print(f"\n{method}:")
        if 'explained_variance_ratio' in result:
            total_var = np.sum(result['explained_variance_ratio'])
            print(f"  Total explained variance: {total_var:.1%}")
        if 'stress' in result:
            print(f"  Stress: {result['stress']:.4f}")
        if 'kl_divergence' in result:
            print(f"  KL divergence: {result['kl_divergence']:.4f}")
    
    results = {
        'data': X,
        'labels': labels,
        'feature_names': feature_names,
        'embeddings': embeddings,
        'scaler': scaler,
        'distances': distances,
        'figure': fig
    }
    
    return results


def get_feature_importance(
    embedding_result: Dict[str, Any],
    feature_names: List[str],
    n_top: int = 10
) -> pd.DataFrame:
    """
    Extract feature importance from PCA embedding results.
    
    Parameters
    ----------
    embedding_result : dict
        Result dictionary from apply_feature_embedding with method='PCA'
    feature_names : list of str
        Names of the original features
    n_top : int
        Number of top features to return for each component
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    if embedding_result['method'].upper() != 'PCA':
        raise ValueError("Feature importance only available for PCA method")
    
    components = embedding_result['components']
    n_components = components.shape[0]
    
    importance_data = []
    
    for i in range(n_components):
        # Get absolute loadings for this component
        loadings = np.abs(components[i])
        
        # Get top features
        top_indices = np.argsort(loadings)[::-1][:n_top]
        
        for rank, idx in enumerate(top_indices):
            importance_data.append({
                'component': f'PC{i+1}',
                'feature': feature_names[idx],
                'loading': components[i, idx],
                'abs_loading': loadings[idx],
                'rank': rank + 1
            })
    
    return pd.DataFrame(importance_data)


def _apply_spectral(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply Spectral Embedding."""
    from sklearn.manifold import SpectralEmbedding
    
    n_neighbors = kwargs.get('n_neighbors', getattr(config, 'n_neighbors_lle', 10))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    n_components = min(n_components, X.shape[0] - 1)
    
    spectral = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity=kwargs.get('affinity', 'nearest_neighbors'),
        random_state=42
    )
    
    embedding = spectral.fit_transform(X)
    
    return {
        'embedding': embedding,
        'model': spectral
    }
