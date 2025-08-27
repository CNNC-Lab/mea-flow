"""
Manifold learning and dimensionality reduction implementations.

This module provides functions for applying various dimensionality reduction
and manifold learning techniques to MEA population dynamics data.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import warnings

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE, SpectralEmbedding
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some manifold methods will be disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. UMAP embedding will be disabled.")


def embed_population_dynamics(
    signal_matrix: np.ndarray,
    method: str = 'PCA',
    n_components: int = 3,
    config: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply dimensionality reduction to population activity signals.
    
    Parameters
    ----------
    signal_matrix : np.ndarray
        Signal matrix (n_channels x n_timepoints)
    method : str
        Dimensionality reduction method
    n_components : int
        Number of components to extract
    config : object, optional
        Configuration object with method parameters
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    dict
        Dictionary containing embedding results
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for dimensionality reduction")
    
    # Transpose to (n_timepoints x n_channels) for sklearn
    X = signal_matrix.T
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply the selected method
    if method.upper() == 'PCA':
        result = _apply_pca(X_scaled, n_components, **kwargs)
    elif method.upper() == 'MDS':
        result = _apply_mds(X_scaled, n_components, config, **kwargs)
    elif method.upper() == 'ISOMAP':
        result = _apply_isomap(X_scaled, n_components, config, **kwargs)
    elif method.upper() == 'LLE':
        result = _apply_lle(X_scaled, n_components, config, **kwargs)
    elif method.upper() == 'UMAP':
        result = _apply_umap(X_scaled, n_components, config, **kwargs)
    elif method.upper() == 'T-SNE' or method.upper() == 'TSNE':
        result = _apply_tsne(X_scaled, n_components, config, **kwargs)
    elif method.upper() == 'SPECTRAL':
        result = _apply_spectral(X_scaled, n_components, config, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add common metadata
    result['method'] = method
    result['n_components'] = n_components
    result['original_shape'] = signal_matrix.shape
    result['scaler'] = scaler
    
    return result


def apply_dimensionality_reduction(
    data: np.ndarray,
    methods: List[str],
    n_components: int = 3,
    config: Optional[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Apply multiple dimensionality reduction methods to data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix
    methods : list of str
        List of methods to apply
    n_components : int
        Number of components for each method
    config : object, optional
        Configuration object
        
    Returns
    -------
    dict
        Dictionary mapping method names to results
    """
    results = {}
    
    for method in methods:
        try:
            result = embed_population_dynamics(
                data, method=method, n_components=n_components, config=config
            )
            results[method] = result
        except Exception as e:
            warnings.warn(f"Failed to apply {method}: {e}")
    
    return results


def _apply_pca(X: np.ndarray, n_components: int, **kwargs) -> Dict[str, Any]:
    """Apply Principal Component Analysis."""
    # Adjust n_components if necessary
    max_components = min(X.shape[0], X.shape[1])
    n_components = min(n_components, max_components)
    
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(X)
    
    return {
        'embedding': embedding,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'explained_variance': pca.explained_variance_,
        'components': pca.components_,
        'model': pca
    }


def _apply_mds(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply Multidimensional Scaling.""" 
    n_components = min(n_components, X.shape[0] - 1)
    
    mds = MDS(
        n_components=n_components,
        metric=kwargs.get('metric', True),
        random_state=42,
        max_iter=kwargs.get('max_iter', 300),
        dissimilarity='euclidean'
    )
    
    embedding = mds.fit_transform(X)
    
    return {
        'embedding': embedding,
        'stress': mds.stress_,
        'model': mds
    }


def _apply_isomap(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply Isomap embedding."""
    n_neighbors = kwargs.get('n_neighbors', getattr(config, 'n_neighbors_lle', 10))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    n_components = min(n_components, X.shape[0] - 1)
    
    isomap = Isomap(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=kwargs.get('metric', 'minkowski')
    )
    
    embedding = isomap.fit_transform(X)
    
    return {
        'embedding': embedding,
        'reconstruction_error': isomap.reconstruction_error(),
        'model': isomap
    }


def _apply_lle(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply Locally Linear Embedding."""
    n_neighbors = kwargs.get('n_neighbors', getattr(config, 'n_neighbors_lle', 10))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    n_components = min(n_components, X.shape[0] - 1)
    
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=n_components,
        method=kwargs.get('method', 'standard'),
        random_state=42
    )
    
    embedding = lle.fit_transform(X)
    
    return {
        'embedding': embedding,
        'reconstruction_error': lle.reconstruction_error_,
        'model': lle
    }


def _apply_umap(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply UMAP embedding."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    n_neighbors = kwargs.get('n_neighbors', getattr(config, 'n_neighbors_umap', 15))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=kwargs.get('min_dist', 0.1),
        metric=kwargs.get('metric', 'euclidean'),
        random_state=42
    )
    
    embedding = reducer.fit_transform(X)
    
    return {
        'embedding': embedding,
        'model': reducer
    }


def _apply_tsne(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply t-SNE embedding."""
    perplexity = kwargs.get('perplexity', getattr(config, 'perplexity_tsne', 30.0))
    perplexity = min(perplexity, (X.shape[0] - 1) / 3.0)
    
    # t-SNE is limited to 2 or 3 components typically
    n_components = min(n_components, 3)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=kwargs.get('learning_rate', 'auto'),
        n_iter=kwargs.get('n_iter', 1000),
        random_state=42
    )
    
    embedding = tsne.fit_transform(X)
    
    return {
        'embedding': embedding,
        'kl_divergence': tsne.kl_divergence_,
        'model': tsne
    }


def _apply_spectral(X: np.ndarray, n_components: int, config=None, **kwargs) -> Dict[str, Any]:
    """Apply Spectral Embedding."""
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


def reconstruct_from_embedding(
    embedding: np.ndarray,
    model: Any,
    scaler: Optional[Any] = None
) -> np.ndarray:
    """
    Attempt to reconstruct original data from embedding.
    
    This works primarily for linear methods like PCA.
    
    Parameters
    ----------
    embedding : np.ndarray
        Low-dimensional embedding
    model : object
        Fitted dimensionality reduction model
    scaler : object, optional
        Fitted scaler object
        
    Returns
    -------
    np.ndarray
        Reconstructed data
    """
    if hasattr(model, 'inverse_transform'):
        # Linear methods with direct inverse
        reconstructed = model.inverse_transform(embedding)
    elif hasattr(model, 'components_'):
        # PCA-like reconstruction
        reconstructed = embedding @ model.components_
    else:
        raise ValueError(f"Cannot reconstruct from {type(model)} model")
    
    # Inverse transform scaling if available
    if scaler is not None:
        reconstructed = scaler.inverse_transform(reconstructed)
    
    return reconstructed


def calculate_intrinsic_dimensionality(
    data: np.ndarray,
    method: str = 'pca_variance',
    variance_threshold: float = 0.95
) -> float:
    """
    Estimate intrinsic dimensionality of the data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix (n_samples x n_features)
    method : str
        Method for estimation ('pca_variance', 'participation_ratio')
    variance_threshold : float
        Variance threshold for PCA method
        
    Returns
    -------
    float
        Estimated intrinsic dimensionality
    """
    if method == 'pca_variance':
        pca = PCA()
        pca.fit(data)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumsum_var >= variance_threshold) + 1
        
        return float(intrinsic_dim)
    
    elif method == 'participation_ratio':
        # Participation ratio method
        pca = PCA()
        pca.fit(data)
        
        var_ratios = pca.explained_variance_ratio_
        participation_ratio = (np.sum(var_ratios)**2) / np.sum(var_ratios**2)
        
        return participation_ratio
    
    else:
        raise ValueError(f"Unknown method: {method}")