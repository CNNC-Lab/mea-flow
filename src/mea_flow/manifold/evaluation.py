"""
Evaluation functions for manifold learning and dimensionality reduction.

This module provides functions to assess the quality of embeddings
and compare different dimensionality reduction methods.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some evaluation metrics will be disabled.")


def evaluate_embedding(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    method_name: str,
    k_neighbors: int = 10
) -> Dict[str, float]:
    """
    Evaluate the quality of a dimensionality reduction embedding.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original high-dimensional data (n_channels x n_timepoints)
    embedded_data : np.ndarray
        Low-dimensional embedding (n_timepoints x n_dimensions)
    method_name : str
        Name of the embedding method
    k_neighbors : int
        Number of neighbors for neighborhood-based metrics
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn required for evaluation metrics")
        return {}
    
    # Transpose original data to match embedded data orientation
    X_original = original_data.T  # (n_timepoints x n_channels)
    X_embedded = embedded_data   # (n_timepoints x n_dimensions)
    
    results = {}
    
    # 1. Reconstruction error (for methods that support it)
    try:
        recon_error = reconstruction_error(X_original, X_embedded)
        results['reconstruction_error'] = recon_error
    except Exception as e:
        warnings.warn(f"Could not calculate reconstruction error: {e}")
        results['reconstruction_error'] = np.nan
    
    # 2. Stress (for distance-preservation methods)
    try:
        stress = calculate_stress(X_original, X_embedded)
        results['stress'] = stress
    except Exception as e:
        warnings.warn(f"Could not calculate stress: {e}")
        results['stress'] = np.nan
    
    # 3. Trustworthiness and continuity
    try:
        trust = trustworthiness(X_original, X_embedded, k=k_neighbors)
        results['trustworthiness'] = trust
    except Exception as e:
        warnings.warn(f"Could not calculate trustworthiness: {e}")
        results['trustworthiness'] = np.nan
    
    try:
        cont = continuity(X_original, X_embedded, k=k_neighbors)
        results['continuity'] = cont
    except Exception as e:
        warnings.warn(f"Could not calculate continuity: {e}")
        results['continuity'] = np.nan
    
    # 4. Neighborhood preservation
    try:
        neighb_pres = neighborhood_preservation(X_original, X_embedded, k=k_neighbors)
        results['neighborhood_preservation'] = neighb_pres
    except Exception as e:
        warnings.warn(f"Could not calculate neighborhood preservation: {e}")
        results['neighborhood_preservation'] = np.nan
    
    # 5. Correlation between distance matrices
    try:
        dist_corr = distance_correlation(X_original, X_embedded)
        results['distance_correlation'] = dist_corr
    except Exception as e:
        warnings.warn(f"Could not calculate distance correlation: {e}")
        results['distance_correlation'] = np.nan
    
    # 6. Effective dimensionality of embedding
    try:
        eff_dim = effective_dimensionality(X_embedded)
        results['embedding_effective_dimensionality'] = eff_dim
    except Exception as e:
        warnings.warn(f"Could not calculate effective dimensionality: {e}")
        results['embedding_effective_dimensionality'] = np.nan
    
    # 7. Linear readout performance (if enough dimensions)
    if X_embedded.shape[1] >= 2:
        try:
            readout_score = linear_readout_performance(X_original, X_embedded)
            results['linear_readout_score'] = readout_score
        except Exception as e:
            warnings.warn(f"Could not calculate readout performance: {e}")
            results['linear_readout_score'] = np.nan
    else:
        results['linear_readout_score'] = np.nan
    
    return results


def reconstruction_error(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    method: str = 'linear'
) -> float:
    """
    Calculate reconstruction error from embedding back to original space.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data (n_samples x n_features_original)
    embedded_data : np.ndarray
        Embedded data (n_samples x n_features_embedded)
    method : str
        Reconstruction method ('linear', 'polynomial')
        
    Returns
    -------
    float
        Mean squared reconstruction error
    """
    if method == 'linear':
        # Use linear regression to map embedding back to original space
        regressor = LinearRegression()
        regressor.fit(embedded_data, original_data)
        reconstructed = regressor.predict(embedded_data)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    
    # Calculate MSE
    mse = np.mean((original_data - reconstructed)**2)
    return mse


def calculate_stress(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Calculate stress (Kruskal's stress formula).
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray
        Embedded data
    metric : str
        Distance metric to use
        
    Returns
    -------
    float
        Stress value
    """
    # Calculate pairwise distances
    D_original = pairwise_distances(original_data, metric=metric)
    D_embedded = pairwise_distances(embedded_data, metric=metric)
    
    # Flatten upper triangular matrices (exclude diagonal)
    mask = np.triu(np.ones_like(D_original, dtype=bool), k=1)
    d_orig = D_original[mask]
    d_emb = D_embedded[mask]
    
    # Calculate stress
    numerator = np.sum((d_orig - d_emb)**2)
    denominator = np.sum(d_orig**2)
    
    stress = np.sqrt(numerator / denominator) if denominator > 0 else np.inf
    return stress


def trustworthiness(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate trustworthiness of the embedding.
    
    Trustworthiness measures whether points that are close in the 
    embedding are also close in the original space.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray
        Embedded data
    k : int
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Trustworthiness score (0-1, higher is better)
    """
    n_samples = original_data.shape[0]
    k = min(k, n_samples - 1)
    
    # Find k nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1)
    nbrs_embedded.fit(embedded_data)
    _, indices_embedded = nbrs_embedded.kneighbors(embedded_data)
    
    # Find k nearest neighbors in original space  
    nbrs_original = NearestNeighbors(n_neighbors=n_samples)
    nbrs_original.fit(original_data)
    _, indices_original = nbrs_original.kneighbors(original_data)
    
    trustworthiness_sum = 0.0
    
    for i in range(n_samples):
        # Get k nearest neighbors in embedded space (excluding self)
        nn_embedded = set(indices_embedded[i, 1:k+1])
        
        # Get ranks in original space
        ranks_original = {idx: rank for rank, idx in enumerate(indices_original[i])}
        
        for j in nn_embedded:
            rank_j = ranks_original.get(j, n_samples)
            if rank_j > k:
                trustworthiness_sum += rank_j - k
    
    # Normalize
    max_sum = k * (2 * n_samples - 3 * k - 1) * n_samples / 2
    trustworthiness = 1 - (2 * trustworthiness_sum) / max_sum
    
    return trustworthiness


def continuity(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate continuity of the embedding.
    
    Continuity measures whether points that are close in the original
    space are also close in the embedding.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray  
        Embedded data
    k : int
        Number of nearest neighbors to consider
        
    Returns
    -------
    float
        Continuity score (0-1, higher is better)
    """
    n_samples = original_data.shape[0]
    k = min(k, n_samples - 1)
    
    # Find k nearest neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=k + 1)
    nbrs_original.fit(original_data)
    _, indices_original = nbrs_original.kneighbors(original_data)
    
    # Find nearest neighbors in embedded space
    nbrs_embedded = NearestNeighbors(n_neighbors=n_samples)
    nbrs_embedded.fit(embedded_data)
    _, indices_embedded = nbrs_embedded.kneighbors(embedded_data)
    
    continuity_sum = 0.0
    
    for i in range(n_samples):
        # Get k nearest neighbors in original space (excluding self)
        nn_original = set(indices_original[i, 1:k+1])
        
        # Get ranks in embedded space
        ranks_embedded = {idx: rank for rank, idx in enumerate(indices_embedded[i])}
        
        for j in nn_original:
            rank_j = ranks_embedded.get(j, n_samples)
            if rank_j > k:
                continuity_sum += rank_j - k
    
    # Normalize
    max_sum = k * (2 * n_samples - 3 * k - 1) * n_samples / 2
    continuity = 1 - (2 * continuity_sum) / max_sum
    
    return continuity


def neighborhood_preservation(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    k: int = 5
) -> float:
    """
    Calculate neighborhood preservation.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray
        Embedded data
    k : int
        Number of nearest neighbors
        
    Returns
    -------
    float
        Neighborhood preservation score (0-1, higher is better)
    """
    n_samples = original_data.shape[0]
    k = min(k, n_samples - 1)
    
    # Find k nearest neighbors in both spaces
    nbrs_original = NearestNeighbors(n_neighbors=k + 1)
    nbrs_original.fit(original_data)
    _, indices_original = nbrs_original.kneighbors(original_data)
    
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1)
    nbrs_embedded.fit(embedded_data)
    _, indices_embedded = nbrs_embedded.kneighbors(embedded_data)
    
    preservation_sum = 0.0
    
    for i in range(n_samples):
        # Get neighbors (excluding self)
        nn_original = set(indices_original[i, 1:k+1])
        nn_embedded = set(indices_embedded[i, 1:k+1])
        
        # Calculate overlap
        overlap = len(nn_original.intersection(nn_embedded))
        preservation_sum += overlap / k
    
    return preservation_sum / n_samples


def distance_correlation(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Calculate correlation between distance matrices.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray
        Embedded data
    metric : str
        Distance metric
        
    Returns
    -------
    float
        Pearson correlation between distance matrices
    """
    # Calculate distance matrices
    D_original = pairwise_distances(original_data, metric=metric)
    D_embedded = pairwise_distances(embedded_data, metric=metric)
    
    # Extract upper triangular parts (excluding diagonal)
    mask = np.triu(np.ones_like(D_original, dtype=bool), k=1)
    d_orig = D_original[mask]
    d_emb = D_embedded[mask]
    
    # Calculate correlation
    correlation = np.corrcoef(d_orig, d_emb)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def effective_dimensionality(data: np.ndarray) -> float:
    """
    Calculate effective dimensionality using PCA.
    
    Based on the participation ratio measure.
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix
        
    Returns
    -------
    float
        Effective dimensionality
    """
    try:
        from sklearn.decomposition import PCA
        
        # Standardize data
        data_centered = data - np.mean(data, axis=0)
        
        # Apply PCA
        pca = PCA()
        pca.fit(data_centered)
        
        # Calculate effective dimensionality
        eigenvalues = pca.explained_variance_
        
        # Participation ratio
        numerator = (np.sum(eigenvalues))**2
        denominator = np.sum(eigenvalues**2)
        
        eff_dim = numerator / denominator if denominator > 0 else np.nan
        
        return eff_dim
        
    except Exception as e:
        warnings.warn(f"Could not calculate effective dimensionality: {e}")
        return np.nan


def linear_readout_performance(
    original_data: np.ndarray,
    embedded_data: np.ndarray,
    cv_folds: int = 5
) -> float:
    """
    Assess how well the embedding preserves linear decodability.
    
    Parameters
    ----------
    original_data : np.ndarray
        Original data
    embedded_data : np.ndarray
        Embedded data
    cv_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    float
        Cross-validated R² score
    """
    # Use the first principal component of original data as target
    try:
        from sklearn.decomposition import PCA
        
        pca_target = PCA(n_components=1)
        target = pca_target.fit_transform(original_data).ravel()
        
        # Train linear regression to predict target from embedding
        regressor = LinearRegression()
        scores = cross_val_score(regressor, embedded_data, target, 
                               cv=cv_folds, scoring='r2')
        
        return np.mean(scores)
        
    except Exception as e:
        warnings.warn(f"Could not calculate readout performance: {e}")
        return np.nan


def compare_embedding_quality(
    evaluations: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compare multiple embeddings and provide overall quality scores.
    
    Parameters
    ----------
    evaluations : dict
        Dictionary mapping method names to evaluation dictionaries
    weights : dict, optional
        Weights for different metrics
        
    Returns
    -------
    dict
        Dictionary with overall quality scores for each method
    """
    if weights is None:
        weights = {
            'trustworthiness': 0.3,
            'continuity': 0.3,
            'reconstruction_error': -0.2,  # Negative because lower is better
            'stress': -0.1,  # Negative because lower is better
            'distance_correlation': 0.1
        }
    
    overall_scores = {}
    
    for method_name, eval_dict in evaluations.items():
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in eval_dict and not np.isnan(eval_dict[metric]):
                if weight < 0:  # For metrics where lower is better
                    # Normalize by converting to 0-1 scale (roughly)
                    normalized_value = 1.0 / (1.0 + eval_dict[metric])
                    score += abs(weight) * normalized_value
                else:
                    score += weight * eval_dict[metric]
                total_weight += abs(weight)
        
        if total_weight > 0:
            overall_scores[method_name] = score / total_weight
        else:
            overall_scores[method_name] = 0.0
    
    return overall_scores