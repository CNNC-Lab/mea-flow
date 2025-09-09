"""
Cross-condition manifold comparison functions.

This module provides functions to compare manifold structures across
different experimental conditions and analyze condition-specific dynamics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some comparison functions will be disabled.")


def compare_manifolds(
    condition_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare manifold structures across multiple conditions.
    
    Parameters
    ----------
    condition_results : dict
        Dictionary mapping condition names to analysis results
        
    Returns
    -------
    dict
        Dictionary containing comparison results
    """
    if len(condition_results) < 2:
        warnings.warn("Need at least 2 conditions for comparison")
        return {}
    
    results = {}
    
    # 1. Compare population statistics
    results['population_statistics_comparison'] = _compare_population_statistics(condition_results)
    
    # 2. Compare effective dimensionalities
    results['dimensionality_comparison'] = _compare_dimensionalities(condition_results)
    
    # 3. Compare embedding qualities
    results['embedding_quality_comparison'] = _compare_embedding_qualities(condition_results)
    
    # 4. Cross-condition classification
    if SKLEARN_AVAILABLE:
        try:
            results['classification_analysis'] = _cross_condition_classification(condition_results)
        except Exception as e:
            warnings.warn(f"Classification analysis failed: {e}")
            results['classification_analysis'] = {}
    
    # 5. Manifold alignment analysis
    try:
        results['manifold_alignment'] = _analyze_manifold_alignment(condition_results)
    except Exception as e:
        warnings.warn(f"Manifold alignment analysis failed: {e}")
        results['manifold_alignment'] = {}
    
    return results


def _compare_population_statistics(
    condition_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Compare population statistics across conditions."""
    stats_data = []
    
    for condition_name, results in condition_results.items():
        if 'population_statistics' in results:
            stats = results['population_statistics'].copy()
            stats['condition'] = condition_name
            stats_data.append(stats)
    
    if len(stats_data) == 0:
        return pd.DataFrame()
    
    return pd.DataFrame(stats_data)


def _compare_dimensionalities(
    condition_results: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """Compare effective dimensionalities across conditions."""
    dimensionalities = {}
    
    for condition_name, results in condition_results.items():
        if 'effective_dimensionality' in results:
            dimensionalities[condition_name] = results['effective_dimensionality']
    
    return dimensionalities


def _compare_embedding_qualities(
    condition_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Compare embedding quality metrics across conditions."""
    quality_data = []
    
    for condition_name, results in condition_results.items():
        if 'evaluation' in results:
            for method_name, evaluation in results['evaluation'].items():
                row = evaluation.copy()
                row['condition'] = condition_name
                row['method'] = method_name
                quality_data.append(row)
    
    return pd.DataFrame(quality_data)


def _cross_condition_classification(
    condition_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform cross-condition classification analysis.
    
    This analysis tests how well we can distinguish between conditions
    based on their manifold representations.
    """
    results = {}
    
    # Collect embeddings from all conditions
    all_embeddings = {}
    all_labels = []
    condition_names = list(condition_results.keys())
    
    for condition_name, cond_results in condition_results.items():
        if 'embeddings' in cond_results:
            all_embeddings[condition_name] = cond_results['embeddings']
    
    if len(all_embeddings) < 2:
        return {'error': 'Need at least 2 conditions with embeddings'}
    
    # For each embedding method, perform classification
    embedding_methods = set()
    for embeddings in all_embeddings.values():
        embedding_methods.update(embeddings.keys())
    
    classification_scores = {}
    
    for method in embedding_methods:
        # Collect data for this method across conditions
        X_data = []
        y_labels = []
        
        for condition_name in condition_names:
            if (condition_name in all_embeddings and 
                method in all_embeddings[condition_name]):
                
                embedding = all_embeddings[condition_name][method]['embedding']
                n_samples = embedding.shape[0]
                
                X_data.append(embedding)
                y_labels.extend([condition_name] * n_samples)
        
        if len(X_data) < 2:
            continue
        
        # Combine data
        X = np.vstack(X_data)
        y = np.array(y_labels)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Try different classifiers
        classifiers = {
            'LDA': LinearDiscriminantAnalysis(),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        method_scores = {}
        
        for clf_name, classifier in classifiers.items():
            try:
                scores = cross_val_score(classifier, X, y_encoded, cv=5, scoring='accuracy')
                method_scores[clf_name] = {
                    'mean_accuracy': np.mean(scores),
                    'std_accuracy': np.std(scores),
                    'scores': scores.tolist()
                }
            except Exception as e:
                warnings.warn(f"Classification failed for {method}-{clf_name}: {e}")
                method_scores[clf_name] = {'mean_accuracy': np.nan, 'std_accuracy': np.nan}
        
        classification_scores[method] = method_scores
    
    results['classification_scores'] = classification_scores
    results['n_conditions'] = len(condition_names)
    results['condition_names'] = condition_names
    
    return results


def _analyze_manifold_alignment(
    condition_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze alignment between manifolds from different conditions.
    
    This uses Procrustes analysis to measure similarity between
    manifold structures.
    """
    results = {}
    
    condition_names = list(condition_results.keys())
    
    if len(condition_names) < 2:
        return {'error': 'Need at least 2 conditions for alignment analysis'}
    
    # For each embedding method, compare alignments
    embedding_methods = set()
    for cond_results in condition_results.values():
        if 'embeddings' in cond_results:
            embedding_methods.update(cond_results['embeddings'].keys())
    
    alignment_scores = {}
    
    for method in embedding_methods:
        method_alignments = {}
        
        # Compare all pairs of conditions
        for i, cond1 in enumerate(condition_names):
            for j, cond2 in enumerate(condition_names):
                if i >= j:  # Only compute upper triangle
                    continue
                
                # Get embeddings for both conditions
                if (cond1 in condition_results and 
                    'embeddings' in condition_results[cond1] and
                    method in condition_results[cond1]['embeddings'] and
                    cond2 in condition_results and
                    'embeddings' in condition_results[cond2] and
                    method in condition_results[cond2]['embeddings']):
                    
                    emb1 = condition_results[cond1]['embeddings'][method]['embedding']
                    emb2 = condition_results[cond2]['embeddings'][method]['embedding']
                    
                    # Compute alignment score
                    try:
                        alignment = _procrustes_alignment(emb1, emb2)
                        method_alignments[f"{cond1}_vs_{cond2}"] = alignment
                    except Exception as e:
                        warnings.warn(f"Alignment failed for {cond1} vs {cond2}: {e}")
        
        alignment_scores[method] = method_alignments
    
    results['alignment_scores'] = alignment_scores
    results['condition_names'] = condition_names
    
    return results


def _procrustes_alignment(
    X: np.ndarray,
    Y: np.ndarray,
    max_samples: int = 1000
) -> Dict[str, float]:
    """
    Compute Procrustes alignment between two point clouds.
    
    Parameters
    ----------
    X, Y : np.ndarray
        Point clouds to align (n_samples x n_dimensions)
    max_samples : int
        Maximum number of samples to use (for computational efficiency)
        
    Returns
    -------
    dict
        Alignment statistics
    """
    # Subsample if necessary
    n_samples = min(X.shape[0], Y.shape[0], max_samples)
    
    if n_samples < X.shape[0]:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[indices]
    
    if n_samples < Y.shape[0]:
        indices = np.random.choice(Y.shape[0], n_samples, replace=False)
        Y = Y[indices]
    
    # Ensure same dimensionality
    min_dims = min(X.shape[1], Y.shape[1])
    X = X[:, :min_dims]
    Y = Y[:, :min_dims]
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute Procrustes transformation
    try:
        # SVD for optimal rotation
        H = X_centered.T @ Y_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply transformation
        Y_aligned = Y_centered @ R.T
        
        # Calculate alignment metrics
        mse = np.mean((X_centered - Y_aligned)**2)
        
        # Correlation between aligned point clouds
        correlations = []
        for i in range(min_dims):
            corr = np.corrcoef(X_centered[:, i], Y_aligned[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        mean_correlation = np.mean(correlations) if correlations else 0.0
        
        return {
            'procrustes_mse': mse,
            'mean_correlation': mean_correlation,
            'n_dimensions': min_dims,
            'n_samples_used': n_samples
        }
        
    except Exception as e:
        warnings.warn(f"Procrustes analysis failed: {e}")
        return {
            'procrustes_mse': np.inf,
            'mean_correlation': 0.0,
            'n_dimensions': min_dims,
            'n_samples_used': n_samples
        }


def cross_condition_analysis(
    spike_lists: Dict[str, Any],
    analysis_config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive cross-condition analysis.
    
    This is a convenience function that combines manifold analysis
    with cross-condition comparisons.
    
    Parameters
    ----------
    spike_lists : dict
        Dictionary mapping condition names to SpikeList objects
    analysis_config : object, optional
        Analysis configuration
        
    Returns
    -------
    dict
        Comprehensive analysis results
    """
    from .analysis import ManifoldAnalysis
    
    # Initialize analyzer
    analyzer = ManifoldAnalysis(config=analysis_config)
    
    # Perform comparative analysis
    results = analyzer.compare_conditions(spike_lists)
    
    return results


# NOTE: identify_discriminative_features has been moved to mea_flow.analysis.discriminant
# This function is deprecated and will be removed in a future version.
# Please use mea_flow.analysis.discriminant.identify_discriminative_features instead.