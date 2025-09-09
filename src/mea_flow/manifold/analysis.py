"""
Main manifold analysis class for MEA population dynamics.

This module provides the ManifoldAnalysis class that orchestrates
manifold learning and dimensionality reduction for MEA data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from dataclasses import dataclass

from ..data import SpikeList
from .embedding import embed_population_dynamics, apply_dimensionality_reduction
from .evaluation import evaluate_embedding, effective_dimensionality
from .comparison import compare_manifolds


@dataclass
class ManifoldConfig:
    """Configuration for manifold analysis."""
    
    # Signal preprocessing
    tau: float = 0.02  # Exponential filter time constant (s)
    dt: float = 0.001  # Sampling interval for continuous signals (s)
    
    # Dimensionality reduction
    max_components: int = 20
    random_state: int = 42
    
    # Manifold learning methods to use
    methods: List[str] = None
    
    # Subsampling for embeddings (applied before all methods)
    max_embedding_samples: int = 50000  # Max timepoints for embedding methods
    embedding_sampling_method: str = 'regular'  # 'regular' or 'random'
    
    # Evaluation parameters
    n_neighbors_lle: int = 10
    perplexity_tsne: float = 30.0
    n_neighbors_umap: int = 15
    
    # Population statistics parameters
    max_distance_samples: int = 10000  # Max timepoints for distance calculations
    use_regular_sampling: bool = True   # Use regular intervals vs random sampling
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ['PCA']  # Default to PCA only to avoid memory issues


class ManifoldAnalysis:
    """
    Main class for manifold learning analysis of MEA population dynamics.
    
    This class provides a comprehensive pipeline for analyzing the geometry
    of neural population activity using various dimensionality reduction
    and manifold learning techniques.
    """
    
    def __init__(self, config: Optional[ManifoldConfig] = None):
        """
        Initialize ManifoldAnalysis.
        
        Parameters
        ----------
        config : ManifoldConfig, optional
            Configuration parameters for analysis
        """
        self.config = config if config is not None else ManifoldConfig()
        self.results = {}
        
    def analyze_population_dynamics(
        self,
        spike_list: SpikeList,
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive manifold analysis of population dynamics.
        
        Parameters
        ----------
        spike_list : SpikeList
            Input spike data
        channels : list of int, optional
            Channels to include in analysis
        time_range : tuple of float, optional
            Time range to analyze (start, end) in seconds
            
        Returns
        -------
        dict
            Dictionary containing all analysis results
        """
        # Filter data if needed
        if time_range is not None:
            start_time, end_time = time_range
            analysis_data = spike_list.get_time_window(start_time, end_time, channels)
        else:
            analysis_data = spike_list
            
        if channels is not None:
            from ..data.preprocessing import filter_channels
            analysis_data = filter_channels(analysis_data, channels)
        
        # Get active channels
        active_channels = analysis_data.get_active_channels(min_spikes=10)
        
        if len(active_channels) < 3:
            warnings.warn("Need at least 3 active channels for manifold analysis")
            return self._get_empty_results()
        
        # Step 1: Convert to continuous signals
        print("Converting spike trains to continuous signals...")
        signal_matrix, time_vector = analysis_data.to_continuous_signal(
            tau=self.config.tau,
            channels=active_channels,
            dt=self.config.dt
        )
        
        # Step 2: Calculate basic population statistics
        pop_stats = self._calculate_population_statistics(signal_matrix)
        
        # Step 3: Subsample signal matrix for embedding methods if needed
        embedding_matrix = signal_matrix
        embedding_time_vector = time_vector
        
        if signal_matrix.shape[1] > self.config.max_embedding_samples:
            print(f"Subsampling signal matrix for embeddings: {signal_matrix.shape[1]} -> {self.config.max_embedding_samples} timepoints")
            
            if self.config.embedding_sampling_method == 'regular':
                step = signal_matrix.shape[1] // self.config.max_embedding_samples
                sample_indices = np.arange(0, signal_matrix.shape[1], step)[:self.config.max_embedding_samples]
                print(f"   Regular sampling: every {step} timepoints")
            else:
                sample_indices = np.random.choice(signal_matrix.shape[1], self.config.max_embedding_samples, replace=False)
                sample_indices = np.sort(sample_indices)
                print(f"   Random sampling: {self.config.max_embedding_samples} timepoints")
            
            embedding_matrix = signal_matrix[:, sample_indices]
            embedding_time_vector = time_vector[sample_indices]
            print(f"   Embedding matrix shape: {embedding_matrix.shape}")
        
        # Step 4: Apply dimensionality reduction methods
        print("Applying dimensionality reduction methods...")
        embeddings = {}
        evaluation = {}
        
        for method in self.config.methods:
            try:
                print(f"   Applying {method}...")
                embedding_result = embed_population_dynamics(
                    embedding_matrix,
                    method=method,
                    n_components=min(self.config.max_components, embedding_matrix.shape[0]),
                    config=self.config
                )
                
                embeddings[method] = embedding_result
                
                # Evaluate embedding quality
                eval_result = evaluate_embedding(
                    embedding_matrix.T,
                    embedding_result['embedding'],
                    method=method
                )
                evaluation[method] = eval_result
                
                print(f"   ✓ {method} completed")
                
            except Exception as e:
                warnings.warn(f"Failed to evaluate {method}: {e}")
        
        # Step 5: Calculate effective dimensionality
        print("Calculating effective dimensionality...")
        try:
            eff_dim = effective_dimensionality(signal_matrix)
        except Exception as e:
            warnings.warn(f"Failed to calculate effective dimensionality: {e}")
            eff_dim = np.nan
        
        # Compile results
        results = {
            'signal_matrix': signal_matrix,
            'time_vector': time_vector,
            'active_channels': active_channels,
            'population_statistics': pop_stats,
            'embeddings': embeddings,
            'evaluation': evaluation,
            'effective_dimensionality': eff_dim,
            'config': self.config
        }
        
        self.results = results
        return results
    
    def compare_conditions(
        self,
        spike_lists: Dict[str, SpikeList],
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Compare manifold structure across multiple experimental conditions.
        
        Parameters
        ----------
        spike_lists : dict
            Dictionary mapping condition names to SpikeList objects
        channels : list of int, optional
            Channels to include in analysis
        time_range : tuple of float, optional
            Time range to analyze
            
        Returns
        -------
        dict
            Dictionary with comparative analysis results
        """
        condition_results = {}
        
        # Analyze each condition
        print("Analyzing individual conditions...")
        for condition_name, spike_list in spike_lists.items():
            print(f"\nProcessing condition: {condition_name}")
            
            try:
                result = self.analyze_population_dynamics(
                    spike_list,
                    channels=channels,
                    time_range=time_range
                )
                condition_results[condition_name] = result
            except Exception as e:
                warnings.warn(f"Failed to analyze condition {condition_name}: {e}")
        
        # Cross-condition comparison
        print("\nPerforming cross-condition comparison...")
        try:
            comparison_result = compare_manifolds(condition_results)
        except Exception as e:
            warnings.warn(f"Failed to compare manifolds: {e}")
            comparison_result = {}
        
        return {
            'individual_results': condition_results,
            'comparison': comparison_result
        }
    
    def _calculate_population_statistics(self, signal_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate basic population-level statistics."""
        n_channels, n_timepoints = signal_matrix.shape
        
        stats = {}
        
        # Population activity measures
        population_activity = np.sum(signal_matrix, axis=0)
        stats['mean_population_activity'] = np.mean(population_activity)
        stats['std_population_activity'] = np.std(population_activity)
        stats['cv_population_activity'] = (
            stats['std_population_activity'] / stats['mean_population_activity']
            if stats['mean_population_activity'] > 0 else np.nan
        )
        
        # Pairwise correlations
        from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
        
        # Correlation between channels
        channel_corr_matrix = np.corrcoef(signal_matrix)
        # Remove diagonal (self-correlations)
        off_diag_corr = channel_corr_matrix[np.triu_indices(n_channels, k=1)]
        stats['mean_pairwise_correlation'] = np.mean(off_diag_corr)
        stats['std_pairwise_correlation'] = np.std(off_diag_corr)
        
        # Population vector statistics
        pop_vector_norms = np.sqrt(np.sum(signal_matrix**2, axis=0))
        stats['mean_pop_vector_norm'] = np.mean(pop_vector_norms)
        stats['std_pop_vector_norm'] = np.std(pop_vector_norms)
        
        # Distance measures - subsample for memory efficiency
        max_samples = self.config.max_distance_samples
        if n_timepoints > max_samples:
            if self.config.use_regular_sampling:
                # Regular interval sampling to preserve temporal structure
                step = n_timepoints // max_samples
                sample_indices = np.arange(0, n_timepoints, step)[:max_samples]
                print(f"   Regular sampling: {len(sample_indices)} timepoints (every {step} samples)")
            else:
                # Random sampling
                sample_indices = np.random.choice(n_timepoints, max_samples, replace=False)
                sample_indices = np.sort(sample_indices)  # Sort for temporal order
                print(f"   Random sampling: {max_samples} timepoints")
            
            sample_matrix = signal_matrix[:, sample_indices]
            euclidean_dist = euclidean_distances(sample_matrix.T)
            triu_indices = np.triu_indices(len(sample_indices), k=1)
            euclidean_dists = euclidean_dist[triu_indices]
        else:
            euclidean_dist = euclidean_distances(signal_matrix.T)
            triu_indices = np.triu_indices(n_timepoints, k=1)
            euclidean_dists = euclidean_dist[triu_indices]
        
        stats['mean_euclidean_distance'] = np.mean(euclidean_dists)
        stats['std_euclidean_distance'] = np.std(euclidean_dists)
        
        # Cosine similarity - use same sampling strategy
        if n_timepoints > max_samples and 'sample_matrix' in locals():
            cosine_dist = cosine_distances(sample_matrix.T)
            cosine_dists = cosine_dist[triu_indices]
        else:
            cosine_dist = cosine_distances(signal_matrix.T)
            cosine_dists = cosine_dist[triu_indices]
        
        stats['mean_cosine_distance'] = np.mean(cosine_dists)
        stats['std_cosine_distance'] = np.std(cosine_dists)
        
        return stats
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'signal_matrix': np.array([]),
            'time_vector': np.array([]),
            'active_channels': [],
            'population_statistics': {},
            'embeddings': {},
            'evaluation': {},
            'effective_dimensionality': np.nan,
            'config': self.config
        }
    
    def get_best_embedding(self, criterion: str = 'reconstruction_error') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best embedding method based on evaluation criterion.
        
        Parameters
        ----------
        criterion : str
            Evaluation criterion ('reconstruction_error', 'stress', 'trustworthiness')
            
        Returns
        -------
        tuple
            (method_name, embedding_result)
        """
        if 'evaluation' not in self.results or len(self.results['evaluation']) == 0:
            raise ValueError("No evaluation results available")
        
        evaluations = self.results['evaluation']
        
        if criterion == 'reconstruction_error':
            # Lower is better
            best_method = min(evaluations.keys(), 
                            key=lambda m: evaluations[m].get('reconstruction_error', float('inf')))
        elif criterion == 'stress':
            # Lower is better
            best_method = min(evaluations.keys(),
                            key=lambda m: evaluations[m].get('stress', float('inf')))
        elif criterion == 'trustworthiness':
            # Higher is better
            best_method = max(evaluations.keys(),
                            key=lambda m: evaluations[m].get('trustworthiness', -float('inf')))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return best_method, self.results['embeddings'][best_method]
    
    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame of all analysis results."""
        if 'evaluation' not in self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for method, evaluation in self.results['evaluation'].items():
            row = {
                'method': method,
                'effective_dimensionality': self.results.get('effective_dimensionality', np.nan),
                **evaluation
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)