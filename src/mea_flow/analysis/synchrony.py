"""
Synchrony metrics for MEA data analysis.

This module provides functions to calculate various synchrony measures
including correlation coefficients and distance-based measures from PySpike.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
from itertools import combinations

try:
    import pyspike
    PYSPIKE_AVAILABLE = True
except ImportError:
    PYSPIKE_AVAILABLE = False
    warnings.warn("PySpike not available. Some synchrony metrics will be disabled.")

from ..data import SpikeList


def compute_synchrony_metrics(
    spike_list: SpikeList,
    config: Any,
    channels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive synchrony metrics for MEA data.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    config : AnalysisConfig
        Configuration object with analysis parameters
    channels : list of int, optional
        Channels to include in analysis (default: all active)
        
    Returns
    -------
    dict
        Dictionary containing synchrony metrics
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=10)
    
    if len(channels) < 2:
        warnings.warn("Need at least 2 active channels for synchrony analysis")
        return _get_empty_synchrony_metrics()
    
    results = {}
    
    # Pairwise correlation analysis
    try:
        correlations = pairwise_correlations(
            spike_list, 
            channels=channels,
            bin_size=config.sync_time_bin,
            max_pairs=config.n_pairs_sync
        )
        
        if len(correlations) > 0:
            results['pearson_cc_mean'] = np.mean(correlations)
            results['pearson_cc_std'] = np.std(correlations)
            results['pearson_cc_median'] = np.median(correlations)
            results['pearson_cc_max'] = np.max(correlations)
            results['pearson_cc_min'] = np.min(correlations)
            results['n_pairs_analyzed'] = len(correlations)
        else:
            results.update(_get_empty_correlation_metrics())
    except Exception as e:
        warnings.warn(f"Pairwise correlation analysis failed: {e}")
        results.update(_get_empty_correlation_metrics())
    
    # PySpike distance measures (if available)
    if PYSPIKE_AVAILABLE:
        try:
            spike_distances = spike_distance_measures(
                spike_list,
                channels=channels,
                tau=config.tau_van_rossum
            )
            results.update(spike_distances)
        except Exception as e:
            warnings.warn(f"PySpike distance analysis failed: {e}")
            results.update(_get_empty_distance_metrics())
    else:
        results.update(_get_empty_distance_metrics())
    
    # Van Rossum distance (custom implementation if PySpike not available)
    if not PYSPIKE_AVAILABLE or 'van_rossum_distance' not in results:
        try:
            vr_distances = van_rossum_distance(
                spike_list,
                channels=channels,
                tau=config.tau_van_rossum,
                max_pairs=config.n_pairs_sync
            )
            
            if len(vr_distances) > 0:
                results['van_rossum_distance_mean'] = np.mean(vr_distances)
                results['van_rossum_distance_std'] = np.std(vr_distances)
            else:
                results['van_rossum_distance_mean'] = np.nan
                results['van_rossum_distance_std'] = np.nan
        except Exception as e:
            warnings.warn(f"Van Rossum distance calculation failed: {e}")
            results['van_rossum_distance_mean'] = np.nan
            results['van_rossum_distance_std'] = np.nan
    
    # Population synchrony measures
    try:
        pop_sync = population_synchrony_measures(
            spike_list,
            channels=channels,
            bin_size=config.sync_time_bin
        )
        results.update(pop_sync)
    except Exception as e:
        warnings.warn(f"Population synchrony analysis failed: {e}")
        pop_keys = ['chi_square_distance', 'cosyne_similarity', 'synchrony_index']
        for key in pop_keys:
            results[key] = np.nan
    
    return results


def pairwise_correlations(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    bin_size: float = 0.01,
    max_pairs: Optional[int] = None
) -> List[float]:
    """
    Calculate pairwise Pearson correlation coefficients between channels.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze (default: all active)
    bin_size : float
        Bin size for discretization (seconds)
    max_pairs : int, optional
        Maximum number of pairs to analyze (for computational efficiency)
        
    Returns
    -------
    list of float
        List of correlation coefficients
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=10)
    
    if len(channels) < 2:
        return []
    
    # Create binned spike matrix
    spike_matrix, time_bins = spike_list.bin_spikes(bin_size, channels)
    
    # Generate channel pairs
    all_pairs = list(combinations(range(len(channels)), 2))
    
    # Limit number of pairs if specified
    if max_pairs and len(all_pairs) > max_pairs:
        pairs_to_analyze = np.random.choice(
            len(all_pairs), 
            size=max_pairs, 
            replace=False
        )
        pairs_to_analyze = [all_pairs[i] for i in pairs_to_analyze]
    else:
        pairs_to_analyze = all_pairs
    
    correlations = []
    
    for i, j in pairs_to_analyze:
        x = spike_matrix[i, :]
        y = spike_matrix[j, :]
        
        # Skip if either channel has no activity
        if np.sum(x) == 0 or np.sum(y) == 0:
            continue
        
        # Calculate Pearson correlation
        if np.std(x) > 0 and np.std(y) > 0:
            corr = np.corrcoef(x, y)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    return correlations


def spike_distance_measures(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    tau: float = 0.02
) -> Dict[str, float]:
    """
    Calculate PySpike distance measures.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze
    tau : float
        Time constant for van Rossum distance
        
    Returns
    -------
    dict
        Dictionary with PySpike distance measures
    """
    if not PYSPIKE_AVAILABLE:
        raise ImportError("PySpike required for spike distance measures")
    
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=5)
    
    if len(channels) < 2:
        return _get_empty_distance_metrics()
    
    # Convert to PySpike format
    spike_trains = []
    bounds = [0.0, spike_list.recording_length]
    
    for ch in channels:
        if ch in spike_list.spike_trains and spike_list.spike_trains[ch].n_spikes > 0:
            spike_times = spike_list.spike_trains[ch].spike_times
            spike_train = pyspike.SpikeTrain(spike_times, bounds)
            spike_trains.append(spike_train)
    
    if len(spike_trains) < 2:
        return _get_empty_distance_metrics()
    
    results = {}
    
    try:
        # ISI distance
        isi_dist = pyspike.isi_distance(spike_trains)
        results['isi_distance'] = isi_dist
    except Exception as e:
        warnings.warn(f"ISI distance calculation failed: {e}")
        results['isi_distance'] = np.nan
    
    try:
        # SPIKE distance
        spike_dist = pyspike.spike_distance(spike_trains)
        results['spike_distance'] = spike_dist
    except Exception as e:
        warnings.warn(f"SPIKE distance calculation failed: {e}")
        results['spike_distance'] = np.nan
    
    try:
        # SPIKE synchrony
        spike_sync = pyspike.spike_sync(spike_trains)
        results['spike_sync_distance'] = spike_sync
    except Exception as e:
        warnings.warn(f"SPIKE synchrony calculation failed: {e}")
        results['spike_sync_distance'] = np.nan
    
    return results


def van_rossum_distance(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    tau: float = 0.02,
    max_pairs: Optional[int] = None
) -> List[float]:
    """
    Calculate van Rossum distances between spike trains.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze
    tau : float
        Time constant for exponential kernel (seconds)
    max_pairs : int, optional
        Maximum number of pairs to analyze
        
    Returns
    -------
    list of float
        List of van Rossum distances
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=5)
    
    if len(channels) < 2:
        return []
    
    # Create continuous signals using exponential filter
    dt = 0.001  # 1ms resolution
    signals, time_vector = spike_list.to_continuous_signal(
        tau=tau, 
        channels=channels, 
        dt=dt
    )
    
    # Generate pairs
    all_pairs = list(combinations(range(len(channels)), 2))
    
    if max_pairs and len(all_pairs) > max_pairs:
        pairs_to_analyze = np.random.choice(
            len(all_pairs),
            size=max_pairs,
            replace=False
        )
        pairs_to_analyze = [all_pairs[i] for i in pairs_to_analyze]
    else:
        pairs_to_analyze = all_pairs
    
    distances = []
    
    for i, j in pairs_to_analyze:
        # Calculate squared Euclidean distance between filtered signals
        diff = signals[i, :] - signals[j, :]
        distance = np.sqrt(np.trapz(diff**2, dx=dt))
        distances.append(distance)
    
    return distances


def population_synchrony_measures(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    bin_size: float = 0.01
) -> Dict[str, float]:
    """
    Calculate population-level synchrony measures.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze
    bin_size : float
        Bin size for analysis (seconds)
        
    Returns
    -------
    dict
        Dictionary with population synchrony measures
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=5)
    
    if len(channels) < 2:
        return {'chi_square_distance': np.nan, 'cosyne_similarity': np.nan, 'synchrony_index': np.nan}
    
    # Create binned spike matrix
    spike_matrix, time_bins = spike_list.bin_spikes(bin_size, channels)
    
    results = {}
    
    # Chi-square distance (measure of deviation from independence)
    try:
        chi_sq_dist = _chi_square_distance(spike_matrix)
        results['chi_square_distance'] = chi_sq_dist
    except:
        results['chi_square_distance'] = np.nan
    
    # CoSyNE-style population vector similarity
    try:
        cosyne_sim = _cosyne_similarity(spike_matrix)
        results['cosyne_similarity'] = cosyne_sim
    except:
        results['cosyne_similarity'] = np.nan
    
    # Simple synchrony index (fraction of bins with >1 channel active)
    try:
        sync_index = _synchrony_index(spike_matrix)
        results['synchrony_index'] = sync_index
    except:
        results['synchrony_index'] = np.nan
    
    return results


def _chi_square_distance(spike_matrix: np.ndarray) -> float:
    """Calculate chi-square distance for population synchrony."""
    n_channels, n_bins = spike_matrix.shape
    
    # Calculate expected vs observed coincident spikes
    total_spikes = np.sum(spike_matrix)
    if total_spikes == 0:
        return np.nan
    
    # For each time bin, calculate expected number of active channels
    # under assumption of independence
    mean_rates = np.mean(spike_matrix, axis=1)
    
    chi_sq = 0.0
    for t in range(n_bins):
        observed = np.sum(spike_matrix[:, t] > 0)
        expected = np.sum(mean_rates)  # Expected number of active channels
        
        if expected > 0:
            chi_sq += (observed - expected)**2 / expected
    
    return chi_sq / n_bins


def _cosyne_similarity(spike_matrix: np.ndarray) -> float:
    """Calculate CoSyNE-style population vector similarity."""
    n_channels, n_bins = spike_matrix.shape
    
    if n_bins < 2:
        return np.nan
    
    # Calculate pairwise cosine similarities between time bins
    similarities = []
    
    for i in range(n_bins - 1):
        for j in range(i + 1, n_bins):
            vec1 = spike_matrix[:, i]
            vec2 = spike_matrix[:, j]
            
            # Cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(similarity)
    
    return np.mean(similarities) if len(similarities) > 0 else np.nan


def _synchrony_index(spike_matrix: np.ndarray) -> float:
    """Calculate simple synchrony index."""
    n_channels, n_bins = spike_matrix.shape
    
    # Count bins with more than one active channel
    active_per_bin = np.sum(spike_matrix > 0, axis=0)
    sync_bins = np.sum(active_per_bin > 1)
    
    return sync_bins / n_bins if n_bins > 0 else 0.0


def _get_empty_synchrony_metrics() -> Dict[str, float]:
    """Return dictionary with NaN values for all synchrony metrics."""
    return {
        **_get_empty_correlation_metrics(),
        **_get_empty_distance_metrics(),
        'van_rossum_distance_mean': np.nan,
        'van_rossum_distance_std': np.nan,
        'chi_square_distance': np.nan,
        'cosyne_similarity': np.nan, 
        'synchrony_index': np.nan
    }


def _get_empty_correlation_metrics() -> Dict[str, float]:
    """Return empty correlation metrics."""
    return {
        'pearson_cc_mean': np.nan,
        'pearson_cc_std': np.nan,
        'pearson_cc_median': np.nan,
        'pearson_cc_max': np.nan,
        'pearson_cc_min': np.nan,
        'n_pairs_analyzed': 0
    }


def _get_empty_distance_metrics() -> Dict[str, float]:
    """Return empty distance metrics."""
    return {
        'isi_distance': np.nan,
        'spike_distance': np.nan,
        'spike_sync_distance': np.nan
    }