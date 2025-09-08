"""
Regularity metrics for MEA data analysis.

This module provides functions to calculate various regularity metrics
such as coefficient of variation, local variation, and entropy measures.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import warnings
from scipy import stats

from ..data import SpikeList


def compute_regularity_metrics(
    spike_list: SpikeList,
    config: Any,
    channels: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive regularity metrics for MEA data.
    
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
        Dictionary containing regularity metrics
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=config.min_isi_samples + 1)
    
    if len(channels) == 0:
        warnings.warn("No channels with sufficient spikes for regularity analysis")
        return _get_empty_regularity_metrics()
    
    if verbose and len(channels) > 20:
        print(f"    Processing {len(channels)} channels for regularity metrics...")
    
    results = {}
    
    # Collect ISI-based metrics for all channels
    cv_isis = []
    lv_isis = []  
    lvr_isis = []
    entropy_isis = []
    isi_5th_percentiles = []
    
    for ch in channels:
        if ch not in spike_list.spike_trains:
            continue
            
        train = spike_list.spike_trains[ch]
        isis = train.get_isi()
        
        if len(isis) < config.min_isi_samples:
            continue
        
        # Coefficient of variation of ISI
        cv = cv_isi(isis)
        if not np.isnan(cv):
            cv_isis.append(cv)
        
        # Local variation of ISI
        lv = lv_isi(isis)
        if not np.isnan(lv):
            lv_isis.append(lv)
        
        # Revised local variation of ISI
        lvr = lvr_isi(isis)
        if not np.isnan(lvr):
            lvr_isis.append(lvr)
        
        # Entropy of ISI distribution
        entropy = entropy_isi(isis)
        if not np.isnan(entropy):
            entropy_isis.append(entropy)
        
        # 5th percentile of ISI (measure of burstiness)
        if len(isis) > 0:
            isi_5th_percentiles.append(np.percentile(isis, 5))
    
    # Aggregate statistics across channels
    if len(cv_isis) > 0:
        results['cv_isi_mean'] = np.mean(cv_isis)
        results['cv_isi_std'] = np.std(cv_isis)
        results['cv_isi_median'] = np.median(cv_isis)
    else:
        results['cv_isi_mean'] = np.nan
        results['cv_isi_std'] = np.nan 
        results['cv_isi_median'] = np.nan
    
    if len(lv_isis) > 0:
        results['lv_isi_mean'] = np.mean(lv_isis)
        results['lv_isi_std'] = np.std(lv_isis)
        results['lv_isi_median'] = np.median(lv_isis)
    else:
        results['lv_isi_mean'] = np.nan
        results['lv_isi_std'] = np.nan
        results['lv_isi_median'] = np.nan
    
    if len(lvr_isis) > 0:
        results['lvr_isi_mean'] = np.mean(lvr_isis)
        results['lvr_isi_std'] = np.std(lvr_isis)
    else:
        results['lvr_isi_mean'] = np.nan
        results['lvr_isi_std'] = np.nan
        
    if len(entropy_isis) > 0:
        results['entropy_isi_mean'] = np.mean(entropy_isis)
        results['entropy_isi_std'] = np.std(entropy_isis)
    else:
        results['entropy_isi_mean'] = np.nan
        results['entropy_isi_std'] = np.nan
    
    if len(isi_5th_percentiles) > 0:
        results['isi_5th_percentile_mean'] = np.mean(isi_5th_percentiles)
        results['isi_5th_percentile_std'] = np.std(isi_5th_percentiles)
    else:
        results['isi_5th_percentile_mean'] = np.nan
        results['isi_5th_percentile_std'] = np.nan
    
    # Population-level regularity measures
    try:
        # Time-binned analysis for population regularity
        bin_size = config.time_bin_size
        spike_matrix, time_bins = spike_list.bin_spikes(bin_size, channels)
        
        # Population activity regularity
        population_activity = np.sum(spike_matrix, axis=0)
        if len(population_activity) > 1 and np.std(population_activity) > 0:
            results['population_cv'] = np.std(population_activity) / np.mean(population_activity)
        else:
            results['population_cv'] = np.nan
        
        # Channel-wise spike count regularity (Fano factor per channel)
        channel_fano_factors = []
        for i in range(spike_matrix.shape[0]):
            channel_counts = spike_matrix[i, :]
            if np.mean(channel_counts) > 0:
                fano = np.var(channel_counts) / np.mean(channel_counts)
                channel_fano_factors.append(fano)
        
        if len(channel_fano_factors) > 0:
            results['fano_factor_mean'] = np.mean(channel_fano_factors)
            results['fano_factor_std'] = np.std(channel_fano_factors)
        else:
            results['fano_factor_mean'] = np.nan
            results['fano_factor_std'] = np.nan
            
    except Exception as e:
        warnings.warn(f"Population regularity analysis failed: {e}")
        results['population_cv'] = np.nan
        results['fano_factor_mean'] = np.nan
        results['fano_factor_std'] = np.nan
    
    # Number of channels contributing to analysis
    results['channels_analyzed'] = len(cv_isis)
    
    return results


def cv_isi(isis: np.ndarray) -> float:
    """
    Calculate coefficient of variation of inter-spike intervals.
    
    CV = std(ISI) / mean(ISI)
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals
        
    Returns
    -------
    float
        Coefficient of variation
    """
    if len(isis) == 0:
        return np.nan
        
    mean_isi = np.mean(isis)
    if mean_isi == 0:
        return np.nan
        
    return np.std(isis) / mean_isi


def lv_isi(isis: np.ndarray) -> float:
    """
    Calculate local variation of inter-spike intervals.
    
    LV measures local spike train irregularity by comparing
    adjacent ISIs. LV = (3/n-1) * sum((ISI[i] - ISI[i+1])^2 / (ISI[i] + ISI[i+1])^2)
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals
        
    Returns
    -------
    float
        Local variation
    """
    if len(isis) < 2:
        return np.nan
    
    n = len(isis)
    lv_sum = 0.0
    
    for i in range(n - 1):
        isi1 = isis[i]
        isi2 = isis[i + 1]
        
        if isi1 + isi2 > 0:
            lv_sum += (isi1 - isi2)**2 / (isi1 + isi2)**2
    
    lv = (3.0 / (n - 1)) * lv_sum
    return lv


def lvr_isi(isis: np.ndarray, refrac_period: float = 0.001) -> float:
    """
    Calculate revised local variation accounting for refractory period.
    
    LVR is a modified version of LV that accounts for the refractory period
    by subtracting the expected refractory period from each ISI.
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals
    refrac_period : float
        Refractory period in seconds (default: 1ms)
        
    Returns
    -------
    float
        Revised local variation
    """
    if len(isis) < 2:
        return np.nan
    
    # Subtract refractory period
    corrected_isis = isis - refrac_period
    
    # Only consider ISIs longer than refractory period
    valid_isis = corrected_isis[corrected_isis > 0]
    
    if len(valid_isis) < 2:
        return np.nan
    
    return lv_isi(valid_isis)


def entropy_isi(isis: np.ndarray, n_bins: Optional[int] = None) -> float:
    """
    Calculate entropy of ISI distribution.
    
    Higher entropy indicates more irregular spike timing.
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals  
    n_bins : int, optional
        Number of bins for histogram (default: sqrt(n))
        
    Returns
    -------
    float
        Entropy in nats (natural log)
    """
    if len(isis) == 0:
        return np.nan
    
    # Determine number of bins
    if n_bins is None:
        n_bins = max(int(np.sqrt(len(isis))), 5)
    
    # Create histogram
    counts, _ = np.histogram(isis, bins=n_bins)
    
    # Normalize to probabilities
    probs = counts / np.sum(counts)
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return np.nan
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs))
    
    return entropy


def burstiness_index(isis: np.ndarray) -> float:
    """
    Calculate burstiness index based on ISI distribution.
    
    B = (σ - μ) / (σ + μ)
    where μ and σ are mean and std of ISIs.
    
    B ranges from -1 (regular) to +1 (bursty).
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals
        
    Returns
    -------
    float
        Burstiness index
    """
    if len(isis) == 0:
        return np.nan
    
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    
    if mean_isi + std_isi == 0:
        return np.nan
    
    return (std_isi - mean_isi) / (std_isi + mean_isi)


def memory_coefficient(isis: np.ndarray) -> float:
    """
    Calculate memory coefficient (serial correlation of ISIs).
    
    M measures correlation between consecutive ISIs.
    
    Parameters
    ----------
    isis : np.ndarray
        Inter-spike intervals
        
    Returns
    -------
    float
        Memory coefficient
    """
    if len(isis) < 2:
        return np.nan
    
    # Calculate correlation between consecutive ISIs
    isi1 = isis[:-1]
    isi2 = isis[1:]
    
    if len(isi1) == 0 or len(isi2) == 0:
        return np.nan
    
    try:
        correlation, _ = stats.pearsonr(isi1, isi2)
        return correlation
    except:
        return np.nan


def regularity_classification(cv: float, lv: float) -> str:
    """
    Classify spike train regularity based on CV and LV values.
    
    Parameters
    ----------
    cv : float
        Coefficient of variation
    lv : float
        Local variation
        
    Returns
    -------
    str
        Classification: 'regular', 'irregular', 'bursty', 'unknown'
    """
    if np.isnan(cv) or np.isnan(lv):
        return 'unknown'
    
    if cv < 0.5 and lv < 0.5:
        return 'regular'
    elif cv > 1.5 or lv > 1.5:
        return 'bursty' 
    else:
        return 'irregular'


def _get_empty_regularity_metrics() -> Dict[str, float]:
    """Return dictionary with NaN values for all regularity metrics."""
    return {
        'cv_isi_mean': np.nan,
        'cv_isi_std': np.nan,
        'cv_isi_median': np.nan,
        'lv_isi_mean': np.nan,
        'lv_isi_std': np.nan, 
        'lv_isi_median': np.nan,
        'lvr_isi_mean': np.nan,
        'lvr_isi_std': np.nan,
        'entropy_isi_mean': np.nan,
        'entropy_isi_std': np.nan,
        'isi_5th_percentile_mean': np.nan,
        'isi_5th_percentile_std': np.nan,
        'population_cv': np.nan,
        'fano_factor_mean': np.nan,
        'fano_factor_std': np.nan,
        'channels_analyzed': 0
    }