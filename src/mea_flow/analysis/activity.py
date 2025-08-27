"""
Activity metrics for MEA data analysis.

This module provides functions to calculate various activity-related metrics
such as firing rates, spike counts, and Fano factors.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import warnings

from ..data import SpikeList


def compute_activity_metrics(
    spike_list: SpikeList,
    config: Any,
    channels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive activity metrics for MEA data.
    
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
        Dictionary containing activity metrics
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=config.min_spikes_for_rate)
    
    if len(channels) == 0:
        warnings.warn("No active channels found for activity analysis")
        return _get_empty_activity_metrics()
    
    results = {}
    
    # Basic firing rate statistics
    firing_rates = []
    spike_counts = []
    
    for ch in channels:
        if ch in spike_list.spike_trains:
            train = spike_list.spike_trains[ch]
            firing_rates.append(train.firing_rate)
            spike_counts.append(train.n_spikes)
    
    if len(firing_rates) == 0:
        return _get_empty_activity_metrics()
    
    # Mean firing rate metrics
    results['mean_firing_rate'] = np.mean(firing_rates)
    results['std_firing_rate'] = np.std(firing_rates)
    results['median_firing_rate'] = np.median(firing_rates)
    results['max_firing_rate'] = np.max(firing_rates)
    results['min_firing_rate'] = np.min(firing_rates)
    
    # Spike count metrics
    results['total_spike_count'] = np.sum(spike_counts)
    results['mean_spike_count'] = np.mean(spike_counts)
    results['std_spike_count'] = np.std(spike_counts)
    
    # Channel statistics
    results['active_channels_count'] = len(channels)
    results['total_channels'] = len(spike_list.channel_ids)
    results['activity_fraction'] = len(channels) / len(spike_list.channel_ids)
    
    # Population-level metrics
    results['network_firing_rate'] = results['total_spike_count'] / spike_list.recording_length
    
    # Fano factor (variance-to-mean ratio of spike counts)
    if len(spike_counts) > 1:
        results['fano_factor_mean'] = np.var(spike_counts) / np.mean(spike_counts)
    else:
        results['fano_factor_mean'] = np.nan
    
    # Time-binned activity analysis
    bin_size = config.time_bin_size
    try:
        spike_matrix, time_bins = spike_list.bin_spikes(bin_size, channels)
        
        # Population vector length (measure of coordinated activity)
        pop_vector_lengths = np.sqrt(np.sum(spike_matrix**2, axis=0))
        results['pop_vector_length_mean'] = np.mean(pop_vector_lengths)
        results['pop_vector_length_std'] = np.std(pop_vector_lengths)
        
        # Participation ratio (how many channels are active per bin)
        active_per_bin = np.sum(spike_matrix > 0, axis=0)
        results['participation_ratio_mean'] = np.mean(active_per_bin) / len(channels)
        results['participation_ratio_std'] = np.std(active_per_bin) / len(channels)
        
        # Global activity fluctuations
        global_activity = np.sum(spike_matrix, axis=0)  # Total spikes per bin
        if len(global_activity) > 1:
            results['global_activity_cv'] = np.std(global_activity) / np.mean(global_activity) if np.mean(global_activity) > 0 else np.nan
        else:
            results['global_activity_cv'] = np.nan
            
    except Exception as e:
        warnings.warn(f"Time-binned analysis failed: {e}")
        results['pop_vector_length_mean'] = np.nan
        results['pop_vector_length_std'] = np.nan  
        results['participation_ratio_mean'] = np.nan
        results['participation_ratio_std'] = np.nan
        results['global_activity_cv'] = np.nan
    
    return results


def firing_rate(spike_list: SpikeList, channels: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Calculate firing rates for specified channels.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze (default: all)
        
    Returns
    -------
    dict
        Mapping from channel ID to firing rate (Hz)
    """
    if channels is None:
        channels = spike_list.channel_ids
    
    rates = {}
    for ch in channels:
        if ch in spike_list.spike_trains:
            rates[ch] = spike_list.spike_trains[ch].firing_rate
        else:
            rates[ch] = 0.0
    
    return rates


def burst_detection(
    spike_list: SpikeList, 
    max_isi: float = 0.1,
    min_spikes: int = 5,
    channels: Optional[List[int]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Detect bursts in individual channels.
    
    A burst is defined as a sequence of spikes with inter-spike intervals
    shorter than max_isi and containing at least min_spikes.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    max_isi : float
        Maximum inter-spike interval for burst detection (seconds)
    min_spikes : int
        Minimum number of spikes to constitute a burst
    channels : list of int, optional
        Channels to analyze (default: all active)
        
    Returns
    -------
    dict
        Mapping from channel ID to list of burst dictionaries
        Each burst dict contains: start_time, end_time, duration, n_spikes, mean_isi
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=min_spikes)
    
    all_bursts = {}
    
    for ch in channels:
        if ch not in spike_list.spike_trains:
            all_bursts[ch] = []
            continue
            
        train = spike_list.spike_trains[ch]
        spike_times = train.spike_times
        
        if len(spike_times) < min_spikes:
            all_bursts[ch] = []
            continue
        
        # Calculate ISIs
        isis = np.diff(spike_times)
        
        # Find burst boundaries
        bursts = []
        in_burst = False
        burst_start_idx = 0
        
        for i, isi in enumerate(isis):
            if isi <= max_isi:
                if not in_burst:
                    # Start of new burst
                    in_burst = True
                    burst_start_idx = i
            else:
                if in_burst:
                    # End of current burst
                    burst_end_idx = i
                    
                    # Check if burst meets minimum spike requirement
                    n_spikes_in_burst = burst_end_idx - burst_start_idx + 1
                    if n_spikes_in_burst >= min_spikes:
                        burst_start_time = spike_times[burst_start_idx]
                        burst_end_time = spike_times[burst_end_idx]
                        burst_duration = burst_end_time - burst_start_time
                        burst_isis = isis[burst_start_idx:burst_end_idx]
                        
                        bursts.append({
                            'start_time': burst_start_time,
                            'end_time': burst_end_time,
                            'duration': burst_duration,
                            'n_spikes': n_spikes_in_burst,
                            'mean_isi': np.mean(burst_isis),
                            'spike_indices': (burst_start_idx, burst_end_idx)
                        })
                    
                    in_burst = False
        
        # Check if recording ends during a burst
        if in_burst:
            burst_end_idx = len(spike_times) - 1
            n_spikes_in_burst = burst_end_idx - burst_start_idx + 1
            if n_spikes_in_burst >= min_spikes:
                burst_start_time = spike_times[burst_start_idx]
                burst_end_time = spike_times[burst_end_idx]
                burst_duration = burst_end_time - burst_start_time
                burst_isis = isis[burst_start_idx:burst_end_idx]
                
                bursts.append({
                    'start_time': burst_start_time,
                    'end_time': burst_end_time,
                    'duration': burst_duration,
                    'n_spikes': n_spikes_in_burst,
                    'mean_isi': np.mean(burst_isis) if len(burst_isis) > 0 else 0,
                    'spike_indices': (burst_start_idx, burst_end_idx)
                })
        
        all_bursts[ch] = bursts
    
    return all_bursts


def compute_burst_statistics(bursts: Dict[int, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Compute summary statistics from detected bursts.
    
    Parameters
    ----------
    bursts : dict
        Output from burst_detection function
        
    Returns
    -------
    dict
        Dictionary with burst statistics
    """
    all_burst_durations = []
    all_burst_rates = []
    channel_burst_counts = []
    
    for ch, ch_bursts in bursts.items():
        channel_burst_counts.append(len(ch_bursts))
        
        for burst in ch_bursts:
            all_burst_durations.append(burst['duration'])
            if burst['duration'] > 0:
                all_burst_rates.append(burst['n_spikes'] / burst['duration'])
    
    results = {}
    
    if len(all_burst_durations) > 0:
        results['total_bursts'] = len(all_burst_durations)
        results['burst_duration_mean'] = np.mean(all_burst_durations)
        results['burst_duration_std'] = np.std(all_burst_durations)
        results['burst_rate_mean'] = np.mean(all_burst_rates) if len(all_burst_rates) > 0 else 0
        results['burst_rate_std'] = np.std(all_burst_rates) if len(all_burst_rates) > 0 else 0
        results['bursting_channels'] = np.sum(np.array(channel_burst_counts) > 0)
        results['bursts_per_channel_mean'] = np.mean(channel_burst_counts)
    else:
        results['total_bursts'] = 0
        results['burst_duration_mean'] = np.nan
        results['burst_duration_std'] = np.nan
        results['burst_rate_mean'] = np.nan
        results['burst_rate_std'] = np.nan
        results['bursting_channels'] = 0
        results['bursts_per_channel_mean'] = 0
    
    return results


def _get_empty_activity_metrics() -> Dict[str, float]:
    """Return dictionary with NaN values for all activity metrics."""
    return {
        'mean_firing_rate': np.nan,
        'std_firing_rate': np.nan,
        'median_firing_rate': np.nan,
        'max_firing_rate': np.nan,
        'min_firing_rate': np.nan,
        'total_spike_count': 0,
        'mean_spike_count': np.nan,
        'std_spike_count': np.nan,
        'active_channels_count': 0,
        'total_channels': 0,
        'activity_fraction': 0,
        'network_firing_rate': 0,
        'fano_factor_mean': np.nan,
        'pop_vector_length_mean': np.nan,
        'pop_vector_length_std': np.nan,
        'participation_ratio_mean': np.nan,
        'participation_ratio_std': np.nan,
        'global_activity_cv': np.nan
    }