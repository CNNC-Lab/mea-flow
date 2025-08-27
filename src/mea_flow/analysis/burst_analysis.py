"""
Burst analysis for MEA data.

This module provides functions for detecting and analyzing bursts at both
individual channel and network levels, following MEA-specific protocols.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings

from ..data import SpikeList
from .activity import burst_detection


def network_burst_analysis(
    spike_list: SpikeList,
    config: Any,
    channels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Perform comprehensive burst analysis including network bursts.
    
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
        Dictionary containing burst metrics
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=config.min_burst_spikes)
    
    if len(channels) < 2:
        warnings.warn("Need at least 2 active channels for network burst analysis")
        return _get_empty_burst_metrics()
    
    results = {}
    
    # Individual channel burst detection
    if config.burst_detection:
        try:
            individual_bursts = burst_detection(
                spike_list,
                max_isi=config.max_isi_burst,
                min_spikes=config.min_burst_spikes,
                channels=channels
            )
            
            # Aggregate individual burst statistics
            burst_stats = _aggregate_burst_statistics(individual_bursts, spike_list.recording_length)
            results.update(burst_stats)
            
        except Exception as e:
            warnings.warn(f"Individual burst detection failed: {e}")
            results.update(_get_empty_individual_burst_metrics())
    else:
        results.update(_get_empty_individual_burst_metrics())
    
    # Network burst detection
    if config.network_burst_detection:
        try:
            network_bursts = detect_network_bursts(
                spike_list,
                channels=channels,
                threshold_factor=config.network_burst_threshold,
                min_duration=config.min_network_burst_duration,
                min_electrodes_fraction=config.min_electrodes_active
            )
            
            # Calculate network burst statistics
            nb_stats = _calculate_network_burst_statistics(
                network_bursts, 
                spike_list.recording_length,
                len(channels)
            )
            results.update(nb_stats)
            
        except Exception as e:
            warnings.warn(f"Network burst detection failed: {e}")
            results.update(_get_empty_network_burst_metrics())
    else:
        results.update(_get_empty_network_burst_metrics())
    
    return results


def detect_network_bursts(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    threshold_factor: float = 1.25,
    min_duration: float = 0.05,
    min_electrodes_fraction: float = 0.35,
    envelope_bin_size: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Detect network bursts using the envelope algorithm.
    
    This implements a version of the envelope algorithm commonly used
    for MEA network burst detection.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int, optional
        Channels to analyze
    threshold_factor : float
        Threshold factor for envelope detection (default: 1.25)
    min_duration : float
        Minimum network burst duration (seconds)
    min_electrodes_fraction : float
        Minimum fraction of electrodes that must be active
    envelope_bin_size : float
        Bin size for envelope calculation (seconds)
        
    Returns
    -------
    list of dict
        List of detected network bursts with properties
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=5)
    
    if len(channels) < 2:
        return []
    
    # Create population spike density function
    spike_matrix, time_bins = spike_list.bin_spikes(envelope_bin_size, channels)
    
    # Calculate envelope (population activity)
    envelope = np.sum(spike_matrix, axis=0)  # Total spikes per bin
    
    # Smooth envelope (simple moving average)
    window_size = max(1, int(0.05 / envelope_bin_size))  # 50ms window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        envelope_smooth = np.convolve(envelope, kernel, mode='same')
    else:
        envelope_smooth = envelope.copy()
    
    # Calculate threshold
    mean_activity = np.mean(envelope_smooth)
    std_activity = np.std(envelope_smooth)
    threshold = mean_activity + threshold_factor * std_activity
    
    # Find periods above threshold
    above_threshold = envelope_smooth > threshold
    
    # Find burst boundaries
    network_bursts = []
    in_burst = False
    burst_start_idx = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_burst:
            # Start of network burst
            in_burst = True
            burst_start_idx = i
        elif not is_above and in_burst:
            # End of network burst
            burst_end_idx = i - 1
            
            # Check if burst meets criteria
            burst_start_time = time_bins[burst_start_idx]
            burst_end_time = time_bins[burst_end_idx]
            burst_duration = burst_end_time - burst_start_time + envelope_bin_size
            
            if burst_duration >= min_duration:
                # Check electrode participation
                burst_matrix = spike_matrix[:, burst_start_idx:burst_end_idx+1]
                participating_electrodes = np.sum(np.sum(burst_matrix, axis=1) > 0)
                participation_fraction = participating_electrodes / len(channels)
                
                if participation_fraction >= min_electrodes_fraction:
                    # Calculate burst properties
                    total_spikes = np.sum(burst_matrix)
                    peak_activity = np.max(envelope[burst_start_idx:burst_end_idx+1])
                    
                    network_bursts.append({
                        'start_time': burst_start_time,
                        'end_time': burst_end_time,
                        'duration': burst_duration,
                        'total_spikes': total_spikes,
                        'participating_electrodes': participating_electrodes,
                        'participation_fraction': participation_fraction,
                        'peak_activity': peak_activity,
                        'mean_activity': np.mean(envelope[burst_start_idx:burst_end_idx+1]),
                        'start_idx': burst_start_idx,
                        'end_idx': burst_end_idx
                    })
            
            in_burst = False
    
    # Handle case where recording ends during a burst
    if in_burst:
        burst_end_idx = len(envelope) - 1
        burst_start_time = time_bins[burst_start_idx]
        burst_end_time = time_bins[burst_end_idx]
        burst_duration = burst_end_time - burst_start_time + envelope_bin_size
        
        if burst_duration >= min_duration:
            burst_matrix = spike_matrix[:, burst_start_idx:burst_end_idx+1]
            participating_electrodes = np.sum(np.sum(burst_matrix, axis=1) > 0)
            participation_fraction = participating_electrodes / len(channels)
            
            if participation_fraction >= min_electrodes_fraction:
                total_spikes = np.sum(burst_matrix)
                peak_activity = np.max(envelope[burst_start_idx:burst_end_idx+1])
                
                network_bursts.append({
                    'start_time': burst_start_time,
                    'end_time': burst_end_time,
                    'duration': burst_duration,
                    'total_spikes': total_spikes,
                    'participating_electrodes': participating_electrodes,
                    'participation_fraction': participation_fraction,
                    'peak_activity': peak_activity,
                    'mean_activity': np.mean(envelope[burst_start_idx:burst_end_idx+1]),
                    'start_idx': burst_start_idx,
                    'end_idx': burst_end_idx
                })
    
    return network_bursts


def burst_statistics(bursts: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate statistics from a list of bursts.
    
    Parameters
    ----------
    bursts : list of dict
        List of burst dictionaries
        
    Returns
    -------
    dict
        Dictionary with burst statistics
    """
    if len(bursts) == 0:
        return {
            'count': 0,
            'rate': 0.0,
            'duration_mean': np.nan,
            'duration_std': np.nan,
            'duration_median': np.nan,
            'isi_mean': np.nan,
            'isi_std': np.nan
        }
    
    durations = [burst['duration'] for burst in bursts]
    
    # Inter-burst intervals
    if len(bursts) > 1:
        start_times = [burst['start_time'] for burst in bursts]
        isis = np.diff(start_times)
    else:
        isis = []
    
    return {
        'count': len(bursts),
        'duration_mean': np.mean(durations),
        'duration_std': np.std(durations),
        'duration_median': np.median(durations),
        'isi_mean': np.mean(isis) if len(isis) > 0 else np.nan,
        'isi_std': np.std(isis) if len(isis) > 0 else np.nan
    }


def _aggregate_burst_statistics(
    individual_bursts: Dict[int, List[Dict[str, Any]]],
    recording_length: float
) -> Dict[str, float]:
    """Aggregate statistics from individual channel bursts."""
    all_durations = []
    all_isis = []
    channel_burst_counts = []
    bursting_channels = 0
    
    for ch, ch_bursts in individual_bursts.items():
        channel_burst_counts.append(len(ch_bursts))
        
        if len(ch_bursts) > 0:
            bursting_channels += 1
            
            # Collect durations
            durations = [burst['duration'] for burst in ch_bursts]
            all_durations.extend(durations)
            
            # Collect inter-burst intervals
            if len(ch_bursts) > 1:
                start_times = [burst['start_time'] for burst in ch_bursts]
                isis = np.diff(start_times)
                all_isis.extend(isis)
    
    results = {}
    
    # Individual burst statistics
    if len(all_durations) > 0:
        results['burst_duration_mean'] = np.mean(all_durations)
        results['burst_duration_std'] = np.std(all_durations)
        results['burst_duration_median'] = np.median(all_durations)
        results['total_individual_bursts'] = len(all_durations)
    else:
        results['burst_duration_mean'] = np.nan
        results['burst_duration_std'] = np.nan
        results['burst_duration_median'] = np.nan
        results['total_individual_bursts'] = 0
    
    # Inter-burst interval statistics  
    if len(all_isis) > 0:
        results['inter_burst_interval_mean'] = np.mean(all_isis)
        results['inter_burst_interval_std'] = np.std(all_isis)
    else:
        results['inter_burst_interval_mean'] = np.nan
        results['inter_burst_interval_std'] = np.nan
    
    # Channel-level statistics
    results['bursting_channels'] = bursting_channels
    results['total_channels_analyzed'] = len(individual_bursts)
    results['bursting_fraction'] = bursting_channels / len(individual_bursts) if len(individual_bursts) > 0 else 0
    
    if len(channel_burst_counts) > 0:
        results['bursts_per_channel_mean'] = np.mean(channel_burst_counts)
        results['bursts_per_channel_std'] = np.std(channel_burst_counts)
        
        # Burst rate per channel (bursts per minute)
        recording_minutes = recording_length / 60.0
        burst_rates = [count / recording_minutes for count in channel_burst_counts if count > 0]
        
        if len(burst_rates) > 0:
            results['burst_rate_mean'] = np.mean(burst_rates)
            results['burst_rate_std'] = np.std(burst_rates)
        else:
            results['burst_rate_mean'] = 0.0
            results['burst_rate_std'] = np.nan
    else:
        results['bursts_per_channel_mean'] = 0.0
        results['bursts_per_channel_std'] = np.nan
        results['burst_rate_mean'] = 0.0
        results['burst_rate_std'] = np.nan
    
    return results


def _calculate_network_burst_statistics(
    network_bursts: List[Dict[str, Any]],
    recording_length: float,
    n_channels: int
) -> Dict[str, float]:
    """Calculate statistics for network bursts."""
    results = {}
    
    if len(network_bursts) == 0:
        results.update(_get_empty_network_burst_metrics())
        return results
    
    # Basic counts and rates
    results['network_burst_count'] = len(network_bursts)
    recording_minutes = recording_length / 60.0
    results['network_burst_rate'] = len(network_bursts) / recording_minutes
    
    # Duration statistics
    durations = [nb['duration'] for nb in network_bursts]
    results['network_burst_duration_mean'] = np.mean(durations)
    results['network_burst_duration_std'] = np.std(durations)
    results['network_burst_duration_median'] = np.median(durations)
    
    # Participation statistics
    participations = [nb['participation_fraction'] for nb in network_bursts]
    results['burst_participation_ratio'] = np.mean(participations)
    results['burst_participation_std'] = np.std(participations)
    
    # Spike count statistics
    spike_counts = [nb['total_spikes'] for nb in network_bursts]
    results['network_burst_spikes_mean'] = np.mean(spike_counts)
    results['network_burst_spikes_std'] = np.std(spike_counts)
    
    # Activity level statistics
    peak_activities = [nb['peak_activity'] for nb in network_bursts]
    results['network_burst_peak_activity_mean'] = np.mean(peak_activities)
    results['network_burst_peak_activity_std'] = np.std(peak_activities)
    
    # Inter-network-burst intervals
    if len(network_bursts) > 1:
        start_times = [nb['start_time'] for nb in network_bursts]
        inter_nb_intervals = np.diff(start_times)
        results['inter_network_burst_interval_mean'] = np.mean(inter_nb_intervals)
        results['inter_network_burst_interval_std'] = np.std(inter_nb_intervals)
    else:
        results['inter_network_burst_interval_mean'] = np.nan
        results['inter_network_burst_interval_std'] = np.nan
    
    return results


def _get_empty_burst_metrics() -> Dict[str, float]:
    """Return empty burst metrics."""
    return {
        **_get_empty_individual_burst_metrics(),
        **_get_empty_network_burst_metrics()
    }


def _get_empty_individual_burst_metrics() -> Dict[str, float]:
    """Return empty individual burst metrics."""
    return {
        'burst_duration_mean': np.nan,
        'burst_duration_std': np.nan,
        'burst_duration_median': np.nan,
        'total_individual_bursts': 0,
        'inter_burst_interval_mean': np.nan,
        'inter_burst_interval_std': np.nan,
        'bursting_channels': 0,
        'total_channels_analyzed': 0,
        'bursting_fraction': 0.0,
        'bursts_per_channel_mean': 0.0,
        'bursts_per_channel_std': np.nan,
        'burst_rate_mean': 0.0,
        'burst_rate_std': np.nan
    }


def _get_empty_network_burst_metrics() -> Dict[str, float]:
    """Return empty network burst metrics."""
    return {
        'network_burst_count': 0,
        'network_burst_rate': 0.0,
        'network_burst_duration_mean': np.nan,
        'network_burst_duration_std': np.nan,
        'network_burst_duration_median': np.nan,
        'burst_participation_ratio': np.nan,
        'burst_participation_std': np.nan,
        'network_burst_spikes_mean': np.nan,
        'network_burst_spikes_std': np.nan,
        'network_burst_peak_activity_mean': np.nan,
        'network_burst_peak_activity_std': np.nan,
        'inter_network_burst_interval_mean': np.nan,
        'inter_network_burst_interval_std': np.nan
    }