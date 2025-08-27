"""
Data preprocessing functions for MEA data.

This module provides functions for filtering, selecting time windows,
and preparing data for analysis.
"""

import numpy as np
from typing import List, Optional, Tuple
from .spike_list import SpikeList


def preprocess_spikes(
    spike_list: SpikeList,
    min_spikes: int = 10,
    max_firing_rate: Optional[float] = None,
    time_window: Optional[Tuple[float, float]] = None,
    remove_inactive: bool = True
) -> SpikeList:
    """
    Preprocess spike data with various filtering options.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    min_spikes : int
        Minimum number of spikes per channel to keep
    max_firing_rate : float, optional
        Maximum firing rate (Hz) to filter out noisy channels
    time_window : tuple of float, optional
        Time window (start, end) in seconds to extract
    remove_inactive : bool
        Whether to remove channels with no spikes
        
    Returns
    -------
    SpikeList
        Preprocessed spike data
    """
    # Apply time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        spike_list = spike_list.get_time_window(start_time, end_time)
    
    # Filter channels based on activity
    active_channels = spike_list.get_active_channels(min_spikes=1 if not remove_inactive else min_spikes)
    
    # Additional filtering based on firing rate
    if max_firing_rate is not None:
        filtered_channels = []
        for ch in active_channels:
            firing_rate = spike_list.spike_trains[ch].firing_rate
            if firing_rate <= max_firing_rate:
                filtered_channels.append(ch)
        active_channels = filtered_channels
    
    # Create filtered spike data
    filtered_data = {}
    for ch in active_channels:
        if spike_list.spike_trains[ch].n_spikes >= min_spikes:
            filtered_data[ch] = spike_list.spike_trains[ch].spike_times
    
    return SpikeList(
        spike_data=filtered_data,
        channel_ids=list(filtered_data.keys()),
        recording_length=spike_list.recording_length,
        well_map=spike_list.well_map,
        sampling_rate=spike_list.sampling_rate
    )


def filter_channels(
    spike_list: SpikeList,
    channels: List[int]
) -> SpikeList:
    """
    Filter spike list to include only specified channels.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    channels : list of int
        Channels to keep
        
    Returns
    -------
    SpikeList
        Filtered spike data
    """
    filtered_data = {}
    for ch in channels:
        if ch in spike_list.spike_trains:
            filtered_data[ch] = spike_list.spike_trains[ch].spike_times
    
    return SpikeList(
        spike_data=filtered_data,
        channel_ids=channels,
        recording_length=spike_list.recording_length,
        well_map=spike_list.well_map,
        sampling_rate=spike_list.sampling_rate
    )


def time_window_selection(
    spike_list: SpikeList,
    window_length: float,
    overlap: float = 0.0,
    min_spikes_per_window: int = 1
) -> List[SpikeList]:
    """
    Split spike data into multiple time windows.
    
    Parameters
    ----------
    spike_list : SpikeList
        Input spike data
    window_length : float
        Length of each window in seconds
    overlap : float
        Overlap between windows (0.0 to 1.0)
    min_spikes_per_window : int
        Minimum spikes required per window to keep it
        
    Returns
    -------
    list of SpikeList
        List of windowed spike data
    """
    windows = []
    step = window_length * (1.0 - overlap)
    
    start_time = 0.0
    while start_time + window_length <= spike_list.recording_length:
        end_time = start_time + window_length
        
        window_data = spike_list.get_time_window(start_time, end_time)
        
        # Check if window has sufficient activity
        total_spikes = sum(train.n_spikes for train in window_data.spike_trains.values())
        if total_spikes >= min_spikes_per_window:
            windows.append(window_data)
        
        start_time += step
    
    return windows