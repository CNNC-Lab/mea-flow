"""
SpikeList class for handling MEA spiking data with channel mapping.

This module provides the core SpikeList class that manages spike trains from
multiple electrodes/channels in MEA recordings.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class SpikeTrain:
    """Individual spike train for a single electrode/channel."""
    
    channel_id: int
    spike_times: np.ndarray
    recording_length: float
    
    def __post_init__(self):
        """Validate and sort spike times."""
        self.spike_times = np.asarray(self.spike_times)
        # Sort but preserve duplicates (MATLAB doesn't remove duplicate timestamps)
        self.spike_times = np.sort(self.spike_times)
        
        if len(self.spike_times) > 0:
            if self.spike_times[-1] > self.recording_length:
                warnings.warn(
                    f"Spike times exceed recording length for channel {self.channel_id}"
                )
    
    @property 
    def n_spikes(self) -> int:
        """Number of spikes in this train."""
        return len(self.spike_times)
    
    @property
    def firing_rate(self) -> float:
        """Mean firing rate in Hz."""
        if self.recording_length > 0:
            return self.n_spikes / self.recording_length
        return 0.0
    
    def get_isi(self) -> np.ndarray:
        """Get inter-spike intervals."""
        if len(self.spike_times) < 2:
            return np.array([])
        return np.diff(self.spike_times)


class SpikeList:
    """
    Container for managing multiple spike trains from MEA recordings.
    
    This class handles spike data from multiple electrodes/channels and provides
    methods for analysis, visualization, and data manipulation specific to MEA
    recordings with well-based organization.
    """
    
    def __init__(
        self,
        spike_data: Union[List[Tuple[int, Union[float, np.ndarray]]], Dict[int, np.ndarray]],
        channel_ids: Optional[List[int]] = None,
        recording_length: float = None,
        well_map: Optional[Dict[int, np.ndarray]] = None,
        sampling_rate: float = 12500.0  # Default Axion sampling rate
    ):
        """
        Initialize SpikeList from various input formats.
        
        Parameters
        ----------
        spike_data : list of tuples or dict
            Spike data in format [(channel_id, spike_time), ...] or
            {channel_id: spike_times_array}
        channel_ids : list of int, optional
            List of all channel IDs (for completeness)
        recording_length : float, optional
            Total recording length in seconds
        well_map : dict, optional
            Mapping from well number to channel arrays
            e.g., {1: np.arange(0, 16), 2: np.arange(16, 32), ...}
        sampling_rate : float
            Sampling rate in Hz (default: 12500 for Axion)
        """
        self.sampling_rate = sampling_rate
        self.spike_trains = {}
        
        # Process input data
        if isinstance(spike_data, list):
            self._init_from_spike_list(spike_data)
        elif isinstance(spike_data, dict):
            self._init_from_dict(spike_data)
        else:
            raise ValueError("spike_data must be list of tuples or dict")
            
        # Set channel IDs
        if channel_ids is not None:
            self.channel_ids = sorted(channel_ids)
        else:
            self.channel_ids = sorted(self.raw_spike_data.keys())
            
        # Determine recording length
        if recording_length is not None:
            self.recording_length = recording_length
        else:
            self._auto_detect_recording_length()
            
        # Set well mapping
        if well_map is not None:
            self.well_map = well_map
        else:
            self.well_map = self._get_default_well_map()
            
        # Create SpikeTrain objects
        self._create_spike_trains()
        
    def _init_from_spike_list(self, spike_data: List[Tuple[int, float]]):
        """Initialize from list of (channel_id, spike_time) tuples."""
        spike_dict = {}
        for channel_id, spike_time in spike_data:
            if channel_id not in spike_dict:
                spike_dict[channel_id] = []
            spike_dict[channel_id].append(spike_time)
        
        # Convert to arrays
        for channel_id in spike_dict:
            spike_dict[channel_id] = np.array(spike_dict[channel_id])
            
        self.raw_spike_data = spike_dict
        
    def _init_from_dict(self, spike_data: Dict):
        """Initialize from dictionary of spike data."""
        if 'times' in spike_data and 'channels' in spike_data:
            # Handle format: {'times': [...], 'channels': [...]}
            times = np.asarray(spike_data['times'])
            channels = np.asarray(spike_data['channels'])
            
            # Convert to channel-indexed format
            spike_dict = {}
            for time, channel in zip(times, channels):
                if channel not in spike_dict:
                    spike_dict[channel] = []
                spike_dict[channel].append(time)
            
            # Convert lists to arrays
            for channel_id in spike_dict:
                spike_dict[channel_id] = np.array(spike_dict[channel_id])
                
            self.raw_spike_data = spike_dict
        else:
            # Handle format: {channel_id: spike_times}
            self.raw_spike_data = spike_data.copy()
        
    def _auto_detect_recording_length(self):
        """Auto-detect recording length from maximum spike time."""
        max_time = 0.0
        for spike_times in self.raw_spike_data.values():
            if len(spike_times) > 0:
                max_time = max(max_time, np.max(spike_times))
        
        # Add 10% buffer
        self.recording_length = max_time * 1.1
        
    def _get_default_well_map(self) -> Dict[int, np.ndarray]:
        """Get default well mapping for standard MEA plates."""
        # Standard 4-well mapping (16 channels per well)
        n_channels = len(self.channel_ids)
        
        if n_channels <= 16:
            return {1: np.array(self.channel_ids)}
        elif n_channels <= 64:
            # 4 wells with 16 channels each
            return {
                1: np.arange(0, 16),
                2: np.arange(16, 32), 
                3: np.arange(32, 48),
                4: np.arange(48, 64)
            }
        else:
            # Single well with all channels
            return {1: np.array(self.channel_ids)}
            
    def _create_spike_trains(self):
        """Create SpikeTrain objects for each channel."""
        for channel_id in self.channel_ids:
            spike_times = self.raw_spike_data.get(channel_id, np.array([]))
            self.spike_trains[channel_id] = SpikeTrain(
                channel_id=channel_id,
                spike_times=spike_times,
                recording_length=self.recording_length
            )
    
    def get_well_channels(self, well_id: int) -> List[int]:
        """Get channel IDs for a specific well."""
        if well_id in self.well_map:
            return [ch for ch in self.well_map[well_id] if ch in self.channel_ids]
        return []
    
    def get_well_spike_trains(self, well_id: int) -> Dict[int, SpikeTrain]:
        """Get spike trains for channels in a specific well."""
        well_channels = self.get_well_channels(well_id)
        return {ch: self.spike_trains[ch] for ch in well_channels}
    
    def get_active_channels(self, min_spikes: int = 1) -> List[int]:
        """Get channels with minimum number of spikes."""
        return [
            ch for ch, train in self.spike_trains.items() 
            if train.n_spikes >= min_spikes
        ]
    
    def get_time_window(
        self, 
        start_time: float, 
        end_time: float, 
        channels: Optional[List[int]] = None
    ) -> 'SpikeList':
        """
        Extract spikes from a specific time window.
        
        Parameters
        ----------
        start_time : float
            Start time in seconds
        end_time : float  
            End time in seconds
        channels : list of int, optional
            Channels to include (default: all)
            
        Returns
        -------
        SpikeList
            New SpikeList with windowed data
        """
        if channels is None:
            channels = self.channel_ids
            
        windowed_data = {}
        for ch in channels:
            if ch in self.spike_trains:
                spike_times = self.spike_trains[ch].spike_times
                mask = (spike_times >= start_time) & (spike_times <= end_time)
                # Adjust times to be relative to start_time
                windowed_data[ch] = spike_times[mask] - start_time
                
        return SpikeList(
            spike_data=windowed_data,
            channel_ids=channels,
            recording_length=end_time - start_time,
            well_map=self.well_map,
            sampling_rate=self.sampling_rate
        )
    
    def bin_spikes(
        self, 
        bin_size: float, 
        channels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin spike trains into time bins.
        
        Parameters
        ----------
        bin_size : float
            Bin size in seconds
        channels : list of int, optional
            Channels to include (default: all active)
            
        Returns
        -------
        spike_matrix : np.ndarray
            Matrix of spike counts (channels x time_bins)
        time_bins : np.ndarray  
            Time bin centers
        """
        if channels is None:
            channels = self.get_active_channels()
            
        n_bins = int(np.ceil(self.recording_length / bin_size))
        time_bins = np.arange(n_bins) * bin_size + bin_size / 2
        
        spike_matrix = np.zeros((len(channels), n_bins))
        
        for i, ch in enumerate(channels):
            if ch in self.spike_trains:
                spike_times = self.spike_trains[ch].spike_times
                counts, _ = np.histogram(spike_times, bins=n_bins, 
                                       range=(0, self.recording_length))
                spike_matrix[i, :] = counts
                
        return spike_matrix, time_bins
    
    def to_continuous_signal(
        self, 
        tau: float = 0.02,
        channels: Optional[List[int]] = None,
        dt: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert spike trains to continuous signals using exponential filter.
        
        Parameters
        ----------
        tau : float
            Exponential filter time constant in seconds
        channels : list of int, optional
            Channels to include
        dt : float
            Sampling interval for continuous signal
            
        Returns
        -------
        signal_matrix : np.ndarray
            Continuous signal matrix (channels x time_points)
        time_vector : np.ndarray
            Time vector for signal
        """
        if channels is None:
            channels = self.get_active_channels()
            
        # Create time vector
        n_points = int(self.recording_length / dt) + 1
        time_vector = np.arange(n_points) * dt
        
        signal_matrix = np.zeros((len(channels), n_points))
        
        for i, ch in enumerate(channels):
            if ch in self.spike_trains and self.spike_trains[ch].n_spikes > 0:
                spike_times = self.spike_trains[ch].spike_times
                
                # Create delta function at spike times
                spike_indices = np.round(spike_times / dt).astype(int)
                spike_indices = spike_indices[spike_indices < n_points]
                
                delta_signal = np.zeros(n_points)
                delta_signal[spike_indices] = 1.0
                
                # Apply exponential filter
                # Convolve with exponential kernel
                kernel_length = int(5 * tau / dt)  # 5 time constants
                kernel_times = np.arange(kernel_length) * dt
                kernel = np.exp(-kernel_times / tau) / tau
                
                signal_matrix[i, :] = np.convolve(delta_signal, kernel, mode='same')
                
        return signal_matrix, time_vector
    
    def summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all channels."""
        stats = []
        
        for ch in self.channel_ids:
            train = self.spike_trains[ch]
            
            # Determine well assignment
            well_assignment = None
            for well_id, well_channels in self.well_map.items():
                if ch in well_channels:
                    well_assignment = well_id
                    break
                    
            stats.append({
                'channel_id': ch,
                'well_id': well_assignment,
                'n_spikes': train.n_spikes,
                'firing_rate': train.firing_rate,
                'first_spike': train.spike_times[0] if train.n_spikes > 0 else np.nan,
                'last_spike': train.spike_times[-1] if train.n_spikes > 0 else np.nan,
                'mean_isi': np.mean(train.get_isi()) if train.n_spikes > 1 else np.nan,
                'cv_isi': np.std(train.get_isi()) / np.mean(train.get_isi()) 
                         if train.n_spikes > 1 else np.nan
            })
            
        return pd.DataFrame(stats)
    
    def get_all_spike_times(self) -> List[float]:
        """Get all spike times across all channels."""
        all_times = []
        for train in self.spike_trains.values():
            all_times.extend(train.spike_times)
        return sorted(all_times)
    
    def __repr__(self) -> str:
        """String representation of SpikeList."""
        n_active = len(self.get_active_channels())
        total_spikes = sum(train.n_spikes for train in self.spike_trains.values())
        return (
            f"SpikeList(channels={len(self.channel_ids)}, "
            f"active={n_active}, spikes={total_spikes}, "
            f"duration={self.recording_length:.1f}s)"
        )