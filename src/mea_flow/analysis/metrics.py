"""
Main metrics calculation class for MEA data analysis.

This module provides the MEAMetrics class that orchestrates the calculation
of all metrics (activity, regularity, synchrony) for MEA recordings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import warnings
from dataclasses import dataclass

from ..data import SpikeList
from .activity import compute_activity_metrics
from .regularity import compute_regularity_metrics 
from .synchrony import compute_synchrony_metrics
from .burst_analysis import network_burst_analysis


@dataclass
class AnalysisConfig:
    """Configuration parameters for MEA analysis."""
    
    # Time binning
    time_bin_size: float = 1.0  # seconds
    
    # Activity metrics
    min_spikes_for_rate: int = 10
    
    # Regularity metrics  
    min_isi_samples: int = 5
    
    # Synchrony metrics
    n_pairs_sync: int = 500
    sync_time_bin: float = 0.01  # seconds for correlation
    tau_van_rossum: float = 0.02  # seconds
    
    # Burst detection
    burst_detection: bool = True
    min_burst_spikes: int = 5
    max_isi_burst: float = 0.1  # seconds
    
    # Network burst detection
    network_burst_detection: bool = True
    network_burst_threshold: float = 1.25
    min_network_burst_duration: float = 0.05  # seconds
    min_electrodes_active: float = 0.35  # fraction


class MEAMetrics:
    """
    Main class for calculating comprehensive MEA metrics.
    
    This class orchestrates the calculation of activity, regularity, and 
    synchrony metrics for MEA recordings, with support for different
    grouping strategies (by condition, well, time window).
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize MEAMetrics calculator.
        
        Parameters
        ----------
        config : AnalysisConfig, optional
            Configuration for analysis parameters
        """
        self.config = config if config is not None else AnalysisConfig()
        
    def compute_all_metrics(
        self,
        spike_list: SpikeList,
        grouping: str = 'global',
        group_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute all metrics for a SpikeList.
        
        Parameters
        ----------
        spike_list : SpikeList
            Input spike data
        grouping : str
            Grouping strategy: 'global', 'well', 'time', or 'channel'
        group_params : dict, optional
            Parameters for grouping (e.g., time_window_length for 'time')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all computed metrics
        """
        if grouping == 'global':
            return self._compute_global_metrics(spike_list)
        elif grouping == 'well':
            return self._compute_well_metrics(spike_list)
        elif grouping == 'time':
            window_length = group_params.get('window_length', 5.0) if group_params else 5.0
            return self._compute_time_metrics(spike_list, window_length)
        elif grouping == 'channel':
            return self._compute_channel_metrics(spike_list)
        else:
            raise ValueError(f"Unknown grouping strategy: {grouping}")
    
    def _compute_global_metrics(self, spike_list: SpikeList) -> pd.DataFrame:
        """Compute metrics for entire recording."""
        results = {}
        
        # Activity metrics
        activity_metrics = compute_activity_metrics(spike_list, self.config)
        results.update(activity_metrics)
        
        # Regularity metrics
        regularity_metrics = compute_regularity_metrics(spike_list, self.config)
        results.update(regularity_metrics)
        
        # Synchrony metrics
        synchrony_metrics = compute_synchrony_metrics(spike_list, self.config)
        results.update(synchrony_metrics)
        
        # Burst analysis
        if self.config.burst_detection or self.config.network_burst_detection:
            burst_metrics = network_burst_analysis(spike_list, self.config)
            results.update(burst_metrics)
        
        # Create DataFrame
        df = pd.DataFrame([results])
        df['group_type'] = 'global'
        df['group_id'] = 'all'
        df['n_channels'] = len(spike_list.get_active_channels())
        df['recording_length'] = spike_list.recording_length
        
        return df
    
    def _compute_well_metrics(self, spike_list: SpikeList) -> pd.DataFrame:
        """Compute metrics per well."""
        results_list = []
        
        for well_id, well_channels in spike_list.well_map.items():
            # Get active channels in this well
            active_channels = [
                ch for ch in well_channels 
                if ch in spike_list.channel_ids and 
                spike_list.spike_trains[ch].n_spikes >= self.config.min_spikes_for_rate
            ]
            
            if len(active_channels) == 0:
                continue
                
            # Create well-specific SpikeList
            well_data = {}
            for ch in active_channels:
                well_data[ch] = spike_list.spike_trains[ch].spike_times
                
            well_spike_list = SpikeList(
                spike_data=well_data,
                channel_ids=active_channels,
                recording_length=spike_list.recording_length,
                well_map={well_id: np.array(active_channels)},
                sampling_rate=spike_list.sampling_rate
            )
            
            # Compute metrics for this well
            well_results = {}
            
            # Activity metrics
            activity_metrics = compute_activity_metrics(well_spike_list, self.config)
            well_results.update(activity_metrics)
            
            # Regularity metrics
            regularity_metrics = compute_regularity_metrics(well_spike_list, self.config)
            well_results.update(regularity_metrics)
            
            # Synchrony metrics
            synchrony_metrics = compute_synchrony_metrics(well_spike_list, self.config)
            well_results.update(synchrony_metrics)
            
            # Burst analysis
            if self.config.burst_detection or self.config.network_burst_detection:
                burst_metrics = network_burst_analysis(well_spike_list, self.config)
                well_results.update(burst_metrics)
            
            # Add metadata
            well_results['group_type'] = 'well'
            well_results['group_id'] = well_id
            well_results['n_channels'] = len(active_channels)
            well_results['recording_length'] = spike_list.recording_length
            
            results_list.append(well_results)
        
        return pd.DataFrame(results_list)
    
    def _compute_time_metrics(
        self, 
        spike_list: SpikeList, 
        window_length: float
    ) -> pd.DataFrame:
        """Compute metrics per time window."""
        results_list = []
        
        # Split into time windows
        from ..data.preprocessing import time_window_selection
        windows = time_window_selection(
            spike_list, 
            window_length=window_length,
            overlap=0.0,
            min_spikes_per_window=10
        )
        
        for i, window_data in enumerate(windows):
            window_results = {}
            
            # Skip windows with too few active channels
            active_channels = window_data.get_active_channels(min_spikes=1)
            if len(active_channels) < 2:
                continue
            
            # Activity metrics
            activity_metrics = compute_activity_metrics(window_data, self.config)
            window_results.update(activity_metrics)
            
            # Regularity metrics
            regularity_metrics = compute_regularity_metrics(window_data, self.config)
            window_results.update(regularity_metrics)
            
            # Synchrony metrics (may be limited for short windows)
            try:
                synchrony_metrics = compute_synchrony_metrics(window_data, self.config)
                window_results.update(synchrony_metrics)
            except Exception as e:
                warnings.warn(f"Synchrony calculation failed for window {i}: {e}")
                # Add NaN values for synchrony metrics
                sync_keys = ['pearson_cc_mean', 'pearson_cc_std', 'isi_distance', 
                           'spike_distance', 'spike_sync_distance']
                for key in sync_keys:
                    window_results[key] = np.nan
            
            # Add metadata
            window_results['group_type'] = 'time'
            window_results['group_id'] = i
            window_results['window_start'] = i * window_length
            window_results['window_end'] = (i + 1) * window_length
            window_results['n_channels'] = len(active_channels)
            window_results['recording_length'] = window_length
            
            results_list.append(window_results)
            
        return pd.DataFrame(results_list)
    
    def _compute_channel_metrics(self, spike_list: SpikeList) -> pd.DataFrame:
        """Compute metrics per channel."""
        results_list = []
        
        active_channels = spike_list.get_active_channels(
            min_spikes=self.config.min_spikes_for_rate
        )
        
        for ch in active_channels:
            # Create single-channel SpikeList
            channel_data = {ch: spike_list.spike_trains[ch].spike_times}
            
            channel_spike_list = SpikeList(
                spike_data=channel_data,
                channel_ids=[ch],
                recording_length=spike_list.recording_length,
                sampling_rate=spike_list.sampling_rate
            )
            
            channel_results = {}
            
            # Activity metrics (individual channel)
            activity_metrics = compute_activity_metrics(channel_spike_list, self.config)
            channel_results.update(activity_metrics)
            
            # Regularity metrics (individual channel)
            regularity_metrics = compute_regularity_metrics(channel_spike_list, self.config)
            channel_results.update(regularity_metrics)
            
            # Note: Synchrony metrics don't make sense for single channels
            # Add NaN values
            sync_keys = ['pearson_cc_mean', 'pearson_cc_std', 'isi_distance',
                        'spike_distance', 'spike_sync_distance']
            for key in sync_keys:
                channel_results[key] = np.nan
            
            # Add metadata
            channel_results['group_type'] = 'channel'
            channel_results['group_id'] = ch
            channel_results['channel_id'] = ch
            channel_results['n_channels'] = 1
            channel_results['recording_length'] = spike_list.recording_length
            
            # Determine well assignment
            well_assignment = None
            for well_id, well_channels in spike_list.well_map.items():
                if ch in well_channels:
                    well_assignment = well_id
                    break
            channel_results['well_id'] = well_assignment
            
            results_list.append(channel_results)
            
        return pd.DataFrame(results_list)
    
    def compare_conditions(
        self,
        spike_lists: Dict[str, SpikeList],
        grouping: str = 'global',
        group_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple experimental conditions.
        
        Parameters
        ---------- 
        spike_lists : dict
            Dictionary mapping condition names to SpikeList objects
        grouping : str
            Grouping strategy for analysis
        group_params : dict, optional
            Parameters for grouping
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with metrics for all conditions
        """
        all_results = []
        
        for condition_name, spike_list in spike_lists.items():
            condition_results = self.compute_all_metrics(
                spike_list, 
                grouping=grouping,
                group_params=group_params
            )
            condition_results['condition'] = condition_name
            all_results.append(condition_results)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns to put condition first
        cols = ['condition'] + [col for col in combined_df.columns if col != 'condition']
        combined_df = combined_df[cols]
        
        return combined_df
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all computed metrics."""
        return {
            # Activity metrics
            'mean_firing_rate': 'Mean firing rate across all active channels (Hz)',
            'total_spike_count': 'Total number of spikes across all channels',
            'active_channels_count': 'Number of channels with activity',
            'network_firing_rate': 'Population firing rate (spikes/s)',
            
            # Regularity metrics
            'cv_isi_mean': 'Mean coefficient of variation of inter-spike intervals',
            'cv_isi_std': 'Standard deviation of CV-ISI across channels',
            'lv_isi_mean': 'Mean local variation of inter-spike intervals',
            'entropy_isi_mean': 'Mean entropy of ISI distributions',
            'fano_factor_mean': 'Mean Fano factor (spike count variance/mean)',
            
            # Synchrony metrics
            'pearson_cc_mean': 'Mean pairwise Pearson correlation coefficient',
            'pearson_cc_std': 'Standard deviation of pairwise correlations',
            'isi_distance': 'ISI distance measure (PySpike)',
            'spike_distance': 'SPIKE distance measure (PySpike)',
            'spike_sync_distance': 'SPIKE synchrony distance (PySpike)',
            
            # Burst metrics
            'burst_rate_mean': 'Mean burst rate across channels (bursts/min)',
            'network_burst_rate': 'Network burst rate (network bursts/min)', 
            'network_burst_duration_mean': 'Mean network burst duration (s)',
            'burst_participation_ratio': 'Fraction of channels participating in network bursts'
        }