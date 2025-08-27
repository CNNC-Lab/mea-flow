"""
Well-based visualizations for MEA data.

This module provides functions for visualizing MEA data organized by wells,
including electrode maps and spatial activity patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings

from ..data import SpikeList


def plot_electrode_map(
    spike_list: SpikeList,
    metric: str = 'firing_rate',
    well_id: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create electrode map showing spatial distribution of activity metrics.
    
    This function creates a 4x4 grid visualization of electrode activity
    for a specific well, with color-coding representing the chosen metric.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to analyze
    metric : str
        Metric to display ('firing_rate', 'spike_count', 'cv_isi', etc.)
    well_id : int, optional
        Specific well to plot (default: first well with data)
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Select well to plot
    if well_id is None:
        # Use first well with active channels
        well_id = None
        for wid, channels in spike_list.well_map.items():
            active_in_well = [ch for ch in channels if ch in spike_list.get_active_channels()]
            if len(active_in_well) > 0:
                well_id = wid
                break
        
        if well_id is None:
            warnings.warn("No wells with active channels found")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No active wells', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    
    if well_id not in spike_list.well_map:
        raise ValueError(f"Well {well_id} not found in well map")
    
    well_channels = spike_list.well_map[well_id]
    
    # Calculate metric values for each channel
    metric_values = {}
    
    for ch in well_channels:
        if ch in spike_list.spike_trains:
            train = spike_list.spike_trains[ch]
            
            if metric == 'firing_rate':
                metric_values[ch] = train.firing_rate
            elif metric == 'spike_count':
                metric_values[ch] = train.n_spikes
            elif metric == 'cv_isi':
                isis = train.get_isi()
                if len(isis) > 1:
                    metric_values[ch] = np.std(isis) / np.mean(isis)
                else:
                    metric_values[ch] = np.nan
            else:
                # Try to get from summary statistics
                summary = spike_list.summary_statistics()
                channel_summary = summary[summary['channel_id'] == ch]
                if len(channel_summary) > 0 and metric in channel_summary.columns:
                    metric_values[ch] = channel_summary[metric].iloc[0]
                else:
                    metric_values[ch] = np.nan
        else:
            metric_values[ch] = 0.0  # Inactive channel
    
    # Create 4x4 grid (standard MEA layout)
    grid_data = np.full((4, 4), np.nan)
    
    # Map channels to grid positions
    # Assuming standard channel numbering (0-15 for 4x4 grid)
    for i, ch in enumerate(well_channels[:16]):  # Only take first 16 channels
        row = i // 4
        col = i % 4
        grid_data[row, col] = metric_values.get(ch, np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(_format_metric_name(metric))
    
    # Add channel labels and values
    for i in range(4):
        for j in range(4):
            channel_idx = i * 4 + j
            if channel_idx < len(well_channels):
                ch = well_channels[channel_idx]
                value = metric_values.get(ch, np.nan)
                
                # Channel ID
                ax.text(j, i - 0.3, f'Ch {ch}', ha='center', va='center', 
                       fontsize=9, color='white', weight='bold')
                
                # Metric value
                if not np.isnan(value):
                    ax.text(j, i + 0.2, f'{value:.2f}', ha='center', va='center',
                           fontsize=8, color='white')
                else:
                    ax.text(j, i + 0.2, 'N/A', ha='center', va='center',
                           fontsize=8, color='white')
    
    # Formatting
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add grid lines
    for x in np.arange(-0.5, 4, 1):
        ax.axvline(x, color='white', linewidth=1)
    for y in np.arange(-0.5, 4, 1):
        ax.axhline(y, color='white', linewidth=1)
    
    ax.set_title(f'Well {well_id} Electrode Map - {_format_metric_name(metric)}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_well_grid(
    spike_lists: Dict[str, SpikeList],
    metric: str = 'firing_rate',
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid showing electrode maps for multiple conditions/wells.
    
    Parameters
    ----------
    spike_lists : dict
        Dictionary mapping condition names to SpikeList objects
    metric : str
        Metric to display across all maps
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    n_conditions = len(spike_lists)
    
    if n_conditions == 0:
        warnings.warn("No spike lists provided")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Create subplots
    fig, axes = plt.subplots(1, n_conditions, figsize=figsize)
    if n_conditions == 1:
        axes = [axes]
    
    # Collect all metric values to set consistent color scale
    all_values = []
    for spike_list in spike_lists.values():
        for ch in spike_list.get_active_channels():
            train = spike_list.spike_trains[ch]
            if metric == 'firing_rate':
                all_values.append(train.firing_rate)
            elif metric == 'spike_count':
                all_values.append(train.n_spikes)
            # Add more metrics as needed
    
    if len(all_values) == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Plot each condition
    for i, (condition, spike_list) in enumerate(spike_lists.items()):
        ax = axes[i]
        
        # Use first well for each condition
        well_id = list(spike_list.well_map.keys())[0]
        well_channels = spike_list.well_map[well_id]
        
        # Calculate metric values
        metric_values = {}
        for ch in well_channels:
            if ch in spike_list.spike_trains:
                train = spike_list.spike_trains[ch]
                
                if metric == 'firing_rate':
                    metric_values[ch] = train.firing_rate
                elif metric == 'spike_count':
                    metric_values[ch] = train.n_spikes
                else:
                    metric_values[ch] = np.nan
            else:
                metric_values[ch] = 0.0
        
        # Create 4x4 grid
        grid_data = np.full((4, 4), np.nan)
        
        for j, ch in enumerate(well_channels[:16]):
            row = j // 4
            col = j % 4
            grid_data[row, col] = metric_values.get(ch, np.nan)
        
        # Plot
        im = ax.imshow(grid_data, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # Add channel labels
        for row in range(4):
            for col in range(4):
                ch_idx = row * 4 + col
                if ch_idx < len(well_channels):
                    ch = well_channels[ch_idx]
                    ax.text(col, row, f'{ch}', ha='center', va='center',
                           fontsize=8, color='white', weight='bold')
        
        ax.set_title(condition)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(_format_metric_name(metric))
    
    fig.suptitle(f'Electrode Maps - {_format_metric_name(metric)}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_well_activity(
    spike_list: SpikeList,
    time_window: float = 1.0,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot activity levels for each well over time.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to analyze
    time_window : float
        Time window for binning activity (seconds)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Calculate time bins
    n_bins = int(np.ceil(spike_list.recording_length / time_window))
    time_bins = np.arange(n_bins) * time_window + time_window / 2
    
    # Calculate activity per well
    well_activities = {}
    well_colors = {}
    
    colors = sns.color_palette("husl", len(spike_list.well_map))
    
    for i, (well_id, well_channels) in enumerate(spike_list.well_map.items()):
        well_colors[well_id] = colors[i]
        
        active_channels = [
            ch for ch in well_channels 
            if ch in spike_list.channel_ids and 
            spike_list.spike_trains[ch].n_spikes > 0
        ]
        
        if len(active_channels) == 0:
            well_activities[well_id] = np.zeros(n_bins)
            continue
        
        # Get spike matrix for this well
        spike_matrix, _ = spike_list.bin_spikes(time_window, active_channels)
        
        # Sum activity across channels
        well_activity = np.sum(spike_matrix, axis=0)
        well_activities[well_id] = well_activity
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for well_id, activity in well_activities.items():
        if np.sum(activity) > 0:  # Only plot wells with activity
            ax.plot(time_bins[:len(activity)], activity, 
                   color=well_colors[well_id], label=f'Well {well_id}', 
                   linewidth=2, alpha=0.8)
            
            # Add filled area
            ax.fill_between(time_bins[:len(activity)], activity, 
                           alpha=0.3, color=well_colors[well_id])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spike Count per Time Window')
    ax.set_title(f'Well Activity Over Time (window = {time_window}s)')
    
    if len(spike_list.well_map) > 1:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _format_metric_name(metric_name: str) -> str:
    """Format metric names for display."""
    name_mapping = {
        'firing_rate': 'Firing Rate (Hz)',
        'spike_count': 'Spike Count',
        'cv_isi': 'CV-ISI',
        'mean_firing_rate': 'Mean Firing Rate (Hz)',
    }
    
    if metric_name in name_mapping:
        return name_mapping[metric_name]
    
    return metric_name.replace('_', ' ').title()