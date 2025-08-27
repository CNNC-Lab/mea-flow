"""
Raster plot visualizations for MEA data.

This module provides functions for creating raster plots and animated
visualizations of spike train data.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional, Tuple, Dict
import warnings

from ..data import SpikeList


def plot_raster(
    spike_list: SpikeList,
    channels: Optional[List[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    color_by_well: bool = True,
    show_well_boundaries: bool = True,
    marker_size: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a raster plot of MEA spike data.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to plot
    channels : list of int, optional
        Channels to include (default: all active channels)
    time_range : tuple of float, optional
        Time range (start_time, end_time) to plot
    color_by_well : bool
        Whether to color spikes by well assignment
    show_well_boundaries : bool
        Whether to show lines separating wells
    marker_size : float
        Size of spike markers
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=1)
    
    if len(channels) == 0:
        warnings.warn("No active channels to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No active channels', 
                transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Apply time range filter if specified
    if time_range is not None:
        start_time, end_time = time_range
        filtered_spike_list = spike_list.get_time_window(start_time, end_time, channels)
        plot_spike_list = filtered_spike_list
        time_offset = start_time
    else:
        plot_spike_list = spike_list
        time_offset = 0.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare colors
    if color_by_well:
        well_colors = _get_well_colors(plot_spike_list.well_map)
        channel_colors = {}
        for well_id, well_channels in plot_spike_list.well_map.items():
            color = well_colors[well_id]
            for ch in well_channels:
                if ch in channels:
                    channel_colors[ch] = color
    else:
        # Single color for all channels
        default_color = '#2E86AB'
        channel_colors = {ch: default_color for ch in channels}
    
    # Plot spikes for each channel
    y_positions = {ch: i for i, ch in enumerate(channels)}
    
    for ch in channels:
        if ch in plot_spike_list.spike_trains:
            spike_times = plot_spike_list.spike_trains[ch].spike_times + time_offset
            y_pos = y_positions[ch]
            
            if len(spike_times) > 0:
                ax.scatter(spike_times, [y_pos] * len(spike_times),
                          c=channel_colors.get(ch, '#2E86AB'),
                          s=marker_size, alpha=0.8, rasterized=True)
    
    # Add well boundaries if requested
    if show_well_boundaries and color_by_well and len(plot_spike_list.well_map) > 1:
        _add_well_boundaries(ax, channels, plot_spike_list.well_map)
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel ID')
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    
    # Set time limits
    if time_range is not None:
        ax.set_xlim(time_range)
    else:
        ax.set_xlim(0, plot_spike_list.recording_length)
    
    ax.set_ylim(-0.5, len(channels) - 0.5)
    
    # Title
    if time_range is not None:
        title = f'Spike Raster Plot ({time_range[0]:.1f}s - {time_range[1]:.1f}s)'
    else:
        title = 'Spike Raster Plot'
    
    ax.set_title(title)
    
    # Add legend for wells if coloring by well
    if color_by_well and len(plot_spike_list.well_map) > 1:
        _add_well_legend(ax, well_colors)
    
    # Add statistics text
    _add_raster_stats(ax, plot_spike_list, channels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_animated_raster(
    spike_list: SpikeList,
    window_length: float = 5.0,
    step_size: float = 1.0,
    channels: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    interval: int = 500,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create an animated raster plot with a sliding time window.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to animate
    window_length : float
        Length of time window to display (seconds)
    step_size : float
        Step size for animation (seconds)
    channels : list of int, optional
        Channels to include
    figsize : tuple
        Figure size
    interval : int
        Animation interval in milliseconds
    save_path : str, optional
        Path to save animation (requires ffmpeg)
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    if channels is None:
        channels = spike_list.get_active_channels(min_spikes=1)
    
    # Calculate time windows
    max_start_time = spike_list.recording_length - window_length
    window_starts = np.arange(0, max_start_time + step_size, step_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare colors
    well_colors = _get_well_colors(spike_list.well_map)
    channel_colors = {}
    for well_id, well_channels in spike_list.well_map.items():
        color = well_colors[well_id]
        for ch in well_channels:
            if ch in channels:
                channel_colors[ch] = color
    
    def animate(frame):
        ax.clear()
        
        start_time = window_starts[frame]
        end_time = start_time + window_length
        
        # Get windowed data
        windowed_data = spike_list.get_time_window(start_time, end_time, channels)
        
        # Plot spikes
        y_positions = {ch: i for i, ch in enumerate(channels)}
        
        for ch in channels:
            if ch in windowed_data.spike_trains:
                spike_times = windowed_data.spike_trains[ch].spike_times + start_time
                y_pos = y_positions[ch]
                
                if len(spike_times) > 0:
                    ax.scatter(spike_times, [y_pos] * len(spike_times),
                              c=channel_colors.get(ch, '#2E86AB'),
                              s=1.0, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel ID') 
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels(channels)
        ax.set_xlim(start_time, end_time)
        ax.set_ylim(-0.5, len(channels) - 0.5)
        ax.set_title(f'Animated Raster Plot - Window: {start_time:.1f}s - {end_time:.1f}s')
        
        return ax.collections
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(window_starts),
        interval=interval, blit=False, repeat=True
    )
    
    if save_path:
        try:
            anim.save(save_path, writer='ffmpeg', fps=2)
        except Exception as e:
            warnings.warn(f"Could not save animation: {e}")
    
    return anim


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
    time_bins = np.arange(n_bins) * time_window
    
    # Calculate activity per well
    well_activities = {}
    
    for well_id, well_channels in spike_list.well_map.items():
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
    
    colors = _get_well_colors(spike_list.well_map)
    
    for well_id, activity in well_activities.items():
        if np.sum(activity) > 0:  # Only plot wells with activity
            ax.plot(time_bins[:len(activity)], activity, 
                   color=colors[well_id], label=f'Well {well_id}', linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spike Count per Time Window')
    ax.set_title(f'Well Activity Over Time (window = {time_window}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _get_well_colors(well_map: Dict[int, np.ndarray]) -> Dict[int, str]:
    """Get color mapping for wells."""
    import seaborn as sns
    
    n_wells = len(well_map)
    if n_wells == 1:
        return {list(well_map.keys())[0]: '#2E86AB'}
    
    colors = sns.color_palette("husl", n_wells).as_hex()
    return {well_id: colors[i] for i, well_id in enumerate(sorted(well_map.keys()))}


def _add_well_boundaries(ax, channels: List[int], well_map: Dict[int, np.ndarray]):
    """Add horizontal lines to separate wells in raster plot."""
    channel_to_y = {ch: i for i, ch in enumerate(channels)}
    
    # Find boundaries between wells
    boundaries = []
    current_well = None
    
    for i, ch in enumerate(channels):
        # Find which well this channel belongs to
        ch_well = None
        for well_id, well_channels in well_map.items():
            if ch in well_channels:
                ch_well = well_id
                break
        
        if current_well is not None and ch_well != current_well:
            boundaries.append(i - 0.5)
        
        current_well = ch_well
    
    # Draw boundary lines
    for boundary in boundaries:
        ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)


def _add_well_legend(ax, well_colors: Dict[int, str]):
    """Add legend showing well colors."""
    handles = []
    for well_id, color in well_colors.items():
        handles.append(plt.scatter([], [], c=color, s=50, label=f'Well {well_id}'))
    
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1))


def _add_raster_stats(ax, spike_list: SpikeList, channels: List[int]):
    """Add statistics text box to raster plot."""
    total_spikes = sum(spike_list.spike_trains[ch].n_spikes for ch in channels 
                      if ch in spike_list.spike_trains)
    
    mean_rate = np.mean([spike_list.spike_trains[ch].firing_rate for ch in channels 
                        if ch in spike_list.spike_trains])
    
    stats_text = f'Channels: {len(channels)}\nTotal Spikes: {total_spikes}\nMean Rate: {mean_rate:.2f} Hz'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=10)