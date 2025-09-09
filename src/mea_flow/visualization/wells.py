"""
Well-based visualizations for MEA data.

This module provides functions for visualizing MEA data organized by wells,
including electrode maps and spatial activity patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
from typing import Optional, Tuple, Dict, List
import warnings
from tqdm.auto import tqdm

from ..data import SpikeList


def plot_electrode_map(
    spike_list: SpikeList,
    metric: str = 'firing_rate',
    well_id: Optional[int] = None,
    all_wells: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create electrode map showing spatial distribution of activity metrics.
    
    This function creates a 4x4 grid visualization of electrode activity
    for a specific well or all wells side by side, with color-coding 
    representing the chosen metric and round markers at electrode centers.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to analyze
    metric : str
        Metric to display ('firing_rate', 'spike_count', 'cv_isi', etc.)
    well_id : int, optional
        Specific well to plot (ignored if all_wells=True)
    all_wells : bool
        If True, plot all wells horizontally side by side
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Determine wells to plot
    if all_wells:
        wells_to_plot = list(spike_list.well_map.keys())
        # Filter to wells with active channels
        active_wells = []
        for wid in wells_to_plot:
            channels = spike_list.well_map[wid]
            active_in_well = [ch for ch in channels if ch in spike_list.get_active_channels()]
            if len(active_in_well) > 0:
                active_wells.append(wid)
        wells_to_plot = active_wells
        
        if len(wells_to_plot) == 0:
            warnings.warn("No wells with active channels found")
            fig, ax = plt.subplots(figsize=figsize or (8, 8))
            ax.text(0.5, 0.5, 'No active wells', 
                   transform=ax.transAxes, ha='center', va='center')
            return fig
    else:
        # Select single well to plot
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
                fig, ax = plt.subplots(figsize=figsize or (8, 8))
                ax.text(0.5, 0.5, 'No active wells', 
                       transform=ax.transAxes, ha='center', va='center')
                return fig
        
        if well_id not in spike_list.well_map:
            raise ValueError(f"Well {well_id} not found in well map")
        
        wells_to_plot = [well_id]
    
    # Calculate figure size if not provided
    n_wells = len(wells_to_plot)
    if figsize is None:
        if all_wells:
            figsize = (6 * n_wells, 6)  # Horizontal layout
        else:
            figsize = (8, 8)  # Single well
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_wells, figsize=figsize)
    if n_wells == 1:
        axes = [axes]
    
    # Calculate global min/max for consistent color scaling
    all_values = []
    well_data = {}
    
    for well_id in wells_to_plot:
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
        for i, ch in enumerate(well_channels[:16]):  # Only take first 16 channels
            row = i // 4
            col = i % 4
            grid_data[row, col] = metric_values.get(ch, np.nan)
        
        well_data[well_id] = {'grid_data': grid_data, 'metric_values': metric_values, 'channels': well_channels}
        
        # Collect values for global scaling
        valid_values = [v for v in metric_values.values() if not np.isnan(v)]
        all_values.extend(valid_values)
    
    # Set color scale limits
    if len(all_values) > 0:
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin, vmax = 0, 1
    
    # Plot each well
    for idx, well_id in enumerate(wells_to_plot):
        ax = axes[idx]
        data = well_data[well_id]
        grid_data = data['grid_data']
        metric_values = data['metric_values']
        well_channels = data['channels']
        
        # Create heatmap
        im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Add round markers at electrode centers
        for i in range(4):
            for j in range(4):
                channel_idx = i * 4 + j
                if channel_idx < len(well_channels):
                    # Add circular marker at electrode center
                    circle = plt.Circle((j, i), 0.35, fill=False, color='white', linewidth=2, alpha=0.8)
                    ax.add_patch(circle)
        
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
        
        ax.set_title(f'Well {well_id} - {_format_metric_name(metric)}')
    
    # Add shared colorbar
    if n_wells > 1:
        # Create colorbar for all subplots
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
    else:
        # Single subplot colorbar
        cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
    
    cbar.set_label(_format_metric_name(metric))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_animated_electrode_map(
    spike_list: SpikeList,
    metric: str = 'firing_rate',
    window_length: float = 10.0,
    step_size: float = 2.0,
    well_id: Optional[int] = None,
    all_wells: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    interval: int = 500,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """
    Create animated electrode map showing activity changes over time windows.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to animate
    metric : str
        Metric to display ('firing_rate', 'spike_count', 'cv_isi', etc.)
    window_length : float
        Length of time window for each frame (seconds)
    step_size : float
        Step size between frames (seconds)
    well_id : int, optional
        Specific well to plot (ignored if all_wells=True)
    all_wells : bool
        If True, plot all wells horizontally side by side
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    interval : int
        Animation interval in milliseconds
    save_path : str, optional
        Path to save animation
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object
    """
    # Determine wells to plot
    if all_wells:
        wells_to_plot = list(spike_list.well_map.keys())
        # Filter to wells with active channels
        active_wells = []
        for wid in wells_to_plot:
            channels = spike_list.well_map[wid]
            active_in_well = [ch for ch in channels if ch in spike_list.get_active_channels()]
            if len(active_in_well) > 0:
                active_wells.append(wid)
        wells_to_plot = active_wells
        
        if len(wells_to_plot) == 0:
            warnings.warn("No wells with active channels found")
            return None
    else:
        # Select single well to plot
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
                return None
        
        if well_id not in spike_list.well_map:
            raise ValueError(f"Well {well_id} not found in well map")
        
        wells_to_plot = [well_id]
    
    # Calculate time windows
    max_start_time = spike_list.recording_length - window_length
    window_starts = np.arange(0, max_start_time + step_size, step_size)
    
    print(f"Creating electrode map animation with {len(window_starts)} frames...")
    
    # Calculate figure size if not provided
    n_wells = len(wells_to_plot)
    if figsize is None:
        if all_wells:
            figsize = (6 * n_wells, 6)  # Horizontal layout
        else:
            figsize = (8, 8)  # Single well
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_wells, figsize=figsize)
    if n_wells == 1:
        axes = [axes]
    
    # Calculate global min/max for consistent color scaling across all frames
    print("Calculating global color scale...")
    all_values = []
    
    for frame_idx in range(0, len(window_starts), max(1, len(window_starts) // 10)):  # Sample frames for scaling
        start_time = window_starts[frame_idx]
        end_time = start_time + window_length
        windowed_data = spike_list.get_time_window(start_time, end_time)
        
        for well_id in wells_to_plot:
            well_channels = spike_list.well_map[well_id]
            
            for ch in well_channels:
                if ch in windowed_data.spike_trains:
                    train = windowed_data.spike_trains[ch]
                    
                    if metric == 'firing_rate':
                        value = train.firing_rate
                    elif metric == 'spike_count':
                        value = train.n_spikes
                    elif metric == 'cv_isi':
                        isis = train.get_isi()
                        if len(isis) > 1:
                            value = np.std(isis) / np.mean(isis)
                        else:
                            value = np.nan
                    else:
                        value = np.nan
                    
                    if not np.isnan(value):
                        all_values.append(value)
    
    # Set color scale limits
    if len(all_values) > 0:
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin, vmax = 0, 1
    
    # Progress bar for animation frames
    pbar = tqdm(total=len(window_starts), desc="Generating electrode map frames", 
                unit="frame", leave=True)
    
    # Store image objects for consistent colorbar
    images = []
    
    def animate(frame):
        start_time = window_starts[frame]
        end_time = start_time + window_length
        
        # Get windowed data
        windowed_data = spike_list.get_time_window(start_time, end_time)
        
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Plot each well
        for idx, well_id in enumerate(wells_to_plot):
            ax = axes[idx]
            well_channels = spike_list.well_map[well_id]
            
            # Calculate metric values for this time window
            metric_values = {}
            
            for ch in well_channels:
                if ch in windowed_data.spike_trains:
                    train = windowed_data.spike_trains[ch]
                    
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
                        metric_values[ch] = np.nan
                else:
                    metric_values[ch] = 0.0  # Inactive channel
            
            # Create 4x4 grid
            grid_data = np.full((4, 4), np.nan)
            
            # Map channels to grid positions
            for i, ch in enumerate(well_channels[:16]):
                row = i // 4
                col = i % 4
                grid_data[row, col] = metric_values.get(ch, np.nan)
            
            # Create heatmap
            im = ax.imshow(grid_data, cmap='viridis', interpolation='nearest', 
                          vmin=vmin, vmax=vmax)
            
            # Store first image for colorbar
            if frame == 0 and idx == 0:
                images.append(im)
            
            # Add round markers at electrode centers
            for i in range(4):
                for j in range(4):
                    channel_idx = i * 4 + j
                    if channel_idx < len(well_channels):
                        # Add circular marker at electrode center
                        circle = plt.Circle((j, i), 0.35, fill=False, color='white', 
                                          linewidth=2, alpha=0.8)
                        ax.add_patch(circle)
            
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
            
            # Title with time window
            ax.set_title(f'Well {well_id} - {_format_metric_name(metric)}\n'
                        f'Time: {start_time:.1f}s - {end_time:.1f}s')
        
        # Update progress bar
        pbar.update(1)
        
        return [ax for ax in axes]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(window_starts),
        interval=interval, blit=False, repeat=True
    )
    
    # Add colorbar after first frame
    def add_colorbar():
        if len(images) > 0:
            if n_wells > 1:
                cbar = fig.colorbar(images[0], ax=axes, shrink=0.8, aspect=20)
            else:
                cbar = plt.colorbar(images[0], ax=axes[0], shrink=0.8)
            cbar.set_label(_format_metric_name(metric))
    
    # Add colorbar after first frame is drawn
    fig.canvas.draw()
    add_colorbar()
    
    if save_path:
        try:
            print(f"Saving electrode map animation to {save_path}...")
            anim.save(save_path, writer='ffmpeg', fps=2)
            print("Animation saved successfully!")
        except Exception as e:
            warnings.warn(f"Could not save animation: {e}")
    
    # Close progress bar
    pbar.close()
    
    return anim


def plot_well_grid(
    spike_lists: Dict[str, SpikeList],
    metric: str = 'firing_rate',
    well_selection: str = 'average',
    specific_well_id: Optional[int] = None,
    show_std: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid showing electrode maps for multiple conditions.
    
    Parameters
    ----------
    spike_lists : dict
        Dictionary mapping condition names to SpikeList objects
    metric : str
        Metric to display across all maps
    well_selection : str
        Selection mode: 'average' (average across all wells), 
        'first' (first well only), or 'specific' (specific well ID)
    specific_well_id : int, optional
        Specific well ID to plot (used when well_selection='specific')
    show_std : bool
        If True and well_selection='average', show both mean and std panels
    figsize : tuple, optional
        Figure size (auto-calculated if None)
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
        fig, ax = plt.subplots(figsize=figsize or (15, 5))
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Determine subplot layout
    if well_selection == 'average' and show_std:
        # Two rows: mean and std
        n_rows = 2
        subplot_labels = ['Mean', 'Std Dev']
        if figsize is None:
            figsize = (5 * n_conditions, 8)
    else:
        # Single row
        n_rows = 1
        subplot_labels = ['']
        if figsize is None:
            figsize = (5 * n_conditions, 5)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_conditions, figsize=figsize)
    if n_rows == 1 and n_conditions == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_conditions == 1:
        axes = [[ax] for ax in axes]
    
    # Collect all metric values to set consistent color scale
    all_values = []
    for spike_list in spike_lists.values():
        for ch in spike_list.get_active_channels():
            train = spike_list.spike_trains[ch]
            if metric == 'firing_rate':
                all_values.append(train.firing_rate)
            elif metric == 'spike_count':
                all_values.append(train.n_spikes)
            elif metric == 'cv_isi':
                isis = train.get_isi()
                if len(isis) > 1:
                    all_values.append(np.std(isis) / np.mean(isis))
    
    if len(all_values) == 0:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = np.min(all_values), np.max(all_values)
    
    # Calculate metric data for all conditions
    condition_data = {}
    
    for condition, spike_list in spike_lists.items():
        if well_selection == 'average':
            # Average across all wells for this condition
            all_well_channels = set()
            for well_channels in spike_list.well_map.values():
                all_well_channels.update(well_channels[:16])  # Standard 4x4 grid
            
            # Calculate mean and std metric values across wells
            metric_means = {}
            metric_stds = {}
            
            for ch in all_well_channels:
                ch_values = []
                
                # Collect values from all wells that contain this channel
                for well_id, well_channels in spike_list.well_map.items():
                    if ch in well_channels and ch in spike_list.spike_trains:
                        train = spike_list.spike_trains[ch]
                        
                        if metric == 'firing_rate':
                            ch_values.append(train.firing_rate)
                        elif metric == 'spike_count':
                            ch_values.append(train.n_spikes)
                        elif metric == 'cv_isi':
                            isis = train.get_isi()
                            if len(isis) > 1:
                                ch_values.append(np.std(isis) / np.mean(isis))
                
                # Calculate mean and std for this channel
                if len(ch_values) > 0:
                    metric_means[ch] = np.mean(ch_values)
                    metric_stds[ch] = np.std(ch_values) if len(ch_values) > 1 else 0.0
                else:
                    metric_means[ch] = 0.0
                    metric_stds[ch] = 0.0
            
            condition_data[condition] = {
                'means': metric_means,
                'stds': metric_stds,
                'channels': list(range(16)),
                'title_suffix': '(averaged across wells)'
            }
            
        elif well_selection == 'specific':
            # Use specific well ID
            if specific_well_id not in spike_list.well_map:
                warnings.warn(f"Well {specific_well_id} not found in {condition}, using first well")
                well_id = list(spike_list.well_map.keys())[0]
            else:
                well_id = specific_well_id
                
            well_channels = spike_list.well_map[well_id]
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
                        metric_values[ch] = np.nan
                else:
                    metric_values[ch] = 0.0
            
            condition_data[condition] = {
                'means': metric_values,
                'stds': {},
                'channels': well_channels[:16],
                'title_suffix': f'(well {well_id})'
            }
            
        else:  # well_selection == 'first'
            # Use first well for each condition
            well_id = list(spike_list.well_map.keys())[0]
            well_channels = spike_list.well_map[well_id]
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
                        metric_values[ch] = np.nan
                else:
                    metric_values[ch] = 0.0
            
            condition_data[condition] = {
                'means': metric_values,
                'stds': {},
                'channels': well_channels[:16],
                'title_suffix': f'(well {well_id})'
            }
    
    # Plot each condition and statistic
    for row_idx in range(n_rows):
        stat_type = 'means' if row_idx == 0 else 'stds'
        
        for col_idx, (condition, data) in enumerate(condition_data.items()):
            ax = axes[row_idx][col_idx]
            
            # Select data to plot
            if stat_type == 'stds' and well_selection == 'average' and show_std:
                values_to_plot = data['stds']
                title_prefix = 'Std Dev - '
            else:
                values_to_plot = data['means']
                title_prefix = 'Mean - ' if (well_selection == 'average' and show_std) else ''
            
            # Create 4x4 grid
            grid_data = np.full((4, 4), np.nan)
            channels_to_plot = data['channels']
            
            for j, ch in enumerate(channels_to_plot):
                row = j // 4
                col = j % 4
                grid_data[row, col] = values_to_plot.get(ch, np.nan)
            
            # Plot with appropriate color scale
            if stat_type == 'stds':
                # Use different color scale for std dev (0 to max std)
                std_values = [v for v in values_to_plot.values() if not np.isnan(v)]
                if len(std_values) > 0:
                    im = ax.imshow(grid_data, cmap='plasma', vmin=0, vmax=np.max(std_values), 
                                  interpolation='nearest')
                else:
                    im = ax.imshow(grid_data, cmap='plasma', interpolation='nearest')
            else:
                im = ax.imshow(grid_data, cmap='viridis', vmin=vmin, vmax=vmax, 
                              interpolation='nearest')
            
            # Add channel labels
            for row in range(4):
                for col in range(4):
                    ch_idx = row * 4 + col
                    if ch_idx < len(channels_to_plot):
                        ch = channels_to_plot[ch_idx]
                        ax.text(col, row, f'{ch}', ha='center', va='center',
                               fontsize=8, color='white', weight='bold')
            
            # Set title
            ax.set_title(f'{title_prefix}{condition}\n{data["title_suffix"]}')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add colorbars
    if well_selection == 'average' and show_std and n_rows == 2:
        # Separate colorbars for mean and std
        fig.subplots_adjust(right=0.82)
        
        # Mean colorbar (top row)
        mean_cbar_ax = fig.add_axes([0.84, 0.55, 0.015, 0.35])
        mean_im = None
        for col_idx in range(n_conditions):
            ax = axes[0][col_idx]
            for child in ax.get_children():
                if hasattr(child, 'get_array'):
                    mean_im = child
                    break
            if mean_im:
                break
        
        if mean_im:
            mean_cbar = fig.colorbar(mean_im, cax=mean_cbar_ax)
            mean_cbar.set_label(f'Mean {_format_metric_name(metric)}')
        
        # Std colorbar (bottom row)
        std_cbar_ax = fig.add_axes([0.84, 0.1, 0.015, 0.35])
        std_im = None
        for col_idx in range(n_conditions):
            ax = axes[1][col_idx]
            for child in ax.get_children():
                if hasattr(child, 'get_array'):
                    std_im = child
                    break
            if std_im:
                break
        
        if std_im:
            std_cbar = fig.colorbar(std_im, cax=std_cbar_ax)
            std_cbar.set_label(f'Std Dev {_format_metric_name(metric)}')
    else:
        # Single colorbar
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
    metric: str = 'spike_count',
    time_window: float = 1.0,
    time_range: Optional[Tuple[float, float]] = None,
    separate_wells: bool = False,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot activity levels for each well over time.
    
    Parameters
    ----------
    spike_list : SpikeList
        Spike data to analyze
    metric : str
        Metric to plot ('spike_count', 'firing_rate', 'mean_rate', 'normalized_count')
    time_window : float
        Time window for binning activity (seconds)
    time_range : tuple of float, optional
        Time range (start, end) to plot (default: full recording)
    separate_wells : bool
        If True, create separate subplots for each well. If False, plot all wells on same axis
    figsize : tuple, optional
        Figure size (default: auto-sized based on layout)
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Determine time range
    if time_range is not None:
        start_time, end_time = time_range
        plot_duration = end_time - start_time
    else:
        start_time = 0.0
        end_time = spike_list.recording_length
        plot_duration = spike_list.recording_length
    
    # Calculate time bins for the specified range
    n_bins = int(np.ceil(plot_duration / time_window))
    time_bins = np.linspace(start_time, end_time, n_bins + 1)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
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
        
        # Calculate activity for each time bin
        activity_values = np.zeros(n_bins)
        
        for bin_idx in range(n_bins):
            bin_start = time_bins[bin_idx]
            bin_end = time_bins[bin_idx + 1]
            
            bin_spike_count = 0
            for ch in active_channels:
                spike_times = spike_list.spike_trains[ch].spike_times
                # Count spikes in this time bin
                spikes_in_bin = np.sum((spike_times >= bin_start) & (spike_times < bin_end))
                bin_spike_count += spikes_in_bin
            
            # Calculate the requested metric
            if metric == 'spike_count':
                activity_values[bin_idx] = bin_spike_count
            elif metric == 'firing_rate':
                # Spikes per second in this bin
                activity_values[bin_idx] = bin_spike_count / time_window
            elif metric == 'mean_rate':
                # Mean firing rate across active channels
                if len(active_channels) > 0:
                    activity_values[bin_idx] = bin_spike_count / (time_window * len(active_channels))
                else:
                    activity_values[bin_idx] = 0.0
            elif metric == 'normalized_count':
                # Normalized by number of active channels
                if len(active_channels) > 0:
                    activity_values[bin_idx] = bin_spike_count / len(active_channels)
                else:
                    activity_values[bin_idx] = 0.0
            else:
                # Default to spike count
                activity_values[bin_idx] = bin_spike_count
        
        well_activities[well_id] = activity_values
    
    # Determine figure layout
    n_wells = len([w for w in well_activities.keys() if np.sum(well_activities[w]) > 0])
    
    if separate_wells and n_wells > 1:
        # Create separate subplots for each well (vertically stacked)
        n_cols = 1
        n_rows = n_wells
        
        if figsize is None:
            figsize = (10, 4 * n_rows)
        
        plt.ioff()  # Turn off interactive mode to prevent automatic display
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_wells == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each well in its own subplot
        plot_idx = 0
        for well_id, activity in well_activities.items():
            if np.sum(activity) > 0:  # Only plot wells with activity
                ax = axes[plot_idx]
                
                ax.plot(time_centers, activity, 
                       color=well_colors[well_id], 
                       linewidth=2, alpha=0.8)
                
                # Add filled area
                ax.fill_between(time_centers, activity, 
                               alpha=0.3, color=well_colors[well_id])
                
                # Set proper axis limits
                if time_range is not None:
                    ax.set_xlim(time_range[0], time_range[1])
                else:
                    ax.set_xlim(0, spike_list.recording_length)
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
                
                # Set y-axis label mapping
                ylabel_mapping = {
                    'spike_count': 'Spike Count per Time Window',
                    'firing_rate': 'Firing Rate (Hz)',
                    'mean_rate': 'Mean Firing Rate per Channel (Hz)',
                    'normalized_count': 'Normalized Spike Count per Channel'
                }
                
                # Labels and formatting
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(ylabel_mapping.get(metric, 'Activity'))
                ax.set_title(f'Well {well_id}')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                plot_idx += 1
        
        # Remove unused subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        # Overall title
        title_parts = [f'Well Activity Over Time']
        if time_range is not None:
            title_parts.append(f'({time_range[0]:.1f}s - {time_range[1]:.1f}s)')
        title_parts.append(f'(window = {time_window}s)')
        fig.suptitle(' '.join(title_parts), fontsize=14)
        
    else:
        # Plot all wells on the same axis (original behavior)
        if figsize is None:
            figsize = (12, 6)
        
        plt.ioff()  # Turn off interactive mode to prevent automatic display
        fig, ax = plt.subplots(figsize=figsize)
        
        for well_id, activity in well_activities.items():
            if np.sum(activity) > 0:  # Only plot wells with activity
                ax.plot(time_centers, activity, 
                       color=well_colors[well_id], label=f'Well {well_id}', 
                       linewidth=2, alpha=0.8)
                
                # Add filled area
                ax.fill_between(time_centers, activity, 
                               alpha=0.3, color=well_colors[well_id])
        
        # Set proper axis limits
        if time_range is not None:
            ax.set_xlim(time_range[0], time_range[1])
        else:
            ax.set_xlim(0, spike_list.recording_length)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Set y-axis label mapping
        ylabel_mapping = {
            'spike_count': 'Spike Count per Time Window',
            'firing_rate': 'Firing Rate (Hz)',
            'mean_rate': 'Mean Firing Rate per Channel (Hz)',
            'normalized_count': 'Normalized Spike Count per Channel'
        }
        
        # Labels and formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel_mapping.get(metric, 'Activity'))
        
        # Create informative title
        title_parts = [f'Well Activity Over Time']
        if time_range is not None:
            title_parts.append(f'({time_range[0]:.1f}s - {time_range[1]:.1f}s)')
        title_parts.append(f'(window = {time_window}s)')
        ax.set_title(' '.join(title_parts))
        
        if len(spike_list.well_map) > 1:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.ion()  # Turn interactive mode back on
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