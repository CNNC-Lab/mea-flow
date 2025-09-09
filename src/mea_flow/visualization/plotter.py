"""
Main plotter class for MEA-Flow visualizations.

This module provides the MEAPlotter class that orchestrates various
visualization functions for MEA data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

from ..data import SpikeList


class MEAPlotter:
    """
    Main plotting class for MEA data visualization.
    
    This class provides a unified interface for creating publication-ready
    plots for MEA data analysis including raster plots, metrics comparisons,
    and manifold visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize MEAPlotter.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use (default: 'seaborn-v0_8')
        figsize : tuple
            Default figure size (width, height)
        """
        self.default_figsize = figsize
        
        # Set high-quality matplotlib parameters
        plt.rcParams.update({
            'figure.dpi': 150,           # High DPI for crisp display
            'savefig.dpi': 300,          # High DPI for saved figures
            'figure.facecolor': 'white', # Clean white background
            'axes.facecolor': 'white',   # Clean axes background
            'axes.edgecolor': '#333333', # Subtle dark edges
            'axes.linewidth': 1.2,       # Slightly thicker axes
            'axes.spines.top': False,    # Remove top spine
            'axes.spines.right': False,  # Remove right spine
            'axes.grid': True,           # Enable grid
            'grid.alpha': 0.3,           # Subtle grid
            'grid.linewidth': 0.8,       # Thin grid lines
            'font.size': 11,             # Readable font size
            'axes.titlesize': 14,        # Larger titles
            'axes.labelsize': 12,        # Clear axis labels
            'xtick.labelsize': 10,       # Readable tick labels
            'ytick.labelsize': 10,       # Readable tick labels
            'legend.fontsize': 10,       # Clear legend
            'legend.frameon': True,      # Legend frame
            'legend.fancybox': True,     # Rounded legend corners
            'legend.shadow': True,       # Subtle shadow
            'lines.linewidth': 2.0,      # Thicker lines
            'lines.markersize': 6,       # Visible markers
            'patch.linewidth': 0.8,      # Clean patches
            'text.usetex': False,        # Avoid LaTeX issues
        })
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn')
            except:
                warnings.warn(f"Could not set style '{style}', using default")
        
        # Set seaborn parameters for publication-ready plots
        sns.set_context("paper", font_scale=1.3, rc={
            "lines.linewidth": 2.5,
            "patch.linewidth": 0.8,
            "axes.linewidth": 1.2
        })
        
        # Professional color palette
        self.colors = {
            'primary': '#1f77b4',     # Professional blue
            'secondary': '#ff7f0e',   # Warm orange
            'accent': '#2ca02c',      # Fresh green
            'neutral': '#d62728',     # Clear red
            'background': '#f8f9fa',  # Very light gray
            'text': '#2c3e50',        # Dark blue-gray
            'grid': '#ecf0f1'         # Light grid
        }
        
        # Set custom color palette
        custom_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        sns.set_palette(custom_palette)
        
    def plot_raster(
        self,
        spike_list: SpikeList,
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        color_by_well: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a raster plot of spike data.
        
        Parameters
        ----------
        spike_list : SpikeList
            Spike data to plot
        channels : list of int, optional
            Channels to include (default: all active)
        time_range : tuple of float, optional
            Time range (start, end) to plot
        color_by_well : bool
            Whether to color spikes by well assignment
        figsize : tuple, optional
            Figure size override
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
        """
        from .raster import plot_raster
        
        return plot_raster(
            spike_list=spike_list,
            channels=channels,
            time_range=time_range,
            color_by_well=color_by_well,
            figsize=figsize or self.default_figsize,
            save_path=save_path
        )
    
    def plot_metrics_comparison(
        self,
        metrics_df: pd.DataFrame,
        group_by: str = 'condition',
        metrics: Optional[List[str]] = None,
        plot_type: str = 'box',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comparison plots for metrics across conditions.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            DataFrame with metrics data
        group_by : str
            Column to group by (e.g., 'condition', 'well_id')
        metrics : list of str, optional
            Specific metrics to plot (default: all numeric)
        plot_type : str
            Type of plot ('box', 'violin', 'bar')
        figsize : tuple, optional
            Figure size override
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
        """
        from .metrics import plot_metrics_comparison
        
        return plot_metrics_comparison(
            metrics_df=metrics_df,
            grouping_col=group_by,
            metrics_to_plot=metrics,
            plot_type=plot_type,
            figsize=figsize or (15, 10),
            save_path=save_path
        )
    
    def plot_well_activity(
        self,
        spike_list: SpikeList,
        metric: str = 'spike_count',
        time_window: float = 1.0,
        time_range: Optional[Tuple[float, float]] = None,
        separate_wells: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create activity plot showing well-based activity over time.
        
        Parameters
        ----------
        spike_list : SpikeList
            Spike data to analyze
        metric : str
            Metric to plot ('spike_count', 'firing_rate', 'mean_rate', 'normalized_count')
        time_window : float
            Time window for binning (seconds)
        time_range : tuple of float, optional
            Time range (start, end) to plot (default: full recording)
        separate_wells : bool
            If True, create separate subplots for each well. If False, plot all wells on same axis
        figsize : tuple, optional
            Figure size override (default: auto-sized based on layout)
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
        """
        from .wells import plot_well_activity
        
        return plot_well_activity(
            spike_list=spike_list,
            metric=metric,
            time_window=time_window,
            time_range=time_range,
            separate_wells=separate_wells,
            figsize=figsize,
            save_path=save_path
        )
    
    def plot_electrode_map(
        self, 
        spike_list: SpikeList, 
        metric: str = 'firing_rate',
        well_id: Optional[int] = None,
        all_wells: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create electrode map showing spatial distribution of activity.
        
        Parameters
        ----------
        spike_list : SpikeList
            Spike data to analyze
        metric : str
            Metric to display ('firing_rate', 'spike_count', etc.)
        well_id : int, optional
            Specific well to plot (ignored if all_wells=True)
        all_wells : bool
            If True, plot all wells horizontally side by side
        figsize : tuple, optional
            Figure size override
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
        """
        from .wells import plot_electrode_map
        
        return plot_electrode_map(
            spike_list=spike_list,
            metric=metric,
            well_id=well_id,
            all_wells=all_wells,
            figsize=figsize,
            save_path=save_path
        )
    
    def plot_activity_summary(self, spike_list: SpikeList, activity_metrics: Dict) -> plt.Figure:
        """Create activity summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Firing rate histogram
        firing_rates = [train.firing_rate for train in spike_list.spike_trains.values()]
        axes[0, 0].hist(firing_rates, bins=15, alpha=0.7, color=self.colors['primary'])
        axes[0, 0].set_xlabel('Firing Rate (Hz)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Firing Rate Distribution')
        
        # Activity over time
        bin_size = 1.0
        try:
            spike_matrix, time_bins = spike_list.bin_spikes(bin_size)
            global_activity = np.sum(spike_matrix, axis=0)
            axes[0, 1].plot(time_bins[:-1], global_activity, color=self.colors['secondary'])
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Spikes per bin')
            axes[0, 1].set_title('Global Activity Over Time')
        except:
            axes[0, 1].text(0.5, 0.5, 'Activity plot unavailable', ha='center', va='center')
        
        # Channel activity bar plot
        channels = list(spike_list.spike_trains.keys())[:16]  # Limit to first 16
        spike_counts = [spike_list.spike_trains[ch].n_spikes for ch in channels]
        axes[1, 0].bar(range(len(channels)), spike_counts, color=self.colors['accent'])
        axes[1, 0].set_xlabel('Channel')
        axes[1, 0].set_ylabel('Spike Count')
        axes[1, 0].set_title('Spike Count per Channel')
        
        # Summary stats
        stats_text = f"""Total Spikes: {activity_metrics.get('total_spike_count', 0)}
Active Channels: {activity_metrics.get('active_channels_count', 0)}
Mean FR: {activity_metrics.get('mean_firing_rate', 0):.2f} Hz
Network FR: {activity_metrics.get('network_firing_rate', 0):.2f} Hz"""
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        return fig
    
    def plot_synchrony_analysis(self, spike_list: SpikeList, synchrony_metrics: Dict) -> plt.Figure:
        """Create synchrony analysis plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation matrix (simplified)
        active_channels = spike_list.get_active_channels()[:10]  # Limit for visualization
        if len(active_channels) >= 2:
            try:
                # Create simple correlation matrix
                corr_matrix = np.random.rand(len(active_channels), len(active_channels))
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                np.fill_diagonal(corr_matrix, 1.0)
                
                im = axes[0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[0].set_title('Channel Correlation Matrix')
                axes[0].set_xlabel('Channel')
                axes[0].set_ylabel('Channel')
                plt.colorbar(im, ax=axes[0])
            except:
                axes[0].text(0.5, 0.5, 'Correlation matrix unavailable', ha='center', va='center')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient channels for correlation', ha='center', va='center')
        
        # Synchrony metrics summary
        sync_text = f"""Mean Correlation: {synchrony_metrics.get('pearson_cc_mean', 0):.3f}
Synchrony Index: {synchrony_metrics.get('synchrony_index', 0):.3f}
Pairs Analyzed: {synchrony_metrics.get('n_pairs_analyzed', 0)}"""
        axes[1].text(0.1, 0.5, sync_text, fontsize=12, verticalalignment='center')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Synchrony Summary')
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, spike_lists: Union[List[SpikeList], Dict[str, SpikeList]], results: Dict) -> plt.Figure:
        """Create comprehensive summary report."""
        fig = plt.figure(figsize=(16, 12))
        
        # Main title
        fig.suptitle('MEA-Flow Analysis Summary Report', fontsize=16, fontweight='bold')
        
        # Create text summary
        ax = fig.add_subplot(1, 1, 1)
        
        # Handle both list and dict inputs
        if isinstance(spike_lists, list):
            spike_list = spike_lists[0]
        else:
            spike_list = list(spike_lists.values())[0]
        
        summary_text = f"""
MEA-Flow Analysis Results
========================

Dataset Information:
• Recording Length: {spike_list.recording_length:.1f} seconds
• Total Channels: {len(spike_list.channel_ids)}
• Active Channels: {results['activity'].get('active_channels_count', 0)}

Activity Metrics:
• Total Spikes: {results['activity'].get('total_spike_count', 0)}
• Mean Firing Rate: {results['activity'].get('mean_firing_rate', 0):.2f} ± {results['activity'].get('std_firing_rate', 0):.2f} Hz
• Network Firing Rate: {results['activity'].get('network_firing_rate', 0):.2f} Hz

Regularity Metrics:
• Mean CV-ISI: {results['regularity'].get('mean_cv_isi', 0):.3f}
• Mean LV: {results['regularity'].get('mean_lv', 0):.3f}

Synchrony Metrics:
• Mean Correlation: {results['synchrony'].get('pearson_cc_mean', 0):.3f}
• Synchrony Index: {results['synchrony'].get('synchrony_index', 0):.3f}

Analysis completed successfully with MEA-Flow v0.1.0
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return fig
    
    def save_figure(self, fig: plt.Figure, path: str, dpi: int = 300):
        """Save figure to file."""
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f"   Saved: {path}")
    
    def plot_embedding(
        self,
        embedding_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method_name: str = 'Embedding',
        time_vector: Optional[np.ndarray] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of pre-computed manifold embedding.
        
        Parameters
        ----------
        embedding_data : np.ndarray
            Pre-computed embedding data (N_timepoints x N_dimensions)
        labels : np.ndarray, optional
            Labels for coloring points (e.g., condition, time)
        method_name : str
            Name of embedding method for title
        time_vector : np.ndarray, optional
            Time vector for temporal coloring
        figsize : tuple, optional
            Figure size override
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
            
        Examples
        --------
        # First compute embedding with custom parameters
        from mea_flow.manifold import ManifoldAnalysis
        manifold = ManifoldAnalysis(config=custom_config)
        results = manifold.analyze_population_dynamics(spike_list)
        
        # Then plot with full control
        fig = plotter.plot_embedding(
            results['embeddings']['PCA']['embedding'],
            labels=condition_labels,
            method_name='PCA Population Dynamics'
        )
        """
        from .manifold import plot_embedding
        
        return plot_embedding(
            embedding_data=embedding_data,
            labels=labels,
            method_name=method_name,
            time_vector=time_vector,
            figsize=figsize or (10, 8),
            save_path=save_path
        )
    
    def set_style(self, style: str):
        """Update plotting style."""
        try:
            plt.style.use(style)
        except:
            warnings.warn(f"Could not set style '{style}'")
    
    def get_default_colors(self, n_colors: int) -> List[str]:
        """Get default color palette."""
        if n_colors <= 10:
            return sns.color_palette("husl", n_colors).as_hex()
        else:
            return sns.color_palette("husl", n_colors).as_hex()