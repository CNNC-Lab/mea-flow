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
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            try:
                plt.style.use('seaborn')
            except:
                warnings.warn(f"Could not set style '{style}', using default")
        
        # Set seaborn parameters for publication-ready plots
        sns.set_context("paper", font_scale=1.2)
        sns.set_palette("husl")
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'neutral': '#C73E1D',
            'background': '#F5F5F5'
        }
        
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
        grouping_col: str = 'condition',
        metrics_to_plot: Optional[List[str]] = None,
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
        grouping_col : str
            Column to group by (e.g., 'condition', 'well_id')
        metrics_to_plot : list of str, optional
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
            grouping_col=grouping_col,
            metrics_to_plot=metrics_to_plot,
            plot_type=plot_type,
            figsize=figsize or (15, 10),
            save_path=save_path
        )
    
    def plot_well_activity(
        self,
        spike_list: SpikeList,
        time_window: float = 1.0,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create activity heatmap showing well-based activity over time.
        
        Parameters
        ----------
        spike_list : SpikeList
            Spike data to analyze
        time_window : float
            Time window for binning (seconds)
        figsize : tuple, optional
            Figure size override
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
            time_window=time_window,
            figsize=figsize or (12, 8),
            save_path=save_path
        )
    
    def plot_electrode_map(
        self,
        spike_list: SpikeList,
        metric: str = 'firing_rate',
        well_id: Optional[int] = None,
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
            Specific well to plot (default: first well)
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
            figsize=figsize or (8, 8),
            save_path=save_path
        )
    
    def plot_embedding(
        self,
        embedding_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method_name: str = 'Embedding',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of manifold embedding.
        
        Parameters
        ----------
        embedding_data : np.ndarray
            Low-dimensional embedding data (N_samples x N_dims)
        labels : np.ndarray, optional
            Labels for coloring points
        method_name : str
            Name of embedding method for title
        figsize : tuple, optional
            Figure size override
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        matplotlib.Figure
            The created figure
        """
        from .manifold import plot_embedding
        
        return plot_embedding(
            embedding_data=embedding_data,
            labels=labels,
            method_name=method_name,
            figsize=figsize or (10, 8),
            save_path=save_path
        )
    
    def create_summary_report(
        self,
        spike_lists: Dict[str, SpikeList],
        metrics_df: pd.DataFrame,
        save_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive summary report with multiple plots.
        
        Parameters
        ----------
        spike_lists : dict
            Dictionary mapping condition names to SpikeList objects
        metrics_df : pd.DataFrame
            Computed metrics for all conditions
        save_dir : str, optional
            Directory to save all figures
            
        Returns
        -------
        dict
            Dictionary mapping plot names to figure objects
        """
        figures = {}
        
        # 1. Raster plots for each condition
        for condition, spike_list in spike_lists.items():
            fig = self.plot_raster(
                spike_list,
                time_range=(0, min(30, spike_list.recording_length)),  # First 30s
                save_path=f"{save_dir}/raster_{condition}.png" if save_dir else None
            )
            figures[f'raster_{condition}'] = fig
        
        # 2. Metrics comparison
        if 'condition' in metrics_df.columns:
            # Key activity metrics
            activity_metrics = ['mean_firing_rate', 'network_firing_rate', 'active_channels_count']
            available_activity = [m for m in activity_metrics if m in metrics_df.columns]
            
            if available_activity:
                fig = self.plot_metrics_comparison(
                    metrics_df,
                    metrics_to_plot=available_activity,
                    save_path=f"{save_dir}/activity_comparison.png" if save_dir else None
                )
                figures['activity_comparison'] = fig
            
            # Key regularity metrics
            regularity_metrics = ['cv_isi_mean', 'lv_isi_mean', 'entropy_isi_mean']
            available_regularity = [m for m in regularity_metrics if m in metrics_df.columns]
            
            if available_regularity:
                fig = self.plot_metrics_comparison(
                    metrics_df,
                    metrics_to_plot=available_regularity,
                    save_path=f"{save_dir}/regularity_comparison.png" if save_dir else None
                )
                figures['regularity_comparison'] = fig
            
            # Key synchrony metrics  
            synchrony_metrics = ['pearson_cc_mean', 'isi_distance', 'spike_distance']
            available_synchrony = [m for m in synchrony_metrics if m in metrics_df.columns]
            
            if available_synchrony:
                fig = self.plot_metrics_comparison(
                    metrics_df,
                    metrics_to_plot=available_synchrony,
                    save_path=f"{save_dir}/synchrony_comparison.png" if save_dir else None
                )
                figures['synchrony_comparison'] = fig
        
        # 3. Well activity heatmaps
        for condition, spike_list in spike_lists.items():
            if len(spike_list.well_map) > 1:  # Only if multiple wells
                fig = self.plot_well_activity(
                    spike_list,
                    save_path=f"{save_dir}/well_activity_{condition}.png" if save_dir else None
                )
                figures[f'well_activity_{condition}'] = fig
        
        return figures
    
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