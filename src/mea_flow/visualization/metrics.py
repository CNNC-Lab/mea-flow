"""
Visualization functions for MEA metrics and statistical comparisons.

This module provides functions for creating publication-ready plots
of computed metrics across different experimental conditions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import warnings
from scipy import stats


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    grouping_col: str = 'condition',
    metrics_to_plot: Optional[List[str]] = None,
    plot_type: str = 'box',
    statistical_test: bool = True,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison plots for metrics across experimental conditions.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing computed metrics
    grouping_col : str
        Column name for grouping (e.g., 'condition', 'well_id')
    metrics_to_plot : list of str, optional
        List of metric names to plot (default: all numeric columns)
    plot_type : str
        Type of plot: 'box', 'violin', 'bar', 'strip'
    statistical_test : bool
        Whether to perform and display statistical tests
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Determine metrics to plot
    if metrics_to_plot is None:
        # Select numeric columns, excluding metadata columns
        exclude_cols = [grouping_col, 'group_type', 'group_id', 'n_channels', 
                       'recording_length', 'channel_id', 'well_id']
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        metrics_to_plot = [col for col in numeric_cols if col not in exclude_cols]
    
    # Filter out columns with all NaN values
    valid_metrics = []
    for metric in metrics_to_plot:
        if metric in metrics_df.columns and not metrics_df[metric].isna().all():
            valid_metrics.append(metric)
    
    metrics_to_plot = valid_metrics
    
    if len(metrics_to_plot) == 0:
        warnings.warn("No valid metrics to plot")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No valid metrics to plot', 
                transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Determine subplot layout
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Prepare data
        plot_data = metrics_df[[grouping_col, metric]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, f'No data for\n{metric}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Create plot based on type
        try:
            if plot_type == 'box':
                sns.boxplot(data=plot_data, x=grouping_col, y=metric, ax=ax)
            elif plot_type == 'violin':
                sns.violinplot(data=plot_data, x=grouping_col, y=metric, ax=ax)
            elif plot_type == 'bar':
                sns.barplot(data=plot_data, x=grouping_col, y=metric, ax=ax, 
                           capsize=0.1, errcolor='black')
            elif plot_type == 'strip':
                sns.stripplot(data=plot_data, x=grouping_col, y=metric, ax=ax,
                             size=4, alpha=0.7)
            else:
                warnings.warn(f"Unknown plot type '{plot_type}', using box plot")
                sns.boxplot(data=plot_data, x=grouping_col, y=metric, ax=ax)
        except Exception as e:
            warnings.warn(f"Failed to plot {metric}: {e}")
            ax.text(0.5, 0.5, f'Plot failed for\n{metric}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Add statistical annotations
        if statistical_test and len(plot_data[grouping_col].unique()) >= 2:
            _add_statistical_annotations(ax, plot_data, grouping_col, metric)
        
        # Formatting
        ax.set_title(_format_metric_name(metric), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(_format_metric_name(metric))
        
        # Rotate x-labels if needed
        if len(plot_data[grouping_col].unique()) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    # Remove extra subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    # Overall title
    fig.suptitle(f'Metrics Comparison by {_format_metric_name(grouping_col)}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance for discriminative analysis.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with feature names and importance scores
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Assume importance_df has columns ['feature', 'importance']
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        raise ValueError("importance_df must have 'feature' and 'importance' columns")
    
    # Sort by importance and select top N
    top_features = importance_df.nlargest(top_n, 'importance')
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features['importance'], 
                   color=sns.color_palette("viridis", len(top_features)))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_metric_name(feat) for feat in top_features['feature']])
    ax.invert_yaxis()  # Top feature at the top
    
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Discriminative Features')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width + 0.01 * max(top_features['importance']), bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(
    metrics_df: pd.DataFrame,
    metrics_subset: Optional[List[str]] = None,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix of metrics.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics
    metrics_subset : list of str, optional
        Subset of metrics to include
    method : str
        Correlation method ('pearson', 'spearman')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Select numeric columns
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    
    if metrics_subset is not None:
        numeric_cols = [col for col in numeric_cols if col in metrics_subset]
    
    # Calculate correlation matrix
    corr_matrix = metrics_df[numeric_cols].corr(method=method)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax,
                fmt='.2f')
    
    # Format labels
    formatted_labels = [_format_metric_name(label) for label in corr_matrix.index]
    ax.set_xticklabels(formatted_labels, rotation=45, ha='right')
    ax.set_yticklabels(formatted_labels, rotation=0)
    
    ax.set_title(f'Metrics Correlation Matrix ({method.capitalize()})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pca_analysis(
    metrics_df: pd.DataFrame,
    grouping_col: str = 'condition',
    n_components: int = 3,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PCA analysis of metrics.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics
    grouping_col : str
        Column for coloring points
    n_components : int
        Number of PCA components to compute
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['group_id', 'n_channels', 'recording_length', 'channel_id', 'well_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    X = metrics_df[feature_cols].dropna()
    groups = metrics_df.loc[X.index, grouping_col]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if n_components >= 3:
        # 3D plot
        ax = fig.add_subplot(121, projection='3d')
        
        # Color by group
        unique_groups = groups.unique()
        colors = sns.color_palette("husl", len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      c=[colors[i]], label=group, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.legend()
        
        # Variance explained plot
        ax2 = fig.add_subplot(122)
    else:
        # 2D plot only
        ax = fig.add_subplot(121)
        
        unique_groups = groups.unique()
        colors = sns.color_palette("husl", len(unique_groups))
        
        for i, group in enumerate(unique_groups):
            mask = groups == group
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], label=group, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend()
        
        # Variance explained plot
        ax2 = fig.add_subplot(122)
    
    # Plot explained variance
    ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
           pca.explained_variance_ratio_)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _add_statistical_annotations(ax, data: pd.DataFrame, group_col: str, value_col: str):
    """Add statistical significance annotations to plots."""
    groups = data[group_col].unique()
    
    if len(groups) < 2:
        return
    
    # Perform pairwise statistical tests
    group_data = [data[data[group_col] == group][value_col].values for group in groups]
    
    # Remove empty groups
    group_data = [g for g in group_data if len(g) > 0]
    
    if len(group_data) < 2:
        return
    
    try:
        # Use appropriate test based on number of groups
        if len(group_data) == 2:
            # Two-group comparison
            stat, p_value = stats.mannwhitneyu(group_data[0], group_data[1])
            
            # Add significance annotation
            if p_value < 0.05:
                y_max = max([np.max(g) for g in group_data if len(g) > 0])
                ax.text(0.5, 0.95, f'p = {p_value:.3f}' + ('***' if p_value < 0.001 else 
                                                          '**' if p_value < 0.01 else '*'),
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            # Multi-group comparison (Kruskal-Wallis)
            stat, p_value = stats.kruskal(*group_data)
            
            if p_value < 0.05:
                ax.text(0.5, 0.95, f'Kruskal-Wallis p = {p_value:.3f}' + 
                       ('***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'),
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    except Exception:
        pass  # Skip if statistical test fails


def _format_metric_name(metric_name: str) -> str:
    """Format metric names for display."""
    # Dictionary for common metric name mappings
    name_mapping = {
        'mean_firing_rate': 'Mean Firing Rate (Hz)',
        'cv_isi_mean': 'CV-ISI',
        'lv_isi_mean': 'LV-ISI', 
        'pearson_cc_mean': 'Pearson Correlation',
        'isi_distance': 'ISI Distance',
        'spike_distance': 'SPIKE Distance',
        'network_firing_rate': 'Network Firing Rate',
        'active_channels_count': 'Active Channels',
        'condition': 'Condition',
        'well_id': 'Well ID'
    }
    
    if metric_name in name_mapping:
        return name_mapping[metric_name]
    
    # General formatting: replace underscores and capitalize
    formatted = metric_name.replace('_', ' ').title()
    return formatted