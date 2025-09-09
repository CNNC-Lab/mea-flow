"""
Visualization functions for discriminant analysis results.

This module provides plotting functions specifically designed for visualizing
discriminant analysis results, including feature importance, model comparison,
and classification performance.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..analysis.discriminant import DiscriminantResults, DiscriminantMethod


def plot_feature_importance(
    results: Union[DiscriminantResults, pd.DataFrame],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance from discriminant analysis results.
    
    Parameters
    ----------
    results : DiscriminantResults or pd.DataFrame
        Either DiscriminantResults object or DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    title : str, optional
        Custom title for the plot
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Extract importance DataFrame
    if isinstance(results, DiscriminantResults):
        importance_df = results.feature_importance
        if title is None:
            title = f'Top {top_n} Discriminative Features'
    else:
        importance_df = results
        if title is None:
            title = f'Top {top_n} Feature Importance Scores'
    
    # Validate DataFrame structure
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        raise ValueError("DataFrame must have 'feature' and 'importance' columns")
    
    # Sort by importance and select top N
    top_features = importance_df.nlargest(top_n, 'importance')
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features['importance'], 
                   color=sns.color_palette("viridis", len(top_features)))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_feature_name(feat) for feat in top_features['feature']])
    ax.invert_yaxis()  # Top feature at the top
    
    ax.set_xlabel('Feature Importance')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax.text(width + 0.01 * max(top_features['importance']), 
                bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_comparison(
    method_results: Dict[str, DiscriminantResults],
    metric: str = 'cv_mean_accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare performance of different discriminant analysis methods.
    
    Parameters
    ----------
    method_results : Dict[str, DiscriminantResults]
        Results from different methods
    metric : str
        Performance metric to compare ('cv_mean_accuracy', 'train_accuracy')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    # Extract performance data
    methods = []
    scores = []
    errors = []
    
    for method_name, result in method_results.items():
        if metric in result.model_performance:
            methods.append(method_name.replace('_', ' ').title())
            scores.append(result.model_performance[metric])
            
            # Add error bars if standard deviation is available
            error_metric = metric.replace('mean', 'std')
            if error_metric in result.model_performance:
                errors.append(result.model_performance[error_metric])
            else:
                errors.append(0)
    
    if not methods:
        raise ValueError(f"No results found for metric '{metric}'")
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, scores, yerr=errors if any(errors) else None,
                  capsize=5, color=sns.color_palette("Set2", len(methods)))
    
    ax.set_xlabel('Method')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Discriminant Analysis Method Comparison ({metric.replace("_", " ").title()})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    results: DiscriminantResults,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix from discriminant analysis results.
    
    Parameters
    ----------
    results : DiscriminantResults
        Results containing confusion matrix
    class_names : List[str], optional
        Names for the classes
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if results.confusion_matrix is None:
        raise ValueError("No confusion matrix available in results")
    
    cm = results.confusion_matrix
    
    # Generate default class names if not provided
    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance_comparison(
    method_results: Dict[str, DiscriminantResults],
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare feature importance across different methods.
    
    Parameters
    ----------
    method_results : Dict[str, DiscriminantResults]
        Results from different methods
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
    # Collect all unique features and their importance across methods
    all_features = set()
    for result in method_results.values():
        all_features.update(result.feature_importance['feature'].tolist())
    
    # Get top features across all methods
    feature_scores = {}
    for feature in all_features:
        scores = []
        for result in method_results.values():
            feat_row = result.feature_importance[
                result.feature_importance['feature'] == feature
            ]
            if not feat_row.empty:
                scores.append(feat_row['importance'].iloc[0])
            else:
                scores.append(0)
        feature_scores[feature] = np.mean(scores)
    
    # Select top features by average importance
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_feature_names = [feat[0] for feat in top_features]
    
    # Create comparison matrix
    comparison_data = []
    method_names = []
    
    for method_name, result in method_results.items():
        method_names.append(method_name.replace('_', ' ').title())
        method_scores = []
        
        for feature in top_feature_names:
            feat_row = result.feature_importance[
                result.feature_importance['feature'] == feature
            ]
            if not feat_row.empty:
                method_scores.append(feat_row['importance'].iloc[0])
            else:
                method_scores.append(0)
        
        comparison_data.append(method_scores)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    comparison_df = pd.DataFrame(
        comparison_data,
        index=method_names,
        columns=[_format_feature_name(feat) for feat in top_feature_names]
    )
    
    sns.heatmap(comparison_df, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    
    ax.set_title(f'Feature Importance Comparison Across Methods (Top {top_n} Features)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Methods')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def _format_feature_name(feature_name: str) -> str:
    """Format feature names for better display."""
    # Replace underscores with spaces and capitalize
    formatted = feature_name.replace('_', ' ').title()
    
    # Handle common abbreviations
    replacements = {
        'Cv': 'CV',
        'Isi': 'ISI',
        'Lv': 'LV',
        'Std': 'Std',
        'Mean': 'Mean',
        'Max': 'Max',
        'Min': 'Min'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted
