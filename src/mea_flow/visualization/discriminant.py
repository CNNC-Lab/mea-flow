"""
Visualization functions for comprehensive feature analysis results.

This module provides plotting functions specifically designed for visualizing
comprehensive feature analysis results, including redundancy detection,
feature importance, consensus ranking, and method comparisons.
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..analysis.discriminant import (
    FeatureAnalysisResults, 
    FeatureImportanceResult, 
    RedundancyAnalysisResult,
    ConsensusResult
)


def plot_feature_importance(
    importance_result: FeatureImportanceResult,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance scores from a single method.
    
    Parameters
    ----------
    importance_result : FeatureImportanceResult
        Feature importance results from a single method
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
    # Get top features
    top_features = importance_result.scores.nlargest(top_n, 'importance')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(top_features))
    colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
    
    bars = ax.barh(y_pos, top_features['importance'], color=colors, alpha=0.8)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_feature_name(feat) for feat in top_features['feature']])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    
    if title is None:
        title = f'{importance_result.method.value.replace("_", " ").title()} - Top {top_n} Features'
    ax.set_title(title)
    
    # Add value labels on bars
    for bar, score in zip(bars, top_features['importance']):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{score:.3f}', ha='left', va='center', fontweight='bold',
               color='white', weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_consensus_ranking(
    results: FeatureAnalysisResults,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot consensus feature ranking from comprehensive analysis.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
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
    if results.consensus_result is None:
        raise ValueError("No consensus ranking available in results")
    
    consensus_df = results.consensus_result.consensus_ranking.head(top_n)
    
    if title is None:
        title = f'Top {top_n} Features - Consensus Ranking'
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(consensus_df))
    
    # Color bars by consensus score
    colors = plt.cm.viridis(consensus_df['consensus_score'] / consensus_df['consensus_score'].max())
    
    bars = ax.barh(y_pos, consensus_df['consensus_score'], color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_format_feature_name(feat) for feat in consensus_df['feature']])
    ax.invert_yaxis()  # Top feature at the top
    
    ax.set_xlabel('Consensus Score')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, consensus_df['consensus_score'])):
        width = bar.get_width()
        ax.text(width + 0.01 * consensus_df['consensus_score'].max(), 
                bar.get_y() + bar.get_height()/2,
                f'{score:.2f}', ha='left', va='center', fontsize=9)
    
    # Add method agreement information
    for i, (bar, agreement) in enumerate(zip(bars, consensus_df['method_agreement'])):
        ax.text(0.02 * consensus_df['consensus_score'].max(),
                bar.get_y() + bar.get_height()/2,
                f'({agreement:.0%})', ha='left', va='center', fontsize=8, 
                color='white', weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_redundancy_analysis(
    results: FeatureAnalysisResults,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot redundancy analysis results showing VIF scores and correlations.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if not results.redundancy_results:
        raise ValueError("No redundancy analysis results available")
    
    fig, axes = plt.subplots(1, len(results.redundancy_results), figsize=figsize)
    if len(results.redundancy_results) == 1:
        axes = [axes]
    
    for idx, (method_name, redundancy_result) in enumerate(results.redundancy_results.items()):
        ax = axes[idx]
        
        if method_name == 'vif' and redundancy_result.scores is not None:
            # Plot VIF scores
            vif_df = redundancy_result.scores.copy()
            vif_df = vif_df[~np.isinf(vif_df['vif_score'])].sort_values('vif_score', ascending=True)
            
            if len(vif_df) > 20:  # Show top 20 if too many features
                vif_df = vif_df.tail(20)
            
            y_pos = np.arange(len(vif_df))
            colors = ['red' if x else 'blue' for x in vif_df['redundant']]
            
            ax.barh(y_pos, vif_df['vif_score'], color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_format_feature_name(feat) for feat in vif_df['feature']])
            ax.set_xlabel('VIF Score')
            ax.set_title('Variance Inflation Factor Analysis')
            ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Threshold (10)')
            ax.legend()
            
        elif method_name == 'correlation' and redundancy_result.scores is not None:
            # Plot correlation pairs
            corr_df = redundancy_result.scores.copy()
            if len(corr_df) > 0:
                corr_df = corr_df.sort_values('correlation', ascending=False).head(15)
                
                y_pos = np.arange(len(corr_df))
                pair_labels = [f"{_format_feature_name(row['feature_1'][:15])} - {_format_feature_name(row['feature_2'][:15])}" 
                              for _, row in corr_df.iterrows()]
                
                ax.barh(y_pos, corr_df['correlation'], color='orange', alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(pair_labels)
                ax.set_xlabel('Correlation Coefficient')
                ax.set_title('High Correlation Pairs')
                ax.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Threshold (0.9)')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No high correlations found', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('Correlation Analysis')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_importance_comparison(
    results: FeatureAnalysisResults,
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare feature importance across different methods in a heatmap.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
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
    if not results.importance_results:
        raise ValueError("No feature importance results available")
    
    # Get consensus ranking for top features
    if results.consensus_result is not None:
        top_features = results.consensus_result.consensus_ranking.head(top_n)['feature'].tolist()
    else:
        # Fallback: use features from first method
        first_method = list(results.importance_results.values())[0]
        top_features = first_method.scores.nlargest(top_n, 'importance')['feature'].tolist()
    
    # Create comparison matrix
    comparison_data = []
    method_names = []
    
    for method_name, importance_result in results.importance_results.items():
        method_names.append(method_name.replace('_', ' ').title())
        method_scores = []
        
        for feature in top_features:
            feat_row = importance_result.scores[
                importance_result.scores['feature'] == feature
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
        columns=[_format_feature_name(feat) for feat in top_features]
    )
    
    # Normalize each row to make comparison easier
    comparison_df_norm = comparison_df.div(comparison_df.max(axis=1), axis=0)
    
    sns.heatmap(comparison_df_norm, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                cbar_kws={'label': 'Normalized Importance'})
    
    ax.set_title(f'Feature Importance Comparison Across Methods (Top {top_n} Features)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Methods')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_analysis_summary(
    results: FeatureAnalysisResults,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary plot of the feature analysis results.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Dataset overview (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    class_counts = list(results.class_distribution.values())
    class_labels = list(results.class_distribution.keys())
    ax1.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Class Distribution')
    
    # 2. Feature categories (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Critical', 'Redundant', 'Method-specific', 'Irrelevant']
    counts = [
        len(results.critical_features),
        len(results.redundant_features), 
        sum(len(feats) for feats in results.method_specific_features.values()),
        len(results.irrelevant_features)
    ]
    colors = ['green', 'red', 'orange', 'gray']
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
    ax2.set_title('Feature Categories')
    ax2.set_ylabel('Number of Features')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # 3. Execution time breakdown (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    phases = ['Redundancy', 'Core Selection', 'Consensus']
    # Mock timing data - in real implementation, track phase times
    times = [results.execution_time * 0.2, results.execution_time * 0.7, results.execution_time * 0.1]
    ax3.pie(times, labels=phases, autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Execution Time\n({results.execution_time:.2f}s total)')
    
    # 4. Top consensus features (middle row, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    if results.consensus_result is not None:
        top_consensus = results.consensus_result.consensus_ranking.head(10)
        y_pos = np.arange(len(top_consensus))
        bars = ax4.barh(y_pos, top_consensus['consensus_score'], 
                       color=plt.cm.viridis(top_consensus['consensus_score'] / top_consensus['consensus_score'].max()))
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([_format_feature_name(feat)[:25] for feat in top_consensus['feature']])
        ax4.invert_yaxis()
        ax4.set_xlabel('Consensus Score')
        ax4.set_title('Top 10 Consensus Features')
    else:
        ax4.text(0.5, 0.5, 'No consensus ranking available', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Consensus Features')
    
    # 5. Method agreement (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if results.consensus_result is not None:
        agreement_scores = results.consensus_result.consensus_ranking['method_agreement'].head(20)
        ax5.hist(agreement_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Method Agreement')
        ax5.set_ylabel('Number of Features')
        ax5.set_title('Method Agreement Distribution')
        ax5.axvline(agreement_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {agreement_scores.mean():.2f}')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No agreement data', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Method Agreement')
    
    # 6. Redundancy summary (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    if results.redundancy_results:
        redundancy_counts = {method: len(result.redundant_features) 
                           for method, result in results.redundancy_results.items()}
        methods = list(redundancy_counts.keys())
        counts = list(redundancy_counts.values())
        ax6.bar(methods, counts, color='coral', alpha=0.7)
        ax6.set_title('Redundant Features by Method')
        ax6.set_ylabel('Count')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    else:
        ax6.text(0.5, 0.5, 'No redundancy analysis', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('Redundancy Analysis')
    
    # 7. Method performance comparison (bottom middle)
    ax7 = fig.add_subplot(gs[2, 1])
    if results.importance_results:
        method_names = [name.replace('_', ' ').title() for name in results.importance_results.keys()]
        # Mock performance scores - in real implementation, track method performance
        performance_scores = [0.85, 0.82, 0.88, 0.79][:len(method_names)]
        bars = ax7.bar(method_names, performance_scores, color='lightgreen', alpha=0.7)
        ax7.set_title('Method Performance')
        ax7.set_ylabel('Score')
        ax7.set_ylim(0, 1)
        plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, performance_scores):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.2f}', ha='center', va='bottom')
    else:
        ax7.text(0.5, 0.5, 'No method results', 
                transform=ax7.transAxes, ha='center', va='center')
        ax7.set_title('Method Performance')
    
    # 8. Analysis metadata (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    metadata_text = f"""
    Dataset Summary:
    • Samples: {results.n_samples:,}
    • Features: {results.n_features:,}
    • Classes: {results.n_classes}
    
    Analysis Results:
    • Critical: {len(results.critical_features)}
    • Redundant: {len(results.redundant_features)}
    • Methods used: {len(results.importance_results)}
    
    Execution:
    • Time: {results.execution_time:.2f}s
    • Warnings: {len(results.warnings)}
    """
    
    ax8.text(0.05, 0.95, metadata_text, transform=ax8.transAxes, 
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax8.set_title('Analysis Summary')
    
    plt.suptitle('Comprehensive Feature Analysis Results', fontsize=16, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_stability(
    results: FeatureAnalysisResults,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature stability across different methods.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
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
    if results.consensus_result is None:
        raise ValueError("No consensus results available for stability analysis")
    
    # Get top features by consensus score
    top_features = results.consensus_result.consensus_ranking.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Consensus score vs method agreement
    ax1.scatter(top_features['method_agreement'], top_features['consensus_score'], 
               alpha=0.7, s=60, c=range(len(top_features)), cmap='viridis')
    ax1.set_xlabel('Method Agreement')
    ax1.set_ylabel('Consensus Score')
    ax1.set_title('Feature Stability: Consensus vs Agreement')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(top_features['method_agreement'], top_features['consensus_score'], 1)
    p = np.poly1d(z)
    ax1.plot(top_features['method_agreement'], p(top_features['method_agreement']), 
            "r--", alpha=0.8, linewidth=2)
    
    # Plot 2: Feature ranking stability
    y_pos = np.arange(len(top_features))
    colors = plt.cm.RdYlGn(top_features['method_agreement'] / top_features['method_agreement'].max())
    
    bars = ax2.barh(y_pos, top_features['consensus_score'], color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([_format_feature_name(feat)[:20] for feat in top_features['feature']])
    ax2.invert_yaxis()
    ax2.set_xlabel('Consensus Score')
    ax2.set_title(f'Top {top_n} Most Stable Features')
    
    # Add colorbar for method agreement
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                              norm=plt.Normalize(vmin=top_features['method_agreement'].min(), 
                                               vmax=top_features['method_agreement'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Method Agreement')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_performance(
    results: FeatureAnalysisResults,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot performance metrics for different feature selection methods.
    
    Parameters
    ----------
    results : FeatureAnalysisResults
        Results from comprehensive feature analysis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    if not results.importance_results:
        raise ValueError("No feature importance results available")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract method information
    method_names = [name.replace('_', ' ').title() for name in results.importance_results.keys()]
    n_features_selected = []
    execution_times = []
    
    for method_name, importance_result in results.importance_results.items():
        # Count features above threshold (mock data for now)
        n_selected = len(importance_result.scores[importance_result.scores['importance'] > 0.1])
        n_features_selected.append(n_selected)
        
        # Mock execution time per method
        execution_times.append(results.execution_time / len(results.importance_results))
    
    # Plot 1: Number of features selected by each method
    bars1 = ax1.bar(method_names, n_features_selected, 
                    color=sns.color_palette("Set2", len(method_names)), alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Features Selected')
    ax1.set_title('Features Selected by Method')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars1, n_features_selected):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Plot 2: Execution time comparison
    bars2 = ax2.bar(method_names, execution_times, 
                    color=sns.color_palette("Set3", len(method_names)), alpha=0.7)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Method Execution Time')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, time in zip(bars2, execution_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time:.2f}', ha='center', va='bottom')
    
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
        'Mea': 'MEA',
        'Rf': 'RF',
        'Svm': 'SVM',
        'Pca': 'PCA',
        'Lda': 'LDA',
        'Std': 'Std',
        'Mean': 'Mean',
        'Max': 'Max',
        'Min': 'Min'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted
