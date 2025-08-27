"""
Manifold learning and dimensionality reduction visualizations.

This module provides functions for visualizing the results of manifold
learning and dimensionality reduction applied to MEA population dynamics.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
import warnings


def plot_embedding(
    embedding_data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method_name: str = 'Embedding',
    time_vector: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize low-dimensional embedding of population dynamics.
    
    Parameters
    ----------
    embedding_data : np.ndarray
        Low-dimensional embedding (N_timepoints x N_dimensions)
    labels : np.ndarray, optional
        Labels for coloring points (e.g., time, condition)
    method_name : str
        Name of the embedding method for title
    time_vector : np.ndarray, optional
        Time vector for temporal coloring
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    n_dims = embedding_data.shape[1]
    
    if n_dims < 2:
        raise ValueError("Embedding data must have at least 2 dimensions")
    
    # Use time as labels if none provided
    if labels is None and time_vector is not None:
        labels = time_vector
    elif labels is None:
        labels = np.arange(len(embedding_data))
    
    # Create figure based on dimensionality
    if n_dims == 2:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        scatter = ax.scatter(embedding_data[:, 0], embedding_data[:, 1], 
                           c=labels, cmap='viridis', alpha=0.7, s=20)
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        
        if time_vector is not None:
            plt.colorbar(scatter, ax=ax, label='Time (s)')
        
    elif n_dims >= 3:
        fig = plt.figure(figsize=figsize)
        
        # 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(embedding_data[:, 0], embedding_data[:, 1], embedding_data[:, 2],
                            c=labels, cmap='viridis', alpha=0.7, s=20)
        ax1.set_xlabel(f'{method_name} 1')
        ax1.set_ylabel(f'{method_name} 2')
        ax1.set_zlabel(f'{method_name} 3')
        
        # 2D projections
        ax2 = fig.add_subplot(222)
        ax2.scatter(embedding_data[:, 0], embedding_data[:, 1], c=labels, 
                   cmap='viridis', alpha=0.7, s=20)
        ax2.set_xlabel(f'{method_name} 1')
        ax2.set_ylabel(f'{method_name} 2')
        
        ax3 = fig.add_subplot(223)
        ax3.scatter(embedding_data[:, 0], embedding_data[:, 2], c=labels, 
                   cmap='viridis', alpha=0.7, s=20)
        ax3.set_xlabel(f'{method_name} 1')
        ax3.set_ylabel(f'{method_name} 3')
        
        ax4 = fig.add_subplot(224)
        ax4.scatter(embedding_data[:, 1], embedding_data[:, 2], c=labels, 
                   cmap='viridis', alpha=0.7, s=20)
        ax4.set_xlabel(f'{method_name} 2')
        ax4.set_ylabel(f'{method_name} 3')
        
        # Add colorbar
        if time_vector is not None:
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Time (s)')
    
    fig.suptitle(f'{method_name} Visualization')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dimensionality_analysis(
    explained_variance: np.ndarray,
    method_name: str = 'PCA',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot dimensionality analysis showing explained variance.
    
    Parameters
    ----------
    explained_variance : np.ndarray
        Explained variance ratio for each component
    method_name : str
        Name of the dimensionality reduction method
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n_components = len(explained_variance)
    components = np.arange(1, n_components + 1)
    
    # Individual explained variance
    ax1.bar(components, explained_variance, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Component Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)
    ax2.plot(components, cumulative_variance, 'o-', color='red', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% Variance')
    ax2.axhline(y=0.90, color='gray', linestyle=':', alpha=0.7, label='90% Variance')
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle(f'{method_name} Dimensionality Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_manifold_comparison(
    embeddings: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare multiple manifold learning methods side by side.
    
    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embedding arrays
    labels : np.ndarray, optional
        Labels for coloring points
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    n_methods = len(embeddings)
    
    if n_methods == 0:
        warnings.warn("No embeddings provided")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No embeddings', transform=ax.transAxes, ha='center', va='center')
        return fig
    
    # Determine subplot layout
    n_cols = min(3, n_methods)
    n_rows = int(np.ceil(n_methods / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten() if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each embedding
    for i, (method_name, embedding_data) in enumerate(embeddings.items()):
        ax = axes[i]
        
        if embedding_data.shape[1] >= 2:
            scatter = ax.scatter(embedding_data[:, 0], embedding_data[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7, s=20)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(method_name)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{method_name}\nInsufficient dimensions', 
                   transform=ax.transAxes, ha='center', va='center')
    
    # Remove extra subplots
    for i in range(n_methods, len(axes)):
        fig.delaxes(axes[i])
    
    # Add shared colorbar if labels provided
    if labels is not None and n_methods > 0:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('Labels')
    
    fig.suptitle('Manifold Learning Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trajectory(
    embedding_data: np.ndarray,
    time_vector: np.ndarray,
    method_name: str = 'Trajectory',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trajectory through embedding space over time.
    
    Parameters
    ----------
    embedding_data : np.ndarray
        Low-dimensional embedding data
    time_vector : np.ndarray
        Time points corresponding to each embedding point
    method_name : str
        Name for the plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    n_dims = embedding_data.shape[1]
    
    if n_dims < 2:
        raise ValueError("Need at least 2 dimensions for trajectory plot")
    
    fig = plt.figure(figsize=figsize)
    
    if n_dims == 2:
        ax = fig.add_subplot(111)
        
        # Plot trajectory as connected line
        ax.plot(embedding_data[:, 0], embedding_data[:, 1], 'b-', alpha=0.6, linewidth=1)
        
        # Color points by time
        scatter = ax.scatter(embedding_data[:, 0], embedding_data[:, 1], 
                           c=time_vector, cmap='viridis', s=30, alpha=0.8)
        
        # Mark start and end points
        ax.scatter(embedding_data[0, 0], embedding_data[0, 1], 
                  c='green', s=100, marker='o', label='Start', edgecolors='black')
        ax.scatter(embedding_data[-1, 0], embedding_data[-1, 1], 
                  c='red', s=100, marker='s', label='End', edgecolors='black')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        
        plt.colorbar(scatter, label='Time (s)')
        
    else:  # 3D trajectory
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(embedding_data[:, 0], embedding_data[:, 1], embedding_data[:, 2], 
               'b-', alpha=0.6, linewidth=1)
        
        # Color points by time
        scatter = ax.scatter(embedding_data[:, 0], embedding_data[:, 1], embedding_data[:, 2],
                           c=time_vector, cmap='viridis', s=30, alpha=0.8)
        
        # Mark start and end
        ax.scatter(embedding_data[0, 0], embedding_data[0, 1], embedding_data[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(embedding_data[-1, 0], embedding_data[-1, 1], embedding_data[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
        
        plt.colorbar(scatter, label='Time (s)', shrink=0.8)
    
    plt.title(f'{method_name} - Population Dynamics Trajectory')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_embedding_quality(
    reconstruction_errors: Dict[str, np.ndarray],
    dimensions: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reconstruction error vs embedding dimensions for different methods.
    
    Parameters
    ----------
    reconstruction_errors : dict
        Dictionary mapping method names to reconstruction error arrays
    dimensions : np.ndarray
        Array of embedding dimensions tested
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(reconstruction_errors))
    
    for i, (method_name, errors) in enumerate(reconstruction_errors.items()):
        ax.plot(dimensions, errors, 'o-', color=colors[i], 
               label=method_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Dimensions')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Embedding Quality vs Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig