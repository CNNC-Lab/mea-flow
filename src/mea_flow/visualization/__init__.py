"""
Visualization module for MEA-Flow.

This module provides MEA-specific plotting and visualization routines
for raw data, metrics, and manifold analysis results.
"""

from .plotter import MEAPlotter
from .raster import plot_raster, plot_animated_raster, plot_well_activity
from .wells import plot_animated_electrode_map
from .metrics import plot_metrics_comparison
from .manifold import plot_embedding, plot_dimensionality_analysis
from .wells import plot_electrode_map, plot_well_grid
from .discriminant import (
    plot_feature_importance,
    plot_consensus_ranking,
    plot_redundancy_analysis,
    plot_method_importance_comparison,
    plot_analysis_summary,
    plot_feature_stability,
    plot_method_performance
)

__all__ = [
    "MEAPlotter",
    "plot_raster",
    "plot_animated_raster", 
    "plot_animated_electrode_map",
    "plot_well_activity",
    "plot_metrics_comparison",
    "plot_feature_importance",
    "plot_consensus_ranking",
    "plot_redundancy_analysis",
    "plot_method_importance_comparison",
    "plot_analysis_summary",
    "plot_feature_stability",
    "plot_method_performance",
    "plot_embedding",
    "plot_dimensionality_analysis",
    "plot_electrode_map",
    "plot_well_grid"
]