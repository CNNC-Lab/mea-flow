"""
Analysis module for MEA-Flow.

This module provides comprehensive metrics and statistical analysis tools
for MEA data including activity, regularity, and synchrony measures.
"""

from .metrics import MEAMetrics, AnalysisConfig
from .activity import compute_activity_metrics, firing_rate, burst_detection
from .regularity import compute_regularity_metrics, cv_isi, entropy_isi  
from .synchrony import compute_synchrony_metrics, pairwise_correlations, spike_distance_measures
from .burst_analysis import network_burst_analysis, burst_statistics
from .discriminant import (
    identify_discriminative_features, 
    compare_discriminant_methods,
    get_available_methods,
    DiscriminantConfig, 
    DiscriminantResults, 
    DiscriminantMethod
)
from .feature_space import (
    load_feature_data,
    apply_feature_embedding,
    apply_multiple_embeddings,
    plot_feature_embeddings,
    analyze_feature_space,
    get_feature_importance
)

__all__ = [
    "MEAMetrics",
    "AnalysisConfig",
    "compute_activity_metrics",
    "compute_regularity_metrics", 
    "compute_synchrony_metrics",
    "firing_rate",
    "burst_detection",
    "cv_isi",
    "entropy_isi",
    "pairwise_correlations",
    "spike_distance_measures",
    "network_burst_analysis",
    "burst_statistics",
    "identify_discriminative_features",
    "compare_discriminant_methods",
    "get_available_methods",
    "DiscriminantConfig",
    "DiscriminantResults",
    "DiscriminantMethod",
    "load_feature_data",
    "apply_feature_embedding",
    "apply_multiple_embeddings",
    "plot_feature_embeddings",
    "analyze_feature_space",
    "get_feature_importance"
]