"""
Manifold learning and population geometry analysis module for MEA-Flow.

This module provides advanced dimensionality reduction techniques and 
state space analysis for neural population dynamics.
"""

from .analysis import ManifoldAnalysis, ManifoldConfig
from .embedding import embed_population_dynamics, apply_dimensionality_reduction
from .evaluation import evaluate_embedding, reconstruction_error, effective_dimensionality
from .comparison import compare_manifolds, cross_condition_analysis

__all__ = [
    "ManifoldAnalysis",
    "ManifoldConfig",
    "embed_population_dynamics", 
    "apply_dimensionality_reduction",
    "evaluate_embedding",
    "reconstruction_error",
    "effective_dimensionality",
    "compare_manifolds",
    "cross_condition_analysis"
]