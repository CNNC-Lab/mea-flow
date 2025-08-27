"""
MEA-Flow: A Python package for analyzing multi-electrode array data with focus 
on neural population dynamics through manifold learning and state space analysis.
"""

__version__ = "0.1.0"
__author__ = "MEA-Flow Team"
__email__ = "contact@mea-flow.org"

# Import main classes and functions for easy access
from .data import SpikeList, load_data, load_axion_spk, load_matlab_file
from .analysis import MEAMetrics, compute_activity_metrics, compute_synchrony_metrics
from .manifold import ManifoldAnalysis, embed_population_dynamics
from .visualization import MEAPlotter, plot_raster, plot_metrics_comparison

__all__ = [
    # Data loading and processing
    "SpikeList",
    "load_data", 
    "load_axion_spk",
    "load_matlab_file",
    
    # Analysis modules
    "MEAMetrics",
    "compute_activity_metrics",
    "compute_synchrony_metrics",
    
    # Manifold analysis
    "ManifoldAnalysis", 
    "embed_population_dynamics",
    
    # Visualization
    "MEAPlotter",
    "plot_raster",
    "plot_metrics_comparison",
]

# Package metadata
__doc__ = """
MEA-Flow: Neural Population Dynamics Analysis

A comprehensive Python package for analyzing multi-electrode array (MEA) data with 
specialized focus on neural population dynamics, manifold learning, and comparative 
analysis across experimental conditions.

Key Features:
- MEA-specific data loading (Axion .spk, .mat files, pandas DataFrames)
- Comprehensive metrics for activity, regularity, and synchrony analysis
- Advanced manifold learning and dimensionality reduction techniques  
- State space analysis of neural population dynamics
- Publication-ready visualizations
- Comparative analysis across experimental conditions
- Well-based and time-resolved analysis capabilities

Main Modules:
- mea_flow.data: Data loading, preprocessing, and SpikeList management
- mea_flow.analysis: Comprehensive metrics calculation and statistical analysis
- mea_flow.manifold: Population geometry analysis and manifold learning
- mea_flow.visualization: MEA-specific plotting and visualization routines
- mea_flow.utils: Utility functions and helper tools

For detailed documentation and examples, see:
https://mea-flow.readthedocs.io
"""