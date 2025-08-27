"""
Configuration utilities for MEA-Flow.

This module provides default configurations and parameter sets
for different analysis types.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def get_default_well_map(n_channels: int = 64) -> Dict[int, np.ndarray]:
    """
    Get default well mapping for standard MEA plates.
    
    Parameters
    ----------
    n_channels : int
        Total number of channels
        
    Returns
    -------
    dict
        Well mapping dictionary
    """
    if n_channels <= 16:
        # Single well
        return {1: np.arange(n_channels)}
    
    elif n_channels <= 64:
        # 4-well plate (16 channels per well)
        return {
            1: np.arange(0, 16),
            2: np.arange(16, 32), 
            3: np.arange(32, 48),
            4: np.arange(48, 64)
        }
    
    elif n_channels <= 96:
        # 6-well plate (16 channels per well)
        wells = {}
        for i in range(6):
            start_ch = i * 16
            end_ch = min(start_ch + 16, n_channels)
            wells[i + 1] = np.arange(start_ch, end_ch)
        return wells
    
    else:
        # Large array - group into wells of 16 channels
        n_wells = int(np.ceil(n_channels / 16))
        wells = {}
        
        for i in range(n_wells):
            start_ch = i * 16
            end_ch = min(start_ch + 16, n_channels)
            wells[i + 1] = np.arange(start_ch, end_ch)
        
        return wells


def get_default_parameters(analysis_type: str = 'comprehensive') -> Dict[str, Any]:
    """
    Get default parameters for different analysis types.
    
    Parameters
    ----------
    analysis_type : str
        Type of analysis ('comprehensive', 'quick', 'detailed', 'manifold')
        
    Returns
    -------
    dict
        Parameter dictionary
    """
    base_params = {
        # Data preprocessing
        'min_spikes_per_channel': 10,
        'max_firing_rate': None,  # No upper limit by default
        
        # Time binning
        'time_bin_size': 1.0,  # seconds
        
        # Activity analysis
        'burst_detection': True,
        'min_burst_spikes': 5,
        'max_isi_burst': 0.1,  # seconds
        
        # Network burst detection
        'network_burst_detection': True,
        'network_burst_threshold': 1.25,
        'min_network_burst_duration': 0.05,  # seconds
        'min_electrodes_active': 0.35,  # fraction
        
        # Regularity analysis
        'min_isi_samples': 5,
        
        # Synchrony analysis
        'n_pairs_sync': 500,
        'sync_time_bin': 0.01,  # seconds
        'tau_van_rossum': 0.02,  # seconds
        
        # Manifold analysis
        'manifold_tau': 0.02,  # seconds
        'manifold_dt': 0.001,  # seconds
        'max_manifold_components': 20,
        'manifold_methods': ['PCA', 'MDS', 'Isomap', 'LLE', 'UMAP', 't-SNE'],
    }
    
    if analysis_type == 'quick':
        # Faster analysis with reduced parameters
        params = base_params.copy()
        params.update({
            'n_pairs_sync': 100,
            'network_burst_detection': False,
            'max_manifold_components': 10,
            'manifold_methods': ['PCA', 'UMAP'],
        })
        
    elif analysis_type == 'detailed':
        # More thorough analysis
        params = base_params.copy()
        params.update({
            'n_pairs_sync': 1000,
            'time_bin_size': 0.5,  # Higher resolution
            'min_isi_samples': 3,  # More sensitive
            'manifold_methods': ['PCA', 'MDS', 'Isomap', 'LLE', 'UMAP', 't-SNE', 'SpectralEmbedding'],
        })
        
    elif analysis_type == 'manifold':
        # Focus on manifold analysis
        params = base_params.copy()
        params.update({
            'burst_detection': False,
            'network_burst_detection': False,
            'n_pairs_sync': 200,
            'max_manifold_components': 50,
            'manifold_dt': 0.0005,  # Higher temporal resolution
            'manifold_methods': ['PCA', 'MDS', 'Isomap', 'LLE', 'UMAP', 't-SNE'],
        })
        
    else:  # comprehensive
        params = base_params.copy()
    
    return params


def get_electrode_layout(plate_type: str = 'MEA60') -> Dict[int, Dict[str, float]]:
    """
    Get electrode layout coordinates for different MEA plate types.
    
    Parameters
    ----------
    plate_type : str
        Type of MEA plate ('MEA60', 'MEA256', 'custom')
        
    Returns
    -------
    dict
        Dictionary mapping electrode IDs to {'x': float, 'y': float} coordinates
    """
    if plate_type == 'MEA60':
        # Standard 8x8 layout with 60 active electrodes
        layout = {}
        electrode_id = 0
        
        # 8x8 grid with some positions unused (corners typically)
        for row in range(8):
            for col in range(8):
                # Skip corner electrodes (common in MEA60)
                if (row, col) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                    continue
                
                layout[electrode_id] = {
                    'x': col * 200,  # 200 μm spacing
                    'y': row * 200
                }
                electrode_id += 1
        
    elif plate_type == 'MEA256':
        # High-density 16x16 layout
        layout = {}
        electrode_id = 0
        
        for row in range(16):
            for col in range(16):
                layout[electrode_id] = {
                    'x': col * 42,  # 42 μm spacing for HD-MEA
                    'y': row * 42
                }
                electrode_id += 1
                
    else:  # custom or default 4x4 per well
        # Simple 4x4 layout per well
        layout = {}
        
        for well in range(4):
            well_offset_x = (well % 2) * 800  # 800 μm between wells
            well_offset_y = (well // 2) * 800
            
            for row in range(4):
                for col in range(4):
                    electrode_id = well * 16 + row * 4 + col
                    layout[electrode_id] = {
                        'x': well_offset_x + col * 200,
                        'y': well_offset_y + row * 200
                    }
    
    return layout


def get_analysis_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined analysis presets for common use cases.
    
    Returns
    -------
    dict
        Dictionary of analysis presets
    """
    presets = {
        'developmental': {
            'description': 'Analysis optimized for developmental studies',
            'parameters': {
                'min_spikes_per_channel': 5,  # Lower threshold for developing networks
                'network_burst_threshold': 1.0,  # More sensitive to weak network activity
                'min_electrodes_active': 0.2,
                'manifold_methods': ['PCA', 'UMAP'],
                'time_bin_size': 2.0  # Longer windows for sparse activity
            }
        },
        
        'pharmacology': {
            'description': 'Analysis for pharmacological interventions',
            'parameters': {
                'burst_detection': True,
                'network_burst_detection': True,
                'n_pairs_sync': 1000,  # Detailed synchrony analysis
                'manifold_methods': ['PCA', 'MDS', 'UMAP'],
                'time_bin_size': 0.5  # High temporal resolution
            }
        },
        
        'disease_model': {
            'description': 'Analysis for disease/pathology models',
            'parameters': {
                'min_spikes_per_channel': 3,  # Account for potentially reduced activity
                'max_firing_rate': 200,  # Filter out extremely hyperactive channels
                'network_burst_threshold': 1.5,
                'manifold_methods': ['PCA', 'Isomap', 'UMAP', 't-SNE'],
            }
        },
        
        'high_throughput': {
            'description': 'Fast analysis for high-throughput screening',
            'parameters': {
                'burst_detection': False,
                'network_burst_detection': True,
                'n_pairs_sync': 50,
                'manifold_methods': ['PCA'],
                'time_bin_size': 2.0
            }
        },
        
        'detailed_characterization': {
            'description': 'Comprehensive analysis for detailed characterization',
            'parameters': {
                'min_isi_samples': 3,
                'n_pairs_sync': 2000,
                'max_manifold_components': 50,
                'manifold_methods': ['PCA', 'MDS', 'Isomap', 'LLE', 'UMAP', 't-SNE'],
                'time_bin_size': 0.25,
                'manifold_dt': 0.0005
            }
        }
    }
    
    return presets


def validate_parameters(params: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate analysis parameters and return warnings/errors.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary to validate
        
    Returns
    -------
    dict
        Dictionary with 'warnings' and 'errors' lists
    """
    warnings = []
    errors = []
    
    # Check required parameters
    required_params = ['time_bin_size', 'min_spikes_per_channel']
    for param in required_params:
        if param not in params:
            errors.append(f"Missing required parameter: {param}")
    
    # Validate parameter ranges
    if 'time_bin_size' in params:
        if params['time_bin_size'] <= 0:
            errors.append("time_bin_size must be positive")
        elif params['time_bin_size'] > 10:
            warnings.append("time_bin_size > 10s may be too large for meaningful analysis")
    
    if 'min_spikes_per_channel' in params:
        if params['min_spikes_per_channel'] < 1:
            errors.append("min_spikes_per_channel must be at least 1")
        elif params['min_spikes_per_channel'] > 100:
            warnings.append("min_spikes_per_channel > 100 may be too restrictive")
    
    if 'n_pairs_sync' in params:
        if params['n_pairs_sync'] < 10:
            warnings.append("n_pairs_sync < 10 may not provide reliable synchrony estimates")
        elif params['n_pairs_sync'] > 5000:
            warnings.append("n_pairs_sync > 5000 may be computationally expensive")
    
    if 'max_manifold_components' in params:
        if params['max_manifold_components'] < 2:
            errors.append("max_manifold_components must be at least 2")
        elif params['max_manifold_components'] > 100:
            warnings.append("max_manifold_components > 100 may be computationally expensive")
    
    # Check parameter consistency
    if ('min_electrodes_active' in params and 
        params['min_electrodes_active'] > 1):
        errors.append("min_electrodes_active must be <= 1 (fraction)")
    
    return {'warnings': warnings, 'errors': errors}