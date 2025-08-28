"""
Data loading functions for various MEA data formats.

This module provides functions to load MEA data from different sources including
Axion .spk files, MATLAB .mat files, and pandas DataFrames.
"""

import os
import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Dict, List, Tuple, Any
from pathlib import Path
import h5py

try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. MATLAB file loading will be disabled.")

from .spike_list import SpikeList


def load_data(
    file_path: Union[str, Path],
    data_format: Optional[str] = None,
    **kwargs
) -> SpikeList:
    """
    Load MEA data from various file formats.
    
    This function automatically detects the file format based on extension
    or uses the provided format specification.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the data file
    data_format : str, optional
        Explicit format specification ('spk', 'mat', 'csv', 'h5')
        If None, format is inferred from file extension
    **kwargs
        Additional arguments passed to specific loading functions
        
    Returns
    -------
    SpikeList
        Loaded spike data as SpikeList object
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    ValueError
        If the file format is not supported or cannot be determined
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine format
    if data_format is None:
        data_format = _infer_format(file_path)
    
    data_format = data_format.lower()
    
    # Route to appropriate loader
    if data_format == 'spk':
        return load_axion_spk(file_path, **kwargs)
    elif data_format == 'mat':
        return load_matlab_file(file_path, **kwargs) 
    elif data_format in ['csv', 'txt']:
        return load_csv_file(file_path, **kwargs)
    elif data_format in ['h5', 'hdf5']:
        return load_hdf5_file(file_path, **kwargs)
    elif data_format == 'dataframe':
        # Assume kwargs contains 'dataframe' key
        return load_from_dataframe(**kwargs)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")


def _infer_format(file_path: Path) -> str:
    """Infer data format from file extension."""
    extension = file_path.suffix.lower()
    
    format_map = {
        '.spk': 'spk',
        '.mat': 'mat', 
        '.csv': 'csv',
        '.txt': 'txt',
        '.h5': 'h5',
        '.hdf5': 'h5'
    }
    
    if extension in format_map:
        return format_map[extension]
    else:
        raise ValueError(f"Cannot infer format from extension: {extension}")


def load_axion_spk(
    file_path: Union[str, Path],
    use_native_loader: bool = True,
    **kwargs
) -> SpikeList:
    """
    Load data from Axion .spk files.
    
    This function can use either a native Python loader or fall back to
    MATLAB conversion for compatibility.
    
    Parameters
    ---------- 
    file_path : str or Path
        Path to the .spk file
    use_native_loader : bool
        Whether to use native Python loader (default: True)
    **kwargs
        Additional loading parameters
        
    Returns
    -------
    SpikeList
        Loaded spike data
        
    Raises
    ------
    ValueError
        If file cannot be loaded with either method
    """
    file_path = Path(file_path)
    
    if use_native_loader:
        try:
            from .axion_spk_loader import load_axion_spk_native
            return load_axion_spk_native(file_path, **kwargs)
        except Exception as e:
            warnings.warn(f"Native .spk loader failed: {e}. Trying MATLAB fallback.")
    
    # Fallback to MATLAB conversion approach
    mat_path = file_path.with_suffix('.mat')
    
    if mat_path.exists():
        warnings.warn(
            f"Using MATLAB-converted file {mat_path}. "
            "For direct .spk loading, ensure the native loader is working correctly."
        )
        return load_matlab_file(mat_path, **kwargs)
    else:
        raise ValueError(
            f"Cannot load .spk file {file_path}. "
            "Native loader failed and no .mat conversion found. "
            "To create .mat conversion, use MATLAB with AxionFileLoader:\n"
            "[Electrodes, Times] = AxisFile('file.spk').SpikeData.LoadAllSpikes;\n"
            "Channels = Electrodes.Channel;\n"
            "save('converted_file.mat', 'Channels', 'Times');"
        )


def load_matlab_file(
    file_path: Union[str, Path],
    channels_key: str = 'Channels',
    times_key: str = 'Times', 
    recording_length_key: Optional[str] = None,
    time_unit: str = 'ms',
    **kwargs
) -> SpikeList:
    """
    Load MEA data from MATLAB .mat files.
    
    Expected format from AxionFileLoader:
    - Channels: array of channel IDs for each spike
    - Times: array of spike times corresponding to channels
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .mat file
    channels_key : str
        Key for channel data in .mat file (default: 'Channels')
    times_key : str  
        Key for spike times data in .mat file (default: 'Times')
    recording_length_key : str, optional
        Key for recording length in .mat file
    time_unit : str
        Unit of spike times ('ms' or 's', default: 'ms')
    **kwargs
        Additional parameters for SpikeList creation
        
    Returns
    -------
    SpikeList
        Loaded spike data
        
    Raises
    ------
    ImportError
        If scipy is not available
    KeyError
        If required keys are not found in .mat file
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for loading .mat files")
    
    # Load .mat file
    try:
        data = loadmat(str(file_path))
    except Exception as e:
        raise ValueError(f"Error loading .mat file: {e}")
    
    # Extract channels and times
    try:
        if channels_key not in data:
            raise KeyError(f"Key '{channels_key}' not found in .mat file")
        if times_key not in data:
            raise KeyError(f"Key '{times_key}' not found in .mat file")
            
        channels = data[channels_key].flatten()
        times = data[times_key].flatten()
        
    except KeyError as e:
        # Try to suggest available keys
        available_keys = [k for k in data.keys() if not k.startswith('__')]
        raise KeyError(f"{e}. Available keys: {available_keys}")
    
    # Convert time units if necessary
    if time_unit.lower() == 'ms':
        times = times / 1000.0  # Convert ms to seconds
    elif time_unit.lower() != 's':
        raise ValueError("time_unit must be 'ms' or 's'")
    
    # Check data consistency
    if len(channels) != len(times):
        raise ValueError(
            f"Channels and times arrays must have same length. "
            f"Got {len(channels)} channels and {len(times)} times."
        )
    
    # Create spike data list
    spike_data = [(int(ch), float(t)) for ch, t in zip(channels, times)]
    
    # Get recording length if available
    recording_length = None
    if recording_length_key and recording_length_key in data:
        recording_length = float(data[recording_length_key])
        
    # Extract additional parameters from kwargs
    well_map = kwargs.get('well_map', None)
    sampling_rate = kwargs.get('sampling_rate', 12500.0)
    
    return SpikeList(
        spike_data=spike_data,
        recording_length=recording_length,
        well_map=well_map,
        sampling_rate=sampling_rate
    )


def load_csv_file(
    file_path: Union[str, Path],
    channel_col: str = 'channel',
    time_col: str = 'time',
    time_unit: str = 's',
    **kwargs
) -> SpikeList:
    """
    Load MEA data from CSV files.
    
    Expected format:
    - CSV with columns for channel ID and spike times
    - One row per spike event
    
    Parameters
    ----------
    file_path : str or Path
        Path to CSV file
    channel_col : str
        Column name for channel IDs (default: 'channel')
    time_col : str
        Column name for spike times (default: 'time') 
    time_unit : str
        Unit of spike times ('ms' or 's', default: 's')
    **kwargs
        Additional parameters for SpikeList creation
        
    Returns
    -------
    SpikeList
        Loaded spike data
    """
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Check required columns
    if channel_col not in df.columns:
        raise KeyError(f"Column '{channel_col}' not found in CSV")
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in CSV")
    
    channels = df[channel_col].values
    times = df[time_col].values
    
    # Convert time units if necessary  
    if time_unit.lower() == 'ms':
        times = times / 1000.0
    elif time_unit.lower() != 's':
        raise ValueError("time_unit must be 'ms' or 's'")
    
    # Create spike data
    spike_data = [(int(ch), float(t)) for ch, t in zip(channels, times)]
    
    return SpikeList(
        spike_data=spike_data,
        **kwargs
    )


def load_hdf5_file(
    file_path: Union[str, Path],
    channels_dataset: str = 'channels',
    times_dataset: str = 'times',
    **kwargs
) -> SpikeList:
    """
    Load MEA data from HDF5 files.
    
    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file
    channels_dataset : str
        Dataset name for channel data
    times_dataset : str  
        Dataset name for spike times
    **kwargs
        Additional parameters for SpikeList creation
        
    Returns
    -------
    SpikeList
        Loaded spike data
    """
    with h5py.File(file_path, 'r') as f:
        if channels_dataset not in f:
            raise KeyError(f"Dataset '{channels_dataset}' not found in HDF5 file")
        if times_dataset not in f:
            raise KeyError(f"Dataset '{times_dataset}' not found in HDF5 file")
            
        channels = f[channels_dataset][:]
        times = f[times_dataset][:]
    
    spike_data = [(int(ch), float(t)) for ch, t in zip(channels, times)]
    
    return SpikeList(
        spike_data=spike_data,
        **kwargs
    )


def load_from_dataframe(
    dataframe: pd.DataFrame,
    channel_col: str = 'channel', 
    time_col: str = 'time',
    **kwargs
) -> SpikeList:
    """
    Load MEA data from pandas DataFrame.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing spike data
    channel_col : str
        Column name for channel IDs
    time_col : str
        Column name for spike times
    **kwargs
        Additional parameters for SpikeList creation
        
    Returns
    -------
    SpikeList
        Loaded spike data
    """
    if channel_col not in dataframe.columns:
        raise KeyError(f"Column '{channel_col}' not found in DataFrame")
    if time_col not in dataframe.columns:
        raise KeyError(f"Column '{time_col}' not found in DataFrame")
        
    channels = dataframe[channel_col].values  
    times = dataframe[time_col].values
    
    spike_data = [(int(ch), float(t)) for ch, t in zip(channels, times)]
    
    return SpikeList(
        spike_data=spike_data,
        **kwargs
    )


def load_multiple_files(
    file_paths: List[Union[str, Path]],
    condition_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, SpikeList]:
    """
    Load multiple MEA data files for comparative analysis.
    
    Parameters
    ----------
    file_paths : list of str or Path
        List of file paths to load
    condition_names : list of str, optional
        Names for each condition/file (default: use filenames)
    **kwargs
        Additional parameters passed to load_data
        
    Returns
    -------
    dict
        Dictionary mapping condition names to SpikeList objects
    """
    if condition_names is None:
        condition_names = [Path(fp).stem for fp in file_paths]
    
    if len(condition_names) != len(file_paths):
        raise ValueError("Number of condition names must match number of files")
    
    spike_lists = {}
    
    for name, file_path in zip(condition_names, file_paths):
        try:
            spike_lists[name] = load_data(file_path, **kwargs)
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            
    return spike_lists