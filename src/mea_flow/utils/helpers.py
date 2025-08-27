"""
Helper utility functions for MEA-Flow.

This module provides common utility functions used across the package.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import warnings
from pathlib import Path


def validate_inputs(*args, **kwargs) -> bool:
    """
    Validate input arguments for MEA-Flow functions.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments to validate
    **kwargs : dict
        Keyword arguments to validate
        
    Returns
    -------
    bool
        True if all inputs are valid
        
    Raises
    ------
    ValueError
        If any inputs are invalid
    """
    # Basic validation for common input types
    for i, arg in enumerate(args):
        if arg is None:
            raise ValueError(f"Argument {i} cannot be None")
        
        # Validate numpy arrays
        if isinstance(arg, np.ndarray):
            if arg.size == 0:
                raise ValueError(f"Array argument {i} cannot be empty")
            if np.any(np.isnan(arg)) or np.any(np.isinf(arg)):
                warnings.warn(f"Array argument {i} contains NaN or infinite values")
        
        # Validate lists
        elif isinstance(arg, list):
            if len(arg) == 0:
                raise ValueError(f"List argument {i} cannot be empty")
    
    return True


def check_data_format(data: Any, expected_type: type = None) -> bool:
    """
    Check if data is in expected format.
    
    Parameters
    ----------
    data : Any
        Data to check
    expected_type : type, optional
        Expected data type
        
    Returns
    -------
    bool
        True if data format is valid
    """
    if expected_type and not isinstance(data, expected_type):
        return False
    
    # Check for common data format issues
    if isinstance(data, np.ndarray):
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            warnings.warn("Data contains NaN or infinite values")
        
        # Check for empty arrays
        if data.size == 0:
            warnings.warn("Data array is empty")
            return False
    
    elif isinstance(data, pd.DataFrame):
        # Check for empty DataFrame
        if data.empty:
            warnings.warn("DataFrame is empty")
            return False
        
        # Check for columns with all NaN values
        all_nan_cols = data.columns[data.isna().all()].tolist()
        if all_nan_cols:
            warnings.warn(f"Columns with all NaN values: {all_nan_cols}")
    
    return True


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for MEA-Flow.
    
    Parameters
    ----------
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : str, optional
        Path to log file (default: console only)
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('mea_flow')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def normalize_data(
    data: np.ndarray,
    method: str = 'zscore',
    axis: int = 0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using specified method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    method : str
        Normalization method ('zscore', 'minmax', 'robust')
    axis : int
        Axis along which to normalize
        
    Returns
    -------
    tuple
        (normalized_data, normalization_params)
    """
    params = {'method': method, 'axis': axis}
    
    if method == 'zscore':
        mean_val = np.mean(data, axis=axis, keepdims=True)
        std_val = np.std(data, axis=axis, keepdims=True)
        
        # Avoid division by zero
        std_val[std_val == 0] = 1
        
        normalized = (data - mean_val) / std_val
        params.update({'mean': mean_val, 'std': std_val})
        
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        
        normalized = (data - min_val) / range_val
        params.update({'min': min_val, 'max': max_val})
        
    elif method == 'robust':
        median_val = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median_val), axis=axis, keepdims=True)
        
        # Avoid division by zero
        mad[mad == 0] = 1
        
        normalized = (data - median_val) / mad
        params.update({'median': median_val, 'mad': mad})
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(
    normalized_data: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Reverse normalization using stored parameters.
    
    Parameters
    ----------
    normalized_data : np.ndarray
        Normalized data array
    params : dict
        Normalization parameters from normalize_data
        
    Returns
    -------
    np.ndarray
        Original scale data
    """
    method = params['method']
    
    if method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    
    elif method == 'minmax':
        range_val = params['max'] - params['min']
        return normalized_data * range_val + params['min']
    
    elif method == 'robust':
        return normalized_data * params['mad'] + params['median']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def find_optimal_window_size(
    data_length: float,
    min_windows: int = 10,
    max_window_size: float = 30.0
) -> float:
    """
    Find optimal window size for time-windowed analysis.
    
    Parameters
    ----------
    data_length : float
        Total data length in seconds
    min_windows : int
        Minimum number of windows desired
    max_window_size : float
        Maximum window size in seconds
        
    Returns
    -------
    float
        Optimal window size in seconds
    """
    # Calculate window size based on minimum windows
    window_from_min = data_length / min_windows
    
    # Choose the smaller of the two constraints
    optimal_size = min(window_from_min, max_window_size)
    
    # Ensure at least 1 second windows
    optimal_size = max(optimal_size, 1.0)
    
    return optimal_size


def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
    confidence : float
        Confidence level (0-1)
        
    Returns
    -------
    tuple
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        return np.nan, np.nan
    
    mean_val = np.mean(data_clean)
    sem = stats.sem(data_clean)
    
    # Calculate t-value for given confidence level
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, len(data_clean) - 1)
    
    margin_error = t_val * sem
    
    return mean_val - margin_error, mean_val + margin_error


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Perform safe division avoiding division by zero.
    
    Parameters
    ----------
    numerator : float or array
        Numerator values
    denominator : float or array
        Denominator values
    default : float
        Value to return when denominator is zero
        
    Returns
    -------
    float or array
        Division result with safe handling of zero denominators
    """
    if np.isscalar(denominator):
        if denominator == 0:
            return default
        else:
            return numerator / denominator
    else:
        result = np.full_like(numerator, default, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result


def ensure_2d(data: np.ndarray) -> np.ndarray:
    """
    Ensure data is 2D array.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array
        
    Returns
    -------
    np.ndarray
        2D version of input data
    """
    if data.ndim == 1:
        return data.reshape(-1, 1)
    elif data.ndim == 0:
        return data.reshape(1, 1)
    else:
        return data


def create_time_vector(
    start_time: float,
    end_time: float,
    dt: float
) -> np.ndarray:
    """
    Create time vector with specified parameters.
    
    Parameters
    ----------
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    dt : float
        Time step in seconds
        
    Returns
    -------
    np.ndarray
        Time vector
    """
    n_points = int((end_time - start_time) / dt) + 1
    return np.linspace(start_time, end_time, n_points)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns
    -------
    dict
        Memory usage statistics
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def progress_bar(
    current: int,
    total: int,
    description: str = "Progress",
    bar_length: int = 50
) -> str:
    """
    Create a simple progress bar string.
    
    Parameters
    ----------
    current : int
        Current progress value
    total : int
        Total progress value
    description : str
        Description text
    bar_length : int
        Length of progress bar in characters
        
    Returns
    -------
    str
        Progress bar string
    """
    progress = current / total
    bar = '█' * int(bar_length * progress) + '░' * (bar_length - int(bar_length * progress))
    
    return f"{description}: [{bar}] {progress:.1%} ({current}/{total})"