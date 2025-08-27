"""
Input/output utilities for MEA-Flow.

This module provides functions for saving and loading analysis results
in various formats.
"""

import pickle
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import warnings
import h5py


def save_results(
    results: Dict[str, Any],
    file_path: Union[str, Path],
    format: str = 'auto'
) -> bool:
    """
    Save analysis results to file.
    
    Parameters
    ----------
    results : dict
        Results dictionary to save
    file_path : str or Path
        Output file path
    format : str
        Output format ('pickle', 'json', 'hdf5', 'auto')
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    
    # Auto-detect format from extension
    if format == 'auto':
        ext = file_path.suffix.lower()
        if ext == '.pkl':
            format = 'pickle'
        elif ext == '.json':
            format = 'json'
        elif ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            format = 'pickle'  # Default
            file_path = file_path.with_suffix('.pkl')
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = _prepare_for_json(results)
            with open(file_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        
        elif format == 'hdf5':
            _save_to_hdf5(results, file_path)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to save results: {e}")
        return False


def load_results(
    file_path: Union[str, Path],
    format: str = 'auto'
) -> Dict[str, Any]:
    """
    Load analysis results from file.
    
    Parameters
    ----------
    file_path : str or Path
        Input file path
    format : str
        File format ('pickle', 'json', 'hdf5', 'auto')
        
    Returns
    -------
    dict
        Loaded results dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect format
    if format == 'auto':
        ext = file_path.suffix.lower()
        if ext == '.pkl':
            format = 'pickle'
        elif ext == '.json':
            format = 'json'
        elif ext in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            # Try to detect from content
            try:
                with open(file_path, 'rb') as f:
                    pickle.load(f)
                format = 'pickle'
            except:
                format = 'json'
    
    if format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    elif format == 'json':
        with open(file_path, 'r') as f:
            json_results = json.load(f)
        return _restore_from_json(json_results)
    
    elif format == 'hdf5':
        return _load_from_hdf5(file_path)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def export_to_format(
    data: Union[pd.DataFrame, Dict[str, Any]],
    file_path: Union[str, Path],
    format: str = 'csv'
) -> bool:
    """
    Export data to various formats for external analysis.
    
    Parameters
    ----------
    data : DataFrame or dict
        Data to export
    file_path : str or Path
        Output file path
    format : str
        Export format ('csv', 'excel', 'matlab', 'json')
        
    Returns
    -------
    bool
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError("Data must be DataFrame or dict for CSV export")
        
        elif format == 'excel':
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False)
            elif isinstance(data, dict):
                with pd.ExcelWriter(file_path) as writer:
                    for sheet_name, sheet_data in data.items():
                        if isinstance(sheet_data, pd.DataFrame):
                            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        elif isinstance(sheet_data, dict):
                            pd.DataFrame(sheet_data).to_excel(writer, sheet_name=sheet_name, index=False)
        
        elif format == 'matlab':
            from scipy.io import savemat
            
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to dict for MATLAB
                matlab_data = {col: data[col].values for col in data.columns}
            elif isinstance(data, dict):
                matlab_data = data.copy()
            else:
                raise ValueError("Data must be DataFrame or dict for MATLAB export")
            
            savemat(file_path, matlab_data)
        
        elif format == 'json':
            json_data = _prepare_for_json(data)
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to export data: {e}")
        return False


def _prepare_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return {
            '__type__': 'dataframe',
            'data': obj.to_dict(orient='records'),
            'columns': obj.columns.tolist()
        }
    elif isinstance(obj, dict):
        return {key: _prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_prepare_for_json(item) for item in obj]
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # Try to convert to string for unknown types
        return str(obj)


def _restore_from_json(obj: Any) -> Any:
    """Restore objects from JSON format."""
    if isinstance(obj, dict):
        if obj.get('__type__') == 'dataframe':
            return pd.DataFrame(obj['data'], columns=obj['columns'])
        else:
            return {key: _restore_from_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_restore_from_json(item) for item in obj]
    else:
        return obj


def _save_to_hdf5(results: Dict[str, Any], file_path: Path) -> None:
    """Save results to HDF5 format."""
    with h5py.File(file_path, 'w') as f:
        _write_dict_to_hdf5(f, results)


def _write_dict_to_hdf5(group, data_dict: Dict[str, Any]) -> None:
    """Recursively write dictionary to HDF5 group."""
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, pd.DataFrame):
            # Save DataFrame as structured data
            df_group = group.create_group(f"{key}_dataframe")
            for col in value.columns:
                df_group.create_dataset(col, data=value[col].values)
            df_group.attrs['columns'] = [str(c) for c in value.columns]
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_dict_to_hdf5(subgroup, value)
        elif isinstance(value, (list, tuple)):
            try:
                group.create_dataset(key, data=np.array(value))
            except:
                # Store as string if array conversion fails
                group.attrs[key] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            group.attrs[key] = value
        else:
            # Store as string attribute for unknown types
            group.attrs[key] = str(value)


def _load_from_hdf5(file_path: Path) -> Dict[str, Any]:
    """Load results from HDF5 format."""
    with h5py.File(file_path, 'r') as f:
        return _read_hdf5_group(f)


def _read_hdf5_group(group) -> Dict[str, Any]:
    """Recursively read HDF5 group to dictionary."""
    result = {}
    
    # Read datasets
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[...]
        elif isinstance(item, h5py.Group):
            if 'columns' in item.attrs:
                # Reconstruct DataFrame
                columns = item.attrs['columns']
                data_dict = {col: item[col][...] for col in columns}
                result[key.replace('_dataframe', '')] = pd.DataFrame(data_dict)
            else:
                result[key] = _read_hdf5_group(item)
    
    # Read attributes
    for key, value in group.attrs.items():
        result[key] = value
    
    return result


def create_analysis_report(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    include_plots: bool = True
) -> Path:
    """
    Create a comprehensive analysis report.
    
    Parameters
    ----------
    results : dict
        Analysis results dictionary
    output_dir : str or Path
        Output directory for report
    include_plots : bool
        Whether to include generated plots
        
    Returns
    -------
    Path
        Path to the generated report directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    save_results(results, output_dir / 'analysis_results.pkl')
    
    # Export summary statistics if available
    if 'metrics_summary' in results:
        export_to_format(
            results['metrics_summary'],
            output_dir / 'metrics_summary.csv',
            format='csv'
        )
    
    # Create README
    readme_content = _generate_report_readme(results)
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # Save plots if available and requested
    if include_plots and 'figures' in results:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        for plot_name, fig in results['figures'].items():
            try:
                fig.savefig(plots_dir / f'{plot_name}.png', dpi=300, bbox_inches='tight')
                fig.savefig(plots_dir / f'{plot_name}.pdf', bbox_inches='tight')
            except Exception as e:
                warnings.warn(f"Failed to save plot {plot_name}: {e}")
    
    return output_dir


def _generate_report_readme(results: Dict[str, Any]) -> str:
    """Generate README content for analysis report."""
    content = "# MEA-Flow Analysis Report\n\n"
    
    content += "This directory contains the results of MEA-Flow analysis.\n\n"
    
    content += "## Files\n\n"
    content += "- `analysis_results.pkl`: Complete analysis results (Python pickle format)\n"
    content += "- `metrics_summary.csv`: Summary metrics in CSV format\n"
    content += "- `plots/`: Directory containing generated plots\n"
    content += "- `README.md`: This file\n\n"
    
    if 'config' in results:
        content += "## Analysis Configuration\n\n"
        config = results['config']
        for key, value in config.__dict__.items():
            content += f"- {key}: {value}\n"
        content += "\n"
    
    if 'summary_stats' in results:
        content += "## Summary Statistics\n\n"
        stats = results['summary_stats']
        for key, value in stats.items():
            content += f"- {key}: {value}\n"
        content += "\n"
    
    content += "## Loading Results\n\n"
    content += "To load the complete results in Python:\n\n"
    content += "```python\n"
    content += "import pickle\n\n"
    content += "with open('analysis_results.pkl', 'rb') as f:\n"
    content += "    results = pickle.load(f)\n"
    content += "```\n\n"
    
    content += "To load the summary metrics:\n\n"
    content += "```python\n"
    content += "import pandas as pd\n\n"
    content += "metrics = pd.read_csv('metrics_summary.csv')\n"
    content += "```\n"
    
    return content