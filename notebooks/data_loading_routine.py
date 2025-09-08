"""
Data loading routine for MEA-Flow analysis with concrete dataset.

This script loads .spk files from specified conditions and creates a dictionary
of SpikeList objects for comparative analysis.
"""

import os
from pathlib import Path
from typing import Dict, List
import warnings

# Import MEA-Flow data loading functions
from mea_flow.data import load_data, SpikeList


def load_condition_datasets(
    data_path: str,
    conditions: List[str],
    filenames: List[str],
    **kwargs
) -> Dict[str, SpikeList]:
    """
    Load MEA datasets for multiple conditions.
    
    Parameters
    ----------
    data_path : str
        Base path where data files are stored
    conditions : list of str
        List of condition names (e.g., ['control', 'chronic-stress', 'miR-186-5p-inhibition'])
    filenames : list of str
        List of corresponding filenames (e.g., ['n1-DIV17-01.spk', 'n2-DIV17-01.spk', 'n3-DIV17-01.spk'])
    **kwargs
        Additional parameters passed to load_data function
        
    Returns
    -------
    dict
        Dictionary mapping condition names to SpikeList objects
        Format: spike_lists[condition] = spike_list
        
    Raises
    ------
    ValueError
        If conditions and filenames lists have different lengths
    FileNotFoundError
        If any of the specified files cannot be found
    """
    
    if len(conditions) != len(filenames):
        raise ValueError(
            f"Number of conditions ({len(conditions)}) must match "
            f"number of filenames ({len(filenames)})"
        )
    
    data_path = Path(data_path)
    spike_lists = {}
    
    print(f"Loading MEA data from: {data_path}")
    print(f"Conditions: {conditions}")
    print(f"Files: {filenames}")
    print("-" * 50)
    
    for condition, filename in zip(conditions, filenames):
        file_path = data_path / filename
        
        print(f"Loading {condition}: {filename}")
        
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load the .spk file using MEA-Flow's native loader
            spike_list = load_data(
                file_path=file_path,
                data_format='spk',
                use_native_loader=True,
                **kwargs
            )
            
            spike_lists[condition] = spike_list
            
            # Print basic statistics
            n_spikes = len(spike_list.spike_data)
            n_channels = len(spike_list.get_active_channels())
            duration = spike_list.recording_length or "Unknown"
            
            print(f"  ✓ Loaded successfully")
            print(f"    - Total spikes: {n_spikes:,}")
            print(f"    - Active channels: {n_channels}")
            print(f"    - Recording duration: {duration}s")
            
        except Exception as e:
            error_msg = f"Failed to load {condition} ({filename}): {str(e)}"
            print(f"  ✗ {error_msg}")
            warnings.warn(error_msg)
            continue
    
    print("-" * 50)
    print(f"Successfully loaded {len(spike_lists)} out of {len(conditions)} conditions")
    
    return spike_lists


def main():
    """
    Main function to load the specific dataset mentioned in the analysis example.
    """
    
    # Dataset configuration from analysis_example.ipynb
    data_path = '/media/neuro/Data/MEA-data/'
    conditions = ['control', 'chronic-stress', 'miR-186-5p-inhibition']
    filenames = ['n1-DIV17-01.spk', 'n2-DIV17-01.spk', 'n3-DIV17-01.spk']
    
    # Load the datasets
    try:
        spike_lists = load_condition_datasets(
            data_path=data_path,
            conditions=conditions,
            filenames=filenames,
            # Optional parameters for .spk loading
            sampling_rate=12500.0,  # Standard Axion sampling rate
        )
        
        # Display summary
        print("\n" + "=" * 60)
        print("DATASET LOADING SUMMARY")
        print("=" * 60)
        
        for condition, spike_list in spike_lists.items():
            print(f"\n{condition.upper()}:")
            print(f"  - Spikes: {len(spike_list.spike_data):,}")
            print(f"  - Channels: {len(spike_list.get_active_channels())}")
            print(f"  - Duration: {spike_list.recording_length}s")
            print(f"  - Sampling rate: {spike_list.sampling_rate} Hz")
            
            # Show channel activity distribution
            channel_counts = {}
            for channel, _ in spike_list.spike_data:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            
            if channel_counts:
                most_active = max(channel_counts.items(), key=lambda x: x[1])
                least_active = min(channel_counts.items(), key=lambda x: x[1])
                print(f"  - Most active channel: {most_active[0]} ({most_active[1]} spikes)")
                print(f"  - Least active channel: {least_active[0]} ({least_active[1]} spikes)")
        
        return spike_lists
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return {}


if __name__ == "__main__":
    spike_lists = main()
