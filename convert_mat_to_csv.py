#!/usr/bin/env python3
"""
Convert MATLAB .mat files to CSV format for MEA spike data.

This script converts Axion .mat files containing spike timing and channel data
to CSV format that can be loaded by MEA-Flow routines.
"""

import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path

def convert_mat_to_csv(mat_file_path: str, csv_file_path: str = None):
    """
    Convert .mat file to CSV format.
    
    Parameters
    ----------
    mat_file_path : str
        Path to input .mat file
    csv_file_path : str, optional
        Path to output CSV file. If None, uses same name with .csv extension
    """
    mat_path = Path(mat_file_path)
    
    if csv_file_path is None:
        csv_path = mat_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_file_path)
    
    print(f"Converting {mat_path.name} to {csv_path.name}...")
    
    # Load MATLAB data
    mat_data = scipy.io.loadmat(str(mat_path))
    
    # Extract spike times and channels
    times = mat_data['Times'].flatten()
    channels = mat_data['Channels'].flatten()
    
    print(f"  Loaded {len(times):,} spikes from {len(np.unique(channels))} channels")
    print(f"  Time range: {times.min():.6f} - {times.max():.6f} seconds")
    print(f"  Channel range: {channels.min()} - {channels.max()}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'channel': channels
    })
    
    # Sort by time for better organization
    df = df.sort_values('time').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False, float_format='%.6f')
    
    print(f"  ✓ Saved {len(df):,} spike records to {csv_path}")
    
    return csv_path

def main():
    """Convert all .mat files in the data directory to CSV format."""
    
    # Data directory
    data_path = Path('/media/neuro/Data/MEA-data/')
    
    # Files to convert
    mat_files = [
        'n1-DIV17-01.mat',
        'n2-DIV17-01.mat', 
        'n3-DIV17-01.mat'
    ]
    
    print("Converting MATLAB .mat files to CSV format...\n")
    
    converted_files = []
    
    for mat_file in mat_files:
        mat_path = data_path / mat_file
        
        if mat_path.exists():
            try:
                csv_path = convert_mat_to_csv(str(mat_path))
                converted_files.append(csv_path)
                print()
            except Exception as e:
                print(f"  ❌ Error converting {mat_file}: {e}\n")
        else:
            print(f"  ⚠️  File not found: {mat_path}\n")
    
    print(f"=== CONVERSION SUMMARY ===")
    print(f"Successfully converted {len(converted_files)} files:")
    for csv_path in converted_files:
        print(f"  ✓ {csv_path.name}")
    
    print(f"\nCSV files are ready for loading with MEA-Flow routines!")

if __name__ == "__main__":
    main()
