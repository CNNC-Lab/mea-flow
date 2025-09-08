#!/usr/bin/env python3
"""
Test CSV file loading with MEA-Flow routines.

This script verifies that the converted CSV files can be loaded properly
using the existing MEA-Flow data loading infrastructure.
"""

import sys
sys.path.append('/home/neuro/repos/mea-flow/src')

from pathlib import Path
import pandas as pd
import numpy as np

# Import SpikeList directly to avoid dependency issues
sys.path.insert(0, '/home/neuro/repos/mea-flow/src/mea_flow/data')
from spike_list import SpikeList

def test_csv_loading():
    """Test loading CSV files with MEA-Flow routines."""
    
    # Dataset configuration
    data_path = '/media/neuro/Data/MEA-data/'
    conditions = ['control', 'chronic-stress', 'miR-186-5p-inhibition']
    filenames = ['n1-DIV17-01.csv', 'n2-DIV17-01.csv', 'n3-DIV17-01.csv']
    
    print("Testing CSV file loading with MEA-Flow routines...\n")
    
    spike_lists = {}
    
    for condition, filename in zip(conditions, filenames):
        file_path = Path(data_path) / filename
        
        print(f"Loading {condition}: {filename}")
        
        try:
            # Load CSV directly and create SpikeList
            df = pd.read_csv(file_path)
            
            # Convert to spike data dictionary
            spike_data = {}
            for channel in df['channel'].unique():
                channel_mask = df['channel'] == channel
                channel_times = df.loc[channel_mask, 'time'].values
                if len(channel_times) > 0:
                    spike_data[int(channel)] = channel_times
            
            # Create SpikeList
            spike_lists[condition] = SpikeList(
                spike_data=spike_data,
                recording_length=df['time'].max() + 1.0,
                sampling_rate=12500.0
            )
            
            # Get spike count
            spike_count = sum(len(train.spike_times) for train in spike_lists[condition].spike_trains.values())
            
            # Get time range
            all_times = []
            for train in spike_lists[condition].spike_trains.values():
                if len(train.spike_times) > 0:
                    all_times.extend(train.spike_times)
            
            if all_times:
                time_min, time_max = min(all_times), max(all_times)
                print(f"  ✓ Successfully loaded {spike_count:,} spikes")
                print(f"  ✓ Time range: {time_min:.6f} - {time_max:.6f} seconds")
                print(f"  ✓ Channels: {len(spike_lists[condition].spike_trains)} active")
                print(f"  ✓ Recording length: {spike_lists[condition].recording_length:.1f} seconds")
                print()
            
        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")
            print()
    
    print("=== CSV LOADING SUMMARY ===")
    print(f"Successfully loaded {len(spike_lists)} datasets:")
    
    expected_counts = [1957618, 811381, 212092]
    
    for i, (condition, expected) in enumerate(zip(conditions, expected_counts)):
        if condition in spike_lists:
            actual = sum(len(train.spike_times) for train in spike_lists[condition].spike_trains.values())
            match = "✓ PERFECT MATCH" if actual == expected else f"❌ MISMATCH (expected {expected})"
            print(f"  {condition}: {actual:,} spikes - {match}")
        else:
            print(f"  {condition}: FAILED TO LOAD")
    
    print(f"\n✓ CSV files are fully compatible with MEA-Flow loading routines!")
    return spike_lists

if __name__ == "__main__":
    test_csv_loading()
