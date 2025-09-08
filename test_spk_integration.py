#!/usr/bin/env python3
"""
Test script to verify .spk file loading works with MEA-Flow analysis pipeline
"""

from pathlib import Path
from mea_flow.data import load_data

def test_spk_loading():
    """Test loading .spk files with MEA-Flow"""
    
    # Dataset configuration - using .spk files instead of .mat
    data_path = '/media/neuro/Data/MEA-data/'
    conditions = ['control', 'chronic-stress', 'miR-186-5p-inhibition']
    filenames = ['n1-DIV17-01.spk', 'n2-DIV17-01.spk', 'n3-DIV17-01.spk']  # Changed to .spk files

    # Load datasets into condition-based dictionary
    spike_lists = {}

    print("Testing .spk file loading with MEA-Flow...")
    
    for condition, filename in zip(conditions, filenames):
        file_path = Path(data_path) / filename
        
        print(f"\nLoading {condition}: {filename}")
        
        if not file_path.exists():
            print(f"  WARNING: File {file_path} does not exist, skipping...")
            continue
            
        try:
            # Load using native .spk loader (no need to specify channels_key/times_key for .spk)
            spike_list = load_data(
                file_path=file_path,
                data_format='spk'  # Use native .spk format
            )
            
            spike_lists[condition] = spike_list
            
            # Verify the loaded data
            total_spikes = sum(len(train.times) for train in spike_list.spike_data.values())
            num_channels = len(spike_list.spike_data)
            
            print(f"  ✓ Successfully loaded {total_spikes:,} spikes from {num_channels} channels")
            
            # Show time range
            all_times = []
            for train in spike_list.spike_data.values():
                all_times.extend(train.times)
            
            if all_times:
                min_time = min(all_times)
                max_time = max(all_times)
                print(f"  ✓ Time range: {min_time:.6f} - {max_time:.6f} seconds")
            
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully loaded {len(spike_lists)} datasets:")
    for condition, spike_list in spike_lists.items():
        total_spikes = sum(len(train.times) for train in spike_list.spike_data.values())
        print(f"  {condition}: {total_spikes:,} spikes")
    
    return spike_lists

if __name__ == "__main__":
    spike_lists = test_spk_loading()
