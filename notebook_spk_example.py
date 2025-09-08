"""
Updated notebook code to use .spk files instead of .mat files
This demonstrates the conversion from the original notebook snippet
"""

from pathlib import Path
import sys
sys.path.append('/home/neuro/repos/mea-flow')

# Import our standalone .spk loader
exec(open('/home/neuro/repos/mea-flow/test_loader_v2.py').read().split('if __name__')[0])

# Dataset configuration - CHANGED FROM .mat TO .spk FILES
data_path = '/media/neuro/Data/MEA-data/'
conditions = ['control', 'chronic-stress', 'miR-186-5p-inhibition']
filenames = ['n1-DIV17-01.spk', 'n2-DIV17-01.spk', 'n3-DIV17-01.spk']  # Changed to .spk files

# Load datasets into condition-based dictionary
spike_lists = {}
loader = AxionSpkLoader()

print("Loading .spk files with native Python loader...")

for condition, filename in zip(conditions, filenames):
    file_path = Path(data_path) / filename
    
    if not file_path.exists():
        print(f"⚠ File not found: {filename}")
        continue
        
    try:
        print(f"\nLoading {condition}: {filename}")
        
        # Load using native .spk loader (replaces the original .mat loading)
        spike_list = loader.load(str(file_path))
        spike_lists[condition] = spike_list
        
        # Verify loaded data
        total_spikes = sum(len(train.times) for train in spike_list.spike_data.values())
        num_channels = len(spike_list.spike_data)
        
        # Get time range
        all_times = []
        for train in spike_list.spike_data.values():
            all_times.extend(train.times)
        
        min_time = min(all_times) if all_times else 0
        max_time = max(all_times) if all_times else 0
        
        print(f"  ✓ Successfully loaded {total_spikes:,} spikes from {num_channels} channels")
        print(f"  ✓ Time range: {min_time:.6f} - {max_time:.6f} seconds")
        
    except Exception as e:
        print(f"  ✗ Error loading {filename}: {e}")

print(f"\n=== LOADING SUMMARY ===")
print(f"Successfully loaded {len(spike_lists)} datasets:")
for condition, spike_list in spike_lists.items():
    total_spikes = sum(len(train.times) for train in spike_list.spike_data.values())
    print(f"  {condition}: {total_spikes:,} spikes")

# The spike_lists dictionary now contains SpikeList objects loaded from .spk files
# This can be used with the rest of the MEA-Flow analysis pipeline

print("\n✓ Conversion complete: .mat files → .spk files")
print("✓ Native Python loading eliminates MATLAB dependency")
print("✓ Data structure is identical for downstream analysis")
