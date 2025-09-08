import sys
sys.path.append('/home/neuro/repos/mea-flow/src')

from pathlib import Path
from mea_flow.data import load_data, load_axion_spk
from mea_flow.data.axion_spk_loader import load_axion_spk_native, probe_spk_file_structure
import warnings

# Test file paths
data_path = '/media/neuro/Data/MEA-data/'
spk_file = Path(data_path) / 'n1-DIV17-01.spk'
mat_file = Path(data_path) / 'n1-DIV17-01.mat'

print("="*60)
print("TESTING AXION .SPK NATIVE LOADER")
print("="*60)

# 1. Probe the .spk file structure first
print("\n1. PROBING .SPK FILE STRUCTURE:")
try:
    spk_info = probe_spk_file_structure(spk_file)
    for key, value in spk_info.items():
        print(f"   {key}: {value}")
except Exception as e:
    print(f"   Error probing file: {e}")

# 2. Test native .spk loader directly
print("\n2. TESTING NATIVE .SPK LOADER:")
try:
    spk_spike_list = load_axion_spk_native(spk_file)
    print(f"   ✓ Native loader succeeded")
    total_spikes = sum(train.n_spikes for train in spk_spike_list.spike_trains.values())
    print(f"   - Total spikes: {total_spikes}")
    print(f"   - Active channels: {len(spk_spike_list.get_active_channels())}")
    print(f"   - Recording length: {spk_spike_list.recording_length}s")
    print(f"   - Sampling rate: {spk_spike_list.sampling_rate} Hz")
except Exception as e:
    print(f"   ✗ Native loader failed: {e}")
    spk_spike_list = None

# 3. Test .spk loader with fallback
print("\n3. TESTING .SPK LOADER WITH FALLBACK:")
try:
    spk_fallback = load_axion_spk(spk_file, use_native_loader=True)
    print(f"   ✓ Fallback loader succeeded")
    total_spikes = sum(train.n_spikes for train in spk_fallback.spike_trains.values())
    print(f"   - Total spikes: {total_spikes}")
    print(f"   - Active channels: {len(spk_fallback.get_active_channels())}")
except Exception as e:
    print(f"   ✗ Fallback loader failed: {e}")
    spk_fallback = None

# 4. Load .mat file for comparison
print("\n4. LOADING .MAT FILE FOR COMPARISON:")
try:
    mat_spike_list = load_data(
        file_path=mat_file,
        data_format='mat',
        channels_key='Channels',
        times_key='Times',
        time_unit='s'
    )
    print(f"   ✓ .MAT file loaded successfully")
    total_spikes = sum(train.n_spikes for train in mat_spike_list.spike_trains.values())
    print(f"   - Total spikes: {total_spikes}")
    print(f"   - Active channels: {len(mat_spike_list.get_active_channels())}")
    print(f"   - Recording length: {mat_spike_list.recording_length}s")
    print(f"   - Sampling rate: {mat_spike_list.sampling_rate} Hz")
except Exception as e:
    print(f"   ✗ .MAT file loading failed: {e}")
    mat_spike_list = None

# 5. Compare results if both loaded successfully
print("\n5. COMPARISON RESULTS:")
if spk_spike_list and mat_spike_list:
    spk_count = sum(train.n_spikes for train in spk_spike_list.spike_trains.values())
    mat_count = sum(train.n_spikes for train in mat_spike_list.spike_trains.values())
    
    print(f"   .SPK spikes: {spk_count:,}")
    print(f"   .MAT spikes: {mat_count:,}")
    print(f"   Match: {'✓' if spk_count == mat_count else '✗'}")
    
    if spk_count == mat_count:
        print("   🎉 Native .spk loader is working correctly!")
    else:
        print("   ⚠️  Spike counts don't match - loader needs debugging")
        
elif mat_spike_list and not spk_spike_list:
    print("   📋 Native .spk loader failed, use .mat files for now")
    mat_count = sum(train.n_spikes for train in mat_spike_list.spike_trains.values())
    print(f"   .MAT file contains {mat_count:,} spikes")
else:
    print("   ❌ Both loaders failed")

print("\n" + "="*60)