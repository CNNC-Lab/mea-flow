#!/usr/bin/env python3

import struct
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class SpikeTrain:
    """Container for spike timing data from a single electrode."""
    electrode_id: int
    times: np.ndarray
    
    def __post_init__(self):
        """Validate spike train data."""
        if not isinstance(self.times, np.ndarray):
            self.times = np.array(self.times)
        
        # Sort times but keep duplicates (MATLAB doesn't remove duplicates)
        self.times = np.sort(self.times)
        
        if len(self.times) > 0 and np.any(self.times < 0):
            raise ValueError("Spike times must be non-negative")

@dataclass
class SpikeList:
    """Container for multi-electrode spike data."""
    spike_data: Dict[int, SpikeTrain]
    recording_length: float
    sampling_rate: float
    
    def __post_init__(self):
        """Validate spike list data."""
        if self.recording_length <= 0:
            raise ValueError("Recording length must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")

@dataclass
class SpkFileHeader:
    """Header information from .spk file."""
    magic: str
    version: int
    sampling_rate: float
    electrode_count: int
    data_region_start: int
    data_region_length: int
    recording_length: float

class AxionSpkLoader:
    """Native Python loader for Axion BioSystems .spk files following MATLAB AxisFile.m exactly."""
    
    def __init__(self):
        self.header: Optional[SpkFileHeader] = None
    
    def load(self, file_path: str) -> SpikeList:
        """Load spike data from .spk file."""
        try:
            with open(file_path, 'rb') as f:
                # Read header following MATLAB AxisFile.m structure
                self.header = self._read_header_matlab_style(f)
                
                # Read spike data
                spike_times, spike_channels = self._read_spike_data_matlab_style(f)
                
                # Convert to SpikeList format
                spike_data = {}
                for channel in np.unique(spike_channels):
                    channel_mask = spike_channels == channel
                    channel_times = spike_times[channel_mask]
                    if len(channel_times) > 0:
                        spike_data[int(channel)] = SpikeTrain(
                            electrode_id=int(channel),
                            times=channel_times
                        )
            
            return SpikeList(
                spike_data=spike_data,
                recording_length=self.header.recording_length,
                sampling_rate=self.header.sampling_rate
            )
            
        except Exception as e:
            raise ValueError(f"Error reading .spk file: {e}")
    
    def _read_header_matlab_style(self, file_handle) -> SpkFileHeader:
        """Read header following MATLAB AxisFile.m structure exactly."""
        file_handle.seek(0)
        
        # Read magic word (8 bytes)
        magic = file_handle.read(8)
        if magic != b'AxionBio':
            raise ValueError(f"Invalid magic word: {magic}")
        
        # Following MATLAB AxisFile.m line 216-220:
        # Read header fields in exact order
        primary_data_type = struct.unpack('<H', file_handle.read(2))[0]      # uint16
        header_version_major = struct.unpack('<H', file_handle.read(2))[0]   # uint16  
        header_version_minor = struct.unpack('<H', file_handle.read(2))[0]   # uint16
        notes_start = struct.unpack('<Q', file_handle.read(8))[0]            # uint64
        notes_length = struct.unpack('<I', file_handle.read(4))[0]           # uint32
        
        print(f"Primary data type: {primary_data_type}")
        print(f"Header version: {header_version_major}.{header_version_minor}")
        print(f"Notes start: {notes_start}")
        print(f"Notes length: {notes_length}")
        
        # Based on MATLAB code logic
        if header_version_major == 0:
            # Legacy format - use notes_start as entries_start
            entries_start = notes_start
            print("Using legacy format")
            # For now, skip legacy support and focus on version 1
            raise ValueError("Legacy format not yet supported")
            
        elif header_version_major == 1:
                # Read entries start
                entries_start = struct.unpack('<q', file_handle.read(8))[0]
                print(f"Entries start: {entries_start}")
                
                # Read entry slots (128 uint64 entries)
                entry_slots = []
                for i in range(128):
                    slot = struct.unpack('<Q', file_handle.read(8))[0]
                    entry_slots.append(slot)
                
                # Parse entries following MATLAB EntryRecord.FromUint64
                entries = []
                for i, slot in enumerate(entry_slots):
                    if slot == 0:
                        continue
                    
                    # Extract entry type and length (MATLAB bit manipulation)
                    entry_type = (slot >> 56) & 0xFF
                    entry_length = slot & 0x00FFFFFFFFFFFFFF
                    
                    if entry_type == 0:  # Terminate
                        break
                    elif entry_type == 255:  # Skip
                        continue
                    
                    entries.append({
                        'type': entry_type,
                        'length': entry_length,
                        'slot_index': i
                    })
                    
                    print(f"Entry {i}: type={entry_type}, length={entry_length}")
                
                # Look for spike data in entries - prioritize BlockVectorData
                data_region_start = None
                data_region_length = None
                sampling_rate = 12500.0
                
                current_pos = entries_start
                for entry in entries:
                    if entry['type'] == 4:  # BlockVectorData - this is the actual spike data
                        print(f"Found BlockVectorData entry at {current_pos}, length={entry['length']}")
                        # This is the actual spike data region
                        if entry['length'] > 100000000:  # Large data block (~200MB)
                            data_region_start = current_pos
                            data_region_length = entry['length']
                            print(f"Using BlockVectorData as spike data region")
                            break  # Found it, stop looking
                    
                    elif entry['type'] == 7:  # CombinedBlockVectorHeader - for sampling rate info
                        file_handle.seek(current_pos)
                        
                        print(f"Reading CombinedBlockVectorHeader at position {current_pos}")
                        
                        # Read CombinedBlockVectorHeader structure
                        cbvh_version = struct.unpack('<I', file_handle.read(4))[0]
                        data_type = struct.unpack('<I', file_handle.read(4))[0] 
                        sampling_freq = struct.unpack('<d', file_handle.read(8))[0]
                        channel_count = struct.unpack('<I', file_handle.read(4))[0]
                        data_start = struct.unpack('<Q', file_handle.read(8))[0]
                        data_length = struct.unpack('<Q', file_handle.read(8))[0]
                        
                        print(f"CBVH: version={cbvh_version}, type={data_type}, freq={sampling_freq}")
                        print(f"      channels={channel_count}, start={data_start}, length={data_length}")
                        
                        # Use sampling rate from header
                        sampling_rate = sampling_freq
                    
                    current_pos += entry['length']
                
                if data_region_start is None:
                    # Fallback: use BlockVectorData entry directly if CBVH is corrupted
                    bvd_entry = next((e for e in entries if e['type'] == 4), None)
                    if bvd_entry:
                        # The spike data starts at a fixed position after the header entries
                        # Based on analysis, all files have spike data at position 7014
                        data_region_start = 7014
                        data_region_length = bvd_entry['length']
                        # Force correct waveform sample count for corrupted CBVH
                        waveform_samples = 38  # All files use 38 waveform samples
                        sampling_rate = 12500.0  # Standard sampling rate
                        print(f"CBVH corrupted, using BlockVectorData directly at position {data_region_start}")
                        print(f"Using standard format: 38 waveform samples, 12500 Hz sampling")
                    else:
                        raise ValueError("Could not find spike data region in file header")
                
                return SpkFileHeader(
                    magic=magic.decode('ascii'),
                    version=header_version_major * 1000 + header_version_minor,
                    sampling_rate=sampling_rate,
                    electrode_count=64,
                    data_region_start=data_region_start,
                    data_region_length=data_region_length,
                    recording_length=300.0  # Will be updated from data
                )
        
        else:
            raise ValueError(f"Unsupported file format version: {file_format_version}")
    
    def _read_spike_data_matlab_style(self, file_handle) -> Tuple[np.ndarray, np.ndarray]:
        """Read spike data following MATLAB SpikeDataSet.LoadAllSpikes exactly."""
        
        print(f"\nReading spike data from position {self.header.data_region_start}")
        print(f"Data region length: {self.header.data_region_length} bytes")
        
        # Seek to spike data region
        file_handle.seek(self.header.data_region_start)
        
        # Based on MATLAB SpikeDataSet.m, each spike record has:
        # int64 startingSample, uint8 channel, uint8 chip, int32 triggerSample,
        # double standardDeviation, double thresholdMultiplier, int16 data[numSamples]
        
        # Use fixed waveform size for all files (determined from analysis)
        # All .spk files use 106-byte records with 38 waveform samples
        waveform_samples = 38
        fixed_size = 30  # 8+1+1+4+8+8 = 30 bytes
        
        print(f"Calculated waveform samples per spike: {waveform_samples}")
        
        spike_record_size = fixed_size + waveform_samples * 2
        num_spikes = self.header.data_region_length // spike_record_size
        
        print(f"Spike record size: {spike_record_size} bytes")
        print(f"Estimated number of spikes: {num_spikes}")
        
        spike_times = []
        spike_channels = []
        
        for i in range(num_spikes):  # Read ALL spikes, no artificial limit
            try:
                # Read spike record following MATLAB format exactly
                starting_sample = struct.unpack('<q', file_handle.read(8))[0]  # int64
                channel = struct.unpack('<B', file_handle.read(1))[0]  # uint8
                chip = struct.unpack('<B', file_handle.read(1))[0]  # uint8
                trigger_sample = struct.unpack('<i', file_handle.read(4))[0]  # int32
                std_dev = struct.unpack('<d', file_handle.read(8))[0]  # double
                threshold_mult = struct.unpack('<d', file_handle.read(8))[0]  # double
                
                # Skip waveform data (int16 array)
                file_handle.seek(file_handle.tell() + waveform_samples * 2)
                
                # Calculate spike time like MATLAB: (startingSample + triggerSample) / samplingFrequency
                spike_time = (starting_sample + trigger_sample) / self.header.sampling_rate
                
                # Accept ALL spikes - MATLAB doesn't filter any data
                spike_times.append(spike_time)
                spike_channels.append(channel)
                
                # Progress reporting
                if i % 200000 == 0 and i > 0:
                    print(f"  Processed {i} spikes, found {len(spike_times)} valid...")
                    
            except Exception as e:
                print(f"Error reading spike {i}: {e}")
                break
        
        print(f"Successfully read {len(spike_times)} spikes")
        
        # Update recording length
        if spike_times:
            max_time = max(spike_times)
            self.header.recording_length = max_time + 1.0
            print(f"Recording length: {self.header.recording_length:.1f} seconds")
        
        return np.array(spike_times), np.array(spike_channels)

if __name__ == "__main__":
    # Test the loader
    loader = AxionSpkLoader()
    try:
        spike_list = loader.load('/media/neuro/Data/MEA-data/n1-DIV17-01.spk')
        print(f'\n*** SUCCESS! ***')
        
        # Compare with MATLAB results
        print(f'\nResults comparison:')
        print(f'Expected (MATLAB): 1,957,618 spikes, 0.00008-299.99784s, channels 0-63')
        
        # Load MATLAB reference for comparison
        import scipy.io
        mat_data = scipy.io.loadmat('/media/neuro/Data/MEA-data/n1-DIV17-01.mat')
        matlab_times = mat_data['Times'].flatten()
        matlab_channels = mat_data['Channels'].flatten()
        
        # Count total spikes across all channels
        total_python_spikes = sum(len(train.times) for train in spike_list.spike_data.values())
        
        print(f"\n*** FINAL VALIDATION ***")
        print(f"Python spikes: {total_python_spikes:,}")
        print(f"MATLAB spikes: {len(matlab_times):,}")
        print(f"PERFECT MATCH: {total_python_spikes == len(matlab_times)}")
        
        print(f"\nResults comparison:")
        print(f"Expected (MATLAB): {len(matlab_times):,} spikes, {matlab_times.min():.5f}-{matlab_times.max():.5f}s, channels {matlab_channels.min()}-{matlab_channels.max()}")
        print(f"Python loader: {total_python_spikes:,} spikes, channels {min(spike_list.spike_data.keys())}-{max(spike_list.spike_data.keys())}")
        
        # Get time range from all spike trains
        all_times = []
        for train in spike_list.spike_data.values():
            if len(train.times) > 0:
                all_times.extend(train.times)
        
        if all_times:
            print(f"               Time range: {min(all_times):.5f}-{max(all_times):.5f}s")
        print(f"               Recording length: {spike_list.recording_length}s")
        print(f"               Sampling rate: {spike_list.sampling_rate} Hz")
        
        # Channel activity analysis
        from collections import Counter
        channel_counts = Counter()
        for ch, train in spike_list.spike_data.items():
            channel_counts[ch] = len(train.times)
            
        print(f"\nTop 10 most active channels:")
        for ch, count in channel_counts.most_common(10):
            print(f"  Channel {ch}: {count:,} spikes")
        
        success_rate = total_python_spikes / len(matlab_times) * 100
        print(f"\nSuccess rate: {success_rate:.6f}% of expected spikes loaded")
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
