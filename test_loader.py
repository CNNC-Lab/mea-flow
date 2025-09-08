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
        
        # Sort times and remove duplicates
        self.times = np.unique(self.times)
        
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
    """Native Python loader for Axion BioSystems .spk files."""
    
    def __init__(self):
        self.header: Optional[SpkFileHeader] = None
    
    def load(self, file_path: str) -> SpikeList:
        """Load spike data from .spk file."""
        try:
            with open(file_path, 'rb') as f:
                # Read header
                self.header = self._read_header(f)
                
                # Read spike data
                spike_times, spike_channels = self._read_spike_data(f)
                
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
    
    def _find_spike_data_region(self, file_handle):
        """Find the spike data region by scanning for valid spike patterns."""
        file_size = file_handle.seek(0, 2)
        file_handle.seek(0)
        
        print("Scanning file for spike data region...")
        
        # Try multiple scanning strategies
        candidates = []
        
        # Strategy 1: Look for regions with high variance (likely spike timestamps)
        print("  Strategy 1: Scanning for high variance regions...")
        chunk_size = 1024
        for pos in range(0, min(file_size, 1000000), chunk_size):
            file_handle.seek(pos)
            data = file_handle.read(chunk_size)
            if len(data) >= 64:
                # Interpret as int64 values and check variance
                values = []
                for i in range(0, len(data) - 7, 8):
                    try:
                        val = struct.unpack('<q', data[i:i+8])[0]
                        if 0 < val < 50000000:  # Plausible sample range
                            values.append(val)
                    except:
                        continue
                
                if len(values) >= 5:
                    variance = np.var(values)
                    if variance > 1000000:  # High variance indicates timestamps
                        candidates.append((pos, variance, len(values)))
        
        # Sort candidates by variance
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy 2: Test top candidates for valid spike patterns
        print(f"  Strategy 2: Testing {min(len(candidates), 10)} high-variance regions...")
        for pos, variance, count in candidates[:10]:
            # Test different alignments within this region
            for offset in range(0, min(1000, file_size - pos), 1):
                test_pos = pos + offset
                file_handle.seek(test_pos)
                
                valid_spikes = 0
                try:
                    for i in range(20):  # Test more spikes
                        starting_sample = struct.unpack('<q', file_handle.read(8))[0]
                        channel = struct.unpack('<B', file_handle.read(1))[0]
                        chip = struct.unpack('<B', file_handle.read(1))[0]
                        trigger_sample = struct.unpack('<i', file_handle.read(4))[0]
                        std_dev = struct.unpack('<d', file_handle.read(8))[0]
                        threshold_mult = struct.unpack('<d', file_handle.read(8))[0]
                        
                        # More lenient validation
                        if (0 <= channel <= 255 and  # Allow more channels
                            0 <= starting_sample < 100000000 and  # Wider range
                            abs(trigger_sample) < 10000 and  # More lenient
                            0 <= std_dev <= 10000 and  # More lenient
                            0 <= threshold_mult <= 1000):  # More lenient
                            
                            spike_time = (starting_sample + trigger_sample) / 12500.0
                            if 0 <= spike_time <= 10000:  # Wider time range
                                valid_spikes += 1
                            else:
                                break
                        else:
                            break
                        
                        # Skip waveform (try different sizes)
                        file_handle.seek(file_handle.tell() + 64)
                        
                except:
                    continue
                
                if valid_spikes >= 15:  # Found good region
                    print(f"  Found {valid_spikes} consecutive valid spikes at position {test_pos}")
                    return test_pos
                elif valid_spikes >= 10:  # Potential candidate
                    print(f"  Found {valid_spikes} spikes at position {test_pos} (candidate)")
        
        # Strategy 3: Brute force scan with byte-level precision
        print("  Strategy 3: Byte-level scanning...")
        best_position = None
        best_spike_count = 0
        
        for pos in range(0, min(file_size, 200000), 1):  # Scan first 200KB byte by byte
            file_handle.seek(pos)
            
            valid_spikes = 0
            try:
                for i in range(5):  # Test fewer spikes for speed
                    starting_sample = struct.unpack('<q', file_handle.read(8))[0]
                    channel = struct.unpack('<B', file_handle.read(1))[0]
                    chip = struct.unpack('<B', file_handle.read(1))[0]
                    trigger_sample = struct.unpack('<i', file_handle.read(4))[0]
                    std_dev = struct.unpack('<d', file_handle.read(8))[0]
                    threshold_mult = struct.unpack('<d', file_handle.read(8))[0]
                    
                    if (0 <= channel <= 63 and 
                        0 <= starting_sample < 50000000 and
                        abs(trigger_sample) < 1000 and
                        0 <= std_dev <= 1000 and
                        0 <= threshold_mult <= 100):
                        
                        spike_time = (starting_sample + trigger_sample) / 12500.0
                        if 0 <= spike_time <= 1000:
                            valid_spikes += 1
                        else:
                            break
                    else:
                        break
                    
                    file_handle.seek(file_handle.tell() + 64)
                    
            except:
                continue
            
            if valid_spikes > best_spike_count:
                best_spike_count = valid_spikes
                best_position = pos
                
                if valid_spikes >= 5:
                    print(f"  Found {valid_spikes} consecutive valid spikes at position {pos}")
                    return pos
        
        if best_position and best_spike_count >= 3:
            print(f"Best spike region found at position {best_position} with {best_spike_count} spikes")
            return best_position
        
        return None

    def _read_header(self, file_handle) -> SpkFileHeader:
        """Read basic header info and find spike data region using pattern scanning."""
        file_handle.seek(0)
        
        # Read magic word
        magic = file_handle.read(8)
        if magic != b'AxionBio':
            raise ValueError(f"Invalid magic word: {magic}")
        
        # Read version
        version = struct.unpack('<I', file_handle.read(4))[0]
        
        print(f"Header: magic={magic}, version={version}")
        
        # Find spike data region by scanning
        data_region_start = self._find_spike_data_region(file_handle)
        
        if data_region_start is None:
            raise ValueError("Could not find spike data region in file")
        
        # Calculate approximate data region length
        file_size = file_handle.seek(0, 2)
        data_region_length = file_size - data_region_start
        
        return SpkFileHeader(
            magic=magic.decode('ascii'),
            version=version,
            sampling_rate=12500.0,  # Standard Axion sampling rate
            electrode_count=64,  # Standard 64-electrode array
            data_region_start=data_region_start,
            data_region_length=data_region_length,
            recording_length=1000.0  # Will be calculated from actual data
        )
    
    def _read_spike_data(self, file_handle) -> Tuple[np.ndarray, np.ndarray]:
        """Read spike timing and electrode data from the discovered spike data region."""
        # Try different waveform sizes to find the correct one
        waveform_sizes_to_try = [32, 26, 40, 50, 64]  # Common sizes
        
        best_result = None
        best_count = 0
        
        for waveform_samples in waveform_sizes_to_try:
            print(f"\nTrying waveform size: {waveform_samples} samples")
            
            # Seek to spike data region
            file_handle.seek(self.header.data_region_start)
            
            spike_times = []
            spike_channels = []
            
            spike_record_size = 30 + waveform_samples * 2
            
            # Expected about 1.96M spikes based on MATLAB data
            expected_spikes = 1957618
            expected_file_size = expected_spikes * spike_record_size
            
            print(f"  Spike record size: {spike_record_size} bytes")
            print(f"  Expected file size for {expected_spikes} spikes: {expected_file_size} bytes")
            print(f"  Actual remaining bytes: {self.header.data_region_length}")
            
            # Calculate number of spikes
            num_spikes = min(self.header.data_region_length // spike_record_size, 2000000)
            
            spikes_read = 0
            consecutive_failures = 0
            
            for i in range(num_spikes):
                try:
                    # Read spike record
                    starting_sample = struct.unpack('<q', file_handle.read(8))[0]
                    channel = struct.unpack('<B', file_handle.read(1))[0]
                    chip = struct.unpack('<B', file_handle.read(1))[0]
                    trigger_sample = struct.unpack('<i', file_handle.read(4))[0]
                    std_dev = struct.unpack('<d', file_handle.read(8))[0]
                    threshold_mult = struct.unpack('<d', file_handle.read(8))[0]
                    
                    # Skip waveform data
                    file_handle.seek(file_handle.tell() + waveform_samples * 2)
                    
                    # Validate spike data based on MATLAB expectations
                    if (0 <= channel <= 63 and 
                        0 <= starting_sample < 4000000 and  # ~300s * 12500 Hz
                        abs(trigger_sample) < 1000 and
                        0 <= std_dev <= 1000 and
                        0 <= threshold_mult <= 100):
                        
                        # Calculate spike time: (startingSample + triggerSample) / samplingFrequency
                        spike_time = (starting_sample + trigger_sample) / self.header.sampling_rate
                        
                        if 0 <= spike_time <= 300:  # Expected ~300 second recording
                            spike_times.append(spike_time)
                            spike_channels.append(channel)
                            spikes_read += 1
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                    else:
                        consecutive_failures += 1
                    
                    # Stop if too many consecutive failures
                    if consecutive_failures > 1000:
                        break
                    
                    # Progress reporting
                    if spikes_read % 100000 == 0 and spikes_read > 0:
                        print(f"    Processed {spikes_read} valid spikes...")
                        
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures > 100:
                        break
            
            print(f"  Found {len(spike_times)} spikes with waveform size {waveform_samples}")
            
            # Keep the best result
            if len(spike_times) > best_count:
                best_count = len(spike_times)
                best_result = (np.array(spike_times), np.array(spike_channels), waveform_samples)
            
            # If we found a lot of spikes, this is probably correct
            if len(spike_times) > 100000:
                print(f"  Good result with {len(spike_times)} spikes!")
                break
        
        if best_result is None:
            return np.array([]), np.array([])
        
        spike_times, spike_channels, best_waveform_size = best_result
        
        print(f"\nBest result: {len(spike_times)} spikes with waveform size {best_waveform_size}")
        
        # Update recording length based on actual data
        if len(spike_times) > 0:
            max_time = max(spike_times)
            self.header.recording_length = max_time + 1.0
            print(f"Updated recording length to {self.header.recording_length:.1f} seconds")
        
        return spike_times, spike_channels

if __name__ == "__main__":
    # Test the loader
    loader = AxionSpkLoader()
    try:
        spike_list = loader.load('/media/neuro/Data/MEA-data/n1-DIV17-01.spk')
        print(f'\nSUCCESS! Loaded spike data')
        
        # Show some statistics
        if spike_list.spike_data:
            times = [st.times for st in spike_list.spike_data.values()]
            all_times = [t for sublist in times for t in sublist]
            channels = list(spike_list.spike_data.keys())
            
            print(f'Channels with spikes: {len(channels)}')
            print(f'Total spikes: {len(all_times)}')
            if all_times:
                print(f'Time range: {min(all_times):.3f} - {max(all_times):.3f} seconds')
            print(f'Recording length: {spike_list.recording_length:.1f} seconds')
            print(f'Sampling rate: {spike_list.sampling_rate} Hz')
            
            # Show spikes per channel
            spikes_per_channel = {ch: len(st.times) for ch, st in spike_list.spike_data.items()}
            active_channels = [(ch, count) for ch, count in spikes_per_channel.items() if count > 0]
            active_channels.sort(key=lambda x: x[1], reverse=True)
            
            print(f'\nTop 10 most active channels:')
            for ch, count in active_channels[:10]:
                print(f'  Channel {ch}: {count} spikes')
        else:
            print('No spikes found!')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
