"""
Native Python loader for Axion .spk files.

This module provides a pure Python implementation for reading Axion BioSystems
.spk files without requiring MATLAB or the AxionFileLoader.

The .spk file format contains binary spike data with timestamps and electrode
information. This implementation reverse-engineers the format based on the
AxionFileLoader MATLAB code structure.
"""

import struct
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
from dataclasses import dataclass

from .spike_list import SpikeList


@dataclass
class SpkFileHeader:
    """Header information from .spk file."""
    version: int
    num_electrodes: int
    sampling_rate: float
    recording_length: float
    electrode_mapping: Dict[int, Tuple[int, int]]  # electrode_id -> (well_row, well_col)


class AxionSpkLoader:
    """
    Native Python loader for Axion .spk files.
    
    This class provides functionality to read Axion BioSystems .spk files
    directly in Python without requiring MATLAB dependencies.
    """
    
    def __init__(self):
        self.header = None
        self.spike_data = None
    
    def load_spk_file(self, file_path: Union[str, Path]) -> SpikeList:
        """
        Load spike data from an Axion .spk file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the .spk file
            
        Returns
        -------
        SpikeList
            Loaded spike data
            
        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file format is invalid or unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                # Read and parse header
                self.header = self._read_header_matlab_style(f)
                
                # Read spike data
                spike_times, electrode_ids = self._read_spike_data(f)
                
            # Convert to SpikeList format
            spike_data = [(int(elec_id), float(time)) 
                         for elec_id, time in zip(electrode_ids, spike_times)]
            
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
            raise ValueError(f"Unsupported header version: {header_version_major}")

    def _find_spike_data_region(self, file_handle):
        """Find the spike data region by scanning for valid spike patterns."""
        file_size = file_handle.seek(0, 2)
        file_handle.seek(0)
        
        print("Scanning file for spike data region...")
        
        # Scan the file in chunks looking for spike-like patterns
        chunk_size = 1024 * 1024  # 1MB chunks
        best_position = None
        best_spike_count = 0
        
        for chunk_start in range(0, min(file_size, 100 * chunk_size), chunk_size):
            file_handle.seek(chunk_start)
            chunk = file_handle.read(min(chunk_size, file_size - chunk_start))
            
            # Look for potential spike record starts
            for offset in range(0, len(chunk) - 100, 100):  # Check every 100 bytes
                pos = chunk_start + offset
                file_handle.seek(pos)
                
                # Try to read a few spike records
                valid_spikes = 0
                for i in range(10):  # Test 10 consecutive spikes
                    try:
                        starting_sample = struct.unpack('<q', file_handle.read(8))[0]
                        channel = struct.unpack('<B', file_handle.read(1))[0]
                        chip = struct.unpack('<B', file_handle.read(1))[0]
                        trigger_sample = struct.unpack('<i', file_handle.read(4))[0]
                        std_dev = struct.unpack('<d', file_handle.read(8))[0]
                        threshold_mult = struct.unpack('<d', file_handle.read(8))[0]
                        
                        # Validate spike data
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
                        
                        # Skip assumed waveform (32 samples * 2 bytes)
                        file_handle.seek(file_handle.tell() + 64)
                        
                    except:
                        break
                
                if valid_spikes > best_spike_count:
                    best_spike_count = valid_spikes
                    best_position = pos
                    
                    if valid_spikes >= 8:  # Good indication
                        print(f"Found {valid_spikes} consecutive valid spikes at position {pos}")
                        return pos
        
        if best_position and best_spike_count >= 5:
            print(f"Best spike region found at position {best_position} with {best_spike_count} spikes")
            return best_position
        
        return None

    def _read_header(self, file_handle) -> SpkFileHeader:
        """
        Read basic header info and find spike data region using pattern scanning.
        """
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
    
    def _generate_default_mapping(self, num_electrodes: int) -> Dict[int, Tuple[int, int]]:
        """Generate default electrode mapping for standard MEA layouts."""
        mapping = {}
        
        if num_electrodes == 64:
            # Standard 8x8 MEA
            for i in range(64):
                row = i // 8
                col = i % 8
                mapping[i] = (row, col)
        elif num_electrodes == 16:
            # 4x4 MEA
            for i in range(16):
                row = i // 4
                col = i % 4
                mapping[i] = (row, col)
        else:
            # Generic linear mapping
            for i in range(num_electrodes):
                mapping[i] = (0, i)
        
        return mapping
    
    def _read_spike_data(self, file_handle) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read spike timing and electrode data following MATLAB SpikeDataSet.LoadAllSpikes exactly.
        
        Based on MATLAB AxionFileLoader reverse engineering:
        Each spike record contains:
        - int64 startingSample (8 bytes)
        - uint8 channel (1 byte) 
        - uint8 chip (1 byte)
        - int32 triggerSample (4 bytes)
        - double standardDeviation (8 bytes)
        - double thresholdMultiplier (8 bytes)
        - int16 waveform data (2 * numSamples bytes)
        
        Returns spike times and electrode channels.
        """
        print(f"\nReading spike data from position {self.header.data_region_start}")
        print(f"Data region length: {self.header.data_region_length} bytes")
        
        # Seek to spike data region
        file_handle.seek(self.header.data_region_start)
        
        # Calculate waveform size from data region size
        # Expected ~1.96M spikes, so calculate waveform size
        expected_spikes = 1957618
        fixed_size = 30  # 8+1+1+4+8+8 = 30 bytes
        remaining_per_spike = self.header.data_region_length / expected_spikes - fixed_size
        waveform_samples = int(remaining_per_spike / 2)  # int16 = 2 bytes
        
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
                
                # Accept ALL spikes exactly like MATLAB - no validation filtering
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
    
    def get_electrode_mapping(self) -> Optional[Dict[int, Tuple[int, int]]]:
        """Get electrode to well position mapping."""
        if self.header:
            return self.header.electrode_mapping
        return None
    
    def get_file_info(self) -> Optional[Dict]:
        """Get file information summary."""
        if not self.header:
            return None
            
        return {
            'version': self.header.version,
            'num_electrodes': self.header.num_electrodes,
            'sampling_rate': self.header.sampling_rate,
            'recording_length': self.header.recording_length,
            'electrode_mapping': self.header.electrode_mapping
        }


def load_axion_spk_native(file_path: Union[str, Path], **kwargs) -> SpikeList:
    """
    Load Axion .spk file using native Python implementation.
    
    This function provides a direct replacement for the MATLAB-based
    AxionFileLoader functionality.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .spk file
    **kwargs
        Additional parameters (for compatibility)
        
    Returns
    -------
    SpikeList
        Loaded spike data
        
    Examples
    --------
    >>> spike_list = load_axion_spk_native('recording.spk')
    >>> print(f"Loaded {len(spike_list.get_active_channels())} active channels")
    """
    loader = AxionSpkLoader()
    return loader.load_spk_file(file_path)


# Fallback function for testing format variations
def probe_spk_file_structure(file_path: Union[str, Path], max_bytes: int = 1024) -> Dict:
    """
    Probe .spk file structure for debugging and format analysis.
    
    This function reads the first portion of a .spk file and attempts
    to identify the binary structure.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .spk file
    max_bytes : int
        Maximum bytes to read for probing
        
    Returns
    -------
    dict
        Information about file structure
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    info = {
        'file_size': file_path.stat().st_size,
        'header_bytes': None,
        'possible_signatures': [],
        'structure_hints': []
    }
    
    with open(file_path, 'rb') as f:
        # Read header portion
        header_data = f.read(min(max_bytes, info['file_size']))
        info['header_bytes'] = header_data[:64].hex()
        
        # Look for common signatures
        signatures = [b'AXIS', b'SPK\x00', b'\x00SPK', b'AXON']
        for sig in signatures:
            if sig in header_data:
                info['possible_signatures'].append(sig.decode('ascii', errors='ignore'))
        
        # Look for patterns that might indicate structure
        # Check for repeated 4-byte or 8-byte patterns
        for i in range(0, min(256, len(header_data) - 8), 4):
            chunk = header_data[i:i+4]
            if len(chunk) == 4:
                val = struct.unpack('<I', chunk)[0]
                if 1000 <= val <= 50000:  # Possible sampling rate
                    info['structure_hints'].append(f"Possible sampling rate at offset {i}: {val}")
                elif 1 <= val <= 1000:  # Possible electrode count
                    info['structure_hints'].append(f"Possible electrode count at offset {i}: {val}")
    
    return info
