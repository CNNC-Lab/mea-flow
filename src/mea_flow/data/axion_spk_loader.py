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
                self.header = self._read_header(f)
                
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
    
    def _read_header(self, file_handle) -> SpkFileHeader:
        """
        Read and parse the .spk file header.
        
        Based on reverse engineering of Axion file format:
        - File signature/magic number
        - Version information
        - Electrode configuration
        - Sampling parameters
        """
        # Read file signature (first 4 bytes)
        signature = file_handle.read(4)
        
        # Check for known Axion signatures
        if signature not in [b'AXIS', b'SPK\x00', b'\x00SPK']:
            warnings.warn(f"Unknown file signature: {signature}. Attempting to parse anyway.")
        
        # Read version (4 bytes, little endian unsigned int)
        version_bytes = file_handle.read(4)
        if len(version_bytes) < 4:
            raise ValueError("Incomplete header: version field")
        version = struct.unpack('<I', version_bytes)[0]
        
        # Read number of electrodes (4 bytes)
        num_elec_bytes = file_handle.read(4)
        if len(num_elec_bytes) < 4:
            raise ValueError("Incomplete header: electrode count")
        num_electrodes = struct.unpack('<I', num_elec_bytes)[0]
        
        # Read sampling rate (8 bytes, double)
        sampling_rate_bytes = file_handle.read(8)
        if len(sampling_rate_bytes) < 8:
            raise ValueError("Incomplete header: sampling rate")
        sampling_rate = struct.unpack('<d', sampling_rate_bytes)[0]
        
        # Read recording length (8 bytes, double)
        rec_length_bytes = file_handle.read(8)
        if len(rec_length_bytes) < 8:
            raise ValueError("Incomplete header: recording length")
        recording_length = struct.unpack('<d', rec_length_bytes)[0]
        
        # Read electrode mapping (if available)
        electrode_mapping = {}
        
        # Try to read electrode mapping table
        # Format: electrode_id (4 bytes), well_row (4 bytes), well_col (4 bytes)
        try:
            for i in range(num_electrodes):
                elec_data = file_handle.read(12)  # 3 * 4 bytes
                if len(elec_data) < 12:
                    # Use default mapping if electrode table incomplete
                    electrode_mapping = self._generate_default_mapping(num_electrodes)
                    break
                    
                elec_id, well_row, well_col = struct.unpack('<III', elec_data)
                electrode_mapping[elec_id] = (well_row, well_col)
                
        except struct.error:
            # Fallback to default mapping
            electrode_mapping = self._generate_default_mapping(num_electrodes)
        
        return SpkFileHeader(
            version=version,
            num_electrodes=num_electrodes,
            sampling_rate=sampling_rate,
            recording_length=recording_length,
            electrode_mapping=electrode_mapping
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
        Read spike timing and electrode data from the file.
        
        Expected format after header:
        - Number of spikes (4 bytes)
        - For each spike:
          - Timestamp (8 bytes, double)
          - Electrode ID (4 bytes, unsigned int)
          - Optional: Waveform data (variable length)
        """
        # Read number of spikes
        num_spikes_bytes = file_handle.read(4)
        if len(num_spikes_bytes) < 4:
            raise ValueError("Cannot read spike count")
        
        num_spikes = struct.unpack('<I', num_spikes_bytes)[0]
        
        if num_spikes == 0:
            return np.array([]), np.array([])
        
        # Initialize arrays
        spike_times = np.zeros(num_spikes)
        electrode_ids = np.zeros(num_spikes, dtype=int)
        
        # Read spike data
        for i in range(num_spikes):
            try:
                # Read timestamp (8 bytes, double)
                time_bytes = file_handle.read(8)
                if len(time_bytes) < 8:
                    warnings.warn(f"Incomplete spike data at spike {i}")
                    break
                
                timestamp = struct.unpack('<d', time_bytes)[0]
                
                # Read electrode ID (4 bytes, unsigned int)
                elec_bytes = file_handle.read(4)
                if len(elec_bytes) < 4:
                    warnings.warn(f"Incomplete electrode data at spike {i}")
                    break
                
                electrode_id = struct.unpack('<I', elec_bytes)[0]
                
                # Convert timestamp from samples to seconds
                spike_times[i] = timestamp / self.header.sampling_rate
                electrode_ids[i] = electrode_id
                
                # Skip waveform data if present (variable length)
                # This is a simplified implementation - waveform parsing would
                # require more detailed format specification
                
            except struct.error as e:
                warnings.warn(f"Error reading spike {i}: {e}")
                break
        
        # Trim arrays to actual data read
        if i < num_spikes - 1:
            spike_times = spike_times[:i+1]
            electrode_ids = electrode_ids[:i+1]
        
        return spike_times, electrode_ids
    
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
