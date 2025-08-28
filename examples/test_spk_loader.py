#!/usr/bin/env python3
"""
Test script for the native Axion .spk loader.

This script demonstrates how to use the new Python-based .spk file loader
and provides utilities for testing and debugging .spk file formats.
"""

import sys
from pathlib import Path
import logging

# Add the src directory to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mea_flow.data import load_axion_spk_native, probe_spk_file_structure, load_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_spk_loader(spk_file_path: str):
    """
    Test the native .spk loader with a real file.
    
    Parameters
    ----------
    spk_file_path : str
        Path to the .spk file to test
    """
    spk_path = Path(spk_file_path)
    
    if not spk_path.exists():
        logger.error(f"File not found: {spk_path}")
        return
    
    logger.info(f"Testing .spk loader with file: {spk_path}")
    
    try:
        # First, probe the file structure
        logger.info("Probing file structure...")
        structure_info = probe_spk_file_structure(spk_path)
        
        logger.info("File structure analysis:")
        for key, value in structure_info.items():
            logger.info(f"  {key}: {value}")
        
        # Attempt to load the file
        logger.info("Attempting to load .spk file...")
        spike_list = load_axion_spk_native(spk_path)
        
        # Display results
        logger.info("Successfully loaded .spk file!")
        logger.info(f"  Active channels: {len(spike_list.get_active_channels())}")
        logger.info(f"  Recording length: {spike_list.recording_length:.2f} seconds")
        logger.info(f"  Sampling rate: {spike_list.sampling_rate} Hz")
        
        # Show spike statistics
        total_spikes = sum(len(train.spike_times) for train in spike_list.spike_trains.values())
        logger.info(f"  Total spikes: {total_spikes}")
        
        if total_spikes > 0:
            all_times = []
            for train in spike_list.spike_trains.values():
                all_times.extend(train.spike_times)
            
            logger.info(f"  Spike time range: {min(all_times):.3f} - {max(all_times):.3f} seconds")
        
        return spike_list
        
    except Exception as e:
        logger.error(f"Failed to load .spk file: {e}")
        logger.info("This might be due to:")
        logger.info("  1. Unknown .spk file format variant")
        logger.info("  2. Corrupted file")
        logger.info("  3. Different Axion software version")
        logger.info("  4. Non-standard electrode configuration")
        return None


def compare_with_matlab_conversion(spk_file_path: str):
    """
    Compare native loader results with MATLAB conversion (if available).
    
    Parameters
    ----------
    spk_file_path : str
        Path to the .spk file
    """
    spk_path = Path(spk_file_path)
    mat_path = spk_path.with_suffix('.mat')
    
    logger.info("Comparing native loader with MATLAB conversion...")
    
    # Load with native loader
    try:
        native_result = load_axion_spk_native(spk_path)
        logger.info("Native loader: SUCCESS")
        native_spikes = sum(len(train.spike_times) for train in native_result.spike_trains.values())
        logger.info(f"  Native spikes: {native_spikes}")
    except Exception as e:
        logger.error(f"Native loader: FAILED - {e}")
        native_result = None
        native_spikes = 0
    
    # Load with MATLAB conversion (if available)
    if mat_path.exists():
        try:
            from mea_flow.data import load_matlab_file
            matlab_result = load_matlab_file(mat_path)
            logger.info("MATLAB conversion: SUCCESS")
            matlab_spikes = sum(len(train.spike_times) for train in matlab_result.spike_trains.values())
            logger.info(f"  MATLAB spikes: {matlab_spikes}")
            
            if native_result and matlab_result:
                logger.info(f"Spike count difference: {abs(native_spikes - matlab_spikes)}")
                
        except Exception as e:
            logger.error(f"MATLAB conversion: FAILED - {e}")
    else:
        logger.info("No MATLAB conversion file (.mat) found for comparison")


def create_test_data():
    """Create a simple test .spk file for development purposes."""
    logger.info("Creating test .spk file...")
    
    # This would create a minimal .spk file for testing
    # Implementation would depend on understanding the exact format
    test_path = Path("examples/output/test_data.spk")
    test_path.parent.mkdir(exist_ok=True)
    
    logger.info(f"Test file would be created at: {test_path}")
    logger.info("Note: Test file creation requires complete format specification")


def main():
    """Main function for testing the .spk loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Axion .spk file loader")
    parser.add_argument("spk_file", nargs='?', help="Path to .spk file to test")
    parser.add_argument("--probe-only", action="store_true", 
                       help="Only probe file structure, don't attempt loading")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with MATLAB conversion if available")
    parser.add_argument("--create-test", action="store_true",
                       help="Create test data file")
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_data()
        return
    
    if not args.spk_file:
        logger.error("Please provide a .spk file path or use --create-test")
        logger.info("Usage: python test_spk_loader.py <file.spk>")
        logger.info("       python test_spk_loader.py --create-test")
        return
    
    if args.probe_only:
        try:
            structure_info = probe_spk_file_structure(args.spk_file)
            logger.info("File structure analysis:")
            for key, value in structure_info.items():
                logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Failed to probe file: {e}")
    else:
        spike_list = test_spk_loader(args.spk_file)
        
        if args.compare and spike_list:
            compare_with_matlab_conversion(args.spk_file)


if __name__ == "__main__":
    main()
