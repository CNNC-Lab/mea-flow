#!/usr/bin/env python3
"""
Axion .spk Loader Demo

This script demonstrates the native Python .spk file loader capabilities
and provides examples of how to use it in place of MATLAB dependencies.
"""

import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mea_flow.data import load_data, load_axion_spk_native, probe_spk_file_structure
    from mea_flow.analysis import MEAMetrics
    from mea_flow.visualization import MEAPlotter
except ImportError as e:
    logger.error(f"Failed to import MEA-Flow modules: {e}")
    logger.error("Please ensure MEA-Flow is installed: pip install -e .")
    exit(1)


def create_synthetic_spk_file(output_path: str = "examples/output/demo.spk"):
    """
    Create a synthetic .spk file for demonstration purposes.
    
    This creates a minimal binary file that follows the expected .spk format
    for testing the native loader.
    """
    import struct
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating synthetic .spk file: {output_path}")
    
    # Synthetic parameters
    num_electrodes = 16
    sampling_rate = 12500.0
    recording_length = 60.0  # 60 seconds
    
    # Generate synthetic spike data
    np.random.seed(42)
    spike_times = []
    electrode_ids = []
    
    for elec_id in range(num_electrodes):
        # Generate random spike times for this electrode
        n_spikes = np.random.poisson(recording_length * 2)  # ~2 Hz average
        times = np.sort(np.random.uniform(0, recording_length, n_spikes))
        
        spike_times.extend(times)
        electrode_ids.extend([elec_id] * n_spikes)
    
    # Sort by time
    sorted_indices = np.argsort(spike_times)
    spike_times = np.array(spike_times)[sorted_indices]
    electrode_ids = np.array(electrode_ids)[sorted_indices]
    
    # Convert times to samples
    spike_samples = (spike_times * sampling_rate).astype(np.uint64)
    
    logger.info(f"Generated {len(spike_times)} spikes across {num_electrodes} electrodes")
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'AXIS')  # File signature
        f.write(struct.pack('<I', 1))  # Version
        f.write(struct.pack('<I', num_electrodes))  # Number of electrodes
        f.write(struct.pack('<d', sampling_rate))  # Sampling rate
        f.write(struct.pack('<d', recording_length))  # Recording length
        
        # Write electrode mapping (simplified)
        for elec_id in range(num_electrodes):
            f.write(struct.pack('<I', elec_id))  # Electrode ID
            f.write(struct.pack('<I', elec_id // 4))  # Well row (4x4 layout)
            f.write(struct.pack('<I', elec_id % 4))   # Well column
        
        # Write spike data
        f.write(struct.pack('<I', len(spike_times)))  # Number of spikes
        
        for i in range(len(spike_times)):
            f.write(struct.pack('<d', float(spike_samples[i])))  # Timestamp (samples)
            f.write(struct.pack('<I', electrode_ids[i]))  # Electrode ID
    
    logger.info(f"Synthetic .spk file created successfully")
    return output_path


def demo_spk_loading():
    """Demonstrate .spk file loading capabilities."""
    logger.info("=== Axion .spk Loader Demo ===")
    
    # Create synthetic test file
    spk_path = create_synthetic_spk_file()
    
    # Demo 1: Probe file structure
    logger.info("\n1. Probing file structure...")
    try:
        structure_info = probe_spk_file_structure(spk_path)
        logger.info("File structure analysis:")
        for key, value in structure_info.items():
            if key == 'header_bytes':
                logger.info(f"  {key}: {value[:32]}...")  # Show first 16 bytes
            else:
                logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.error(f"Structure probing failed: {e}")
    
    # Demo 2: Load with native loader
    logger.info("\n2. Loading with native Python loader...")
    try:
        spike_list = load_axion_spk_native(spk_path)
        
        logger.info("Successfully loaded .spk file!")
        logger.info(f"  Active channels: {len(spike_list.get_active_channels())}")
        logger.info(f"  Recording length: {spike_list.recording_length:.2f} seconds")
        logger.info(f"  Sampling rate: {spike_list.sampling_rate} Hz")
        
        # Calculate total spikes
        total_spikes = sum(len(train.spike_times) for train in spike_list.spike_trains.values())
        logger.info(f"  Total spikes: {total_spikes}")
        
        if total_spikes > 0:
            # Show per-channel statistics
            logger.info("  Per-channel spike counts:")
            for channel_id in sorted(spike_list.get_active_channels())[:8]:  # Show first 8
                count = len(spike_list.spike_trains[channel_id].spike_times)
                rate = count / spike_list.recording_length
                logger.info(f"    Channel {channel_id}: {count} spikes ({rate:.2f} Hz)")
            
            if len(spike_list.get_active_channels()) > 8:
                logger.info(f"    ... and {len(spike_list.get_active_channels()) - 8} more channels")
        
        return spike_list
        
    except Exception as e:
        logger.error(f"Native loader failed: {e}")
        return None


def demo_analysis_workflow(spike_list):
    """Demonstrate analysis workflow with loaded .spk data."""
    if not spike_list:
        logger.warning("No spike data available for analysis demo")
        return
    
    logger.info("\n3. Running analysis workflow...")
    
    try:
        # Compute basic metrics
        metrics = MEAMetrics()
        results_df = metrics.compute_all_metrics(spike_list, grouping='global')
        
        logger.info("Analysis results:")
        logger.info(f"  Network firing rate: {results_df.iloc[0]['mean_firing_rate']:.2f} Hz")
        logger.info(f"  Active electrodes: {len(spike_list.get_active_channels())}")
        
        # Create visualization
        plotter = MEAPlotter()
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        
        # Raster plot
        fig = plotter.plot_raster(spike_list, time_range=(0, 10))
        plotter.save_figure(fig, "examples/output/spk_demo_raster.png")
        logger.info("  Saved: examples/output/spk_demo_raster.png")
        
        logger.info("Analysis workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis workflow failed: {e}")


def demo_format_comparison():
    """Compare different loading methods."""
    logger.info("\n4. Format comparison demo...")
    
    spk_path = Path("examples/output/demo.spk")
    
    if not spk_path.exists():
        logger.warning("No .spk file found for comparison")
        return
    
    # Load with automatic format detection
    try:
        spike_list_auto = load_data(spk_path)
        total_spikes_auto = sum(len(train.spike_times) for train in spike_list_auto.spike_trains.values())
        logger.info(f"  Automatic loader: {total_spikes_auto} spikes")
    except Exception as e:
        logger.error(f"  Automatic loader failed: {e}")
    
    # Load with explicit native loader
    try:
        spike_list_native = load_axion_spk_native(spk_path)
        total_spikes_native = sum(len(train.spike_times) for train in spike_list_native.spike_trains.values())
        logger.info(f"  Native loader: {total_spikes_native} spikes")
    except Exception as e:
        logger.error(f"  Native loader failed: {e}")


def main():
    """Main demo function."""
    logger.info("Starting Axion .spk Loader Demo")
    
    # Run the demo
    spike_list = demo_spk_loading()
    demo_analysis_workflow(spike_list)
    demo_format_comparison()
    
    logger.info("\n=== Demo Complete ===")
    logger.info("Key benefits of the native .spk loader:")
    logger.info("  ✓ No MATLAB dependency required")
    logger.info("  ✓ Direct Python integration")
    logger.info("  ✓ Faster loading performance")
    logger.info("  ✓ Built-in format debugging tools")
    logger.info("  ✓ Seamless MEA-Flow integration")
    
    logger.info("\nNext steps:")
    logger.info("  • Test with your real .spk files")
    logger.info("  • Use probe_spk_file_structure() for unknown formats")
    logger.info("  • Report any format compatibility issues")


if __name__ == "__main__":
    main()
