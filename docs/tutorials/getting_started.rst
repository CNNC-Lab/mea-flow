Getting Started with MEA-Flow
=============================

This tutorial provides a gentle introduction to MEA-Flow, covering the basic concepts and workflows for analyzing multi-electrode array (MEA) data.

What is MEA Data?
-----------------

Multi-electrode arrays (MEAs) are devices that record neural activity from multiple electrodes simultaneously. They consist of:

- **Electrodes**: Individual recording sites that detect neural spikes
- **Wells**: Compartments containing neural cultures (for multi-well plates)
- **Channels**: Data streams from each electrode
- **Spike Trains**: Sequences of action potentials recorded over time

MEA-Flow is designed to handle this hierarchical structure efficiently.

Basic Concepts
--------------

Core Data Structures
~~~~~~~~~~~~~~~~~~~~

MEA-Flow uses several key data structures:

.. code-block:: python

    from mea_flow.data import SpikeData, MEARecording

    # SpikeData: Represents spike times from a single electrode
    spike_data = SpikeData(
        spike_times=[0.1, 0.5, 1.2, 1.8],  # Times in seconds
        spike_samples=[2000, 10000, 24000, 36000],  # Sample indices
        channel_id="Channel_01",
        sampling_rate=20000.0  # Hz
    )

    # MEARecording: Collection of spike trains from multiple electrodes
    recording = MEARecording(
        spike_trains={"Channel_01": spike_data},
        duration=300.0,  # seconds
        sampling_rate=20000.0,
        metadata={"experiment": "test"}
    )

Analysis Modules
~~~~~~~~~~~~~~~~

MEA-Flow provides several analysis modules:

- **ActivityAnalyzer**: Basic activity metrics (firing rates, ISI analysis)
- **BurstDetector**: Detect individual and network bursts
- **ManifoldAnalyzer**: Dimensionality reduction and population dynamics
- **Visualization**: Plotting tools for all analysis types

Your First Analysis
-------------------

Let's walk through a complete basic analysis workflow.

Step 1: Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Import MEA-Flow modules
    from mea_flow.data import SpikeData, MEARecording
    from mea_flow.analysis import ActivityAnalyzer, BurstDetector
    from mea_flow.visualization import ActivityPlotter

Step 2: Create or Load Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this tutorial, we'll create synthetic data:

.. code-block:: python

    def create_sample_data():
        """Create synthetic MEA data for demonstration."""
        np.random.seed(42)  # For reproducible results
        
        spike_trains = {}
        duration = 300.0  # 5 minutes
        sampling_rate = 20000.0
        
        # Create data for 16 channels (4x4 electrode grid)
        for channel_id in range(16):
            channel_name = f"Channel_{channel_id:02d}"
            
            # Generate realistic spike times
            base_rate = np.random.uniform(1.0, 8.0)  # 1-8 Hz baseline
            spike_times = []
            t = 0.0
            
            while t < duration:
                # Add some variability in firing
                current_rate = base_rate * (1 + 0.3 * np.sin(2 * np.pi * t / 60))
                t += np.random.exponential(1.0 / current_rate)
                if t < duration:
                    spike_times.append(t)
            
            # Convert to numpy array and samples
            spike_times = np.array(spike_times)
            spike_samples = (spike_times * sampling_rate).astype(int)
            
            # Create SpikeData object
            spike_trains[channel_name] = SpikeData(
                spike_times=spike_times,
                spike_samples=spike_samples,
                channel_id=channel_name,
                sampling_rate=sampling_rate
            )
        
        # Create MEA recording
        recording = MEARecording(
            spike_trains=spike_trains,
            duration=duration,
            sampling_rate=sampling_rate,
            metadata={"created_by": "getting_started_tutorial"}
        )
        
        return recording

    # Create the sample data
    recording = create_sample_data()
    print(f"Created recording with {len(recording.spike_trains)} channels")
    print(f"Duration: {recording.duration} seconds")

Step 3: Basic Activity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now let's analyze the basic activity patterns:

.. code-block:: python

    # Initialize the analyzer
    analyzer = ActivityAnalyzer()

    # Calculate firing rates for each channel
    firing_rates = {}
    for channel_id, spike_data in recording.spike_trains.items():
        firing_rate = analyzer.calculate_firing_rate(spike_data)
        firing_rates[channel_id] = firing_rate
        print(f"{channel_id}: {firing_rate:.2f} Hz")

    # Calculate overall statistics
    all_rates = list(firing_rates.values())
    print(f"\nOverall firing rate: {np.mean(all_rates):.2f} ± {np.std(all_rates):.2f} Hz")
    print(f"Active channels (>0.5 Hz): {sum(1 for rate in all_rates if rate > 0.5)}")

Step 4: ISI (Inter-Spike Interval) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze the timing patterns of spikes:

.. code-block:: python

    # Analyze ISI statistics for each channel
    isi_stats = {}
    for channel_id, spike_data in recording.spike_trains.items():
        if len(spike_data.spike_times) > 1:
            isis = analyzer.calculate_isi_statistics(spike_data)
            isi_stats[channel_id] = isis
            print(f"{channel_id} - Mean ISI: {isis['mean']:.3f}s, "
                  f"CV: {isis['cv']:.2f}")

Step 5: Burst Detection
~~~~~~~~~~~~~~~~~~~~~~~

Detect burst events in the spike data:

.. code-block:: python

    # Initialize burst detector
    burst_detector = BurstDetector()

    # Detect bursts for each channel
    all_bursts = {}
    for channel_id, spike_data in recording.spike_trains.items():
        try:
            bursts = burst_detector.detect_bursts(spike_data)
            all_bursts[channel_id] = bursts
            
            if bursts:
                burst_rate = len(bursts) / recording.duration * 60  # per minute
                avg_duration = np.mean([b.duration for b in bursts])
                print(f"{channel_id}: {len(bursts)} bursts "
                      f"({burst_rate:.1f}/min, avg duration: {avg_duration:.2f}s)")
        except Exception as e:
            print(f"Burst detection failed for {channel_id}: {e}")
            all_bursts[channel_id] = []

    # Detect network bursts (coordinated activity across channels)
    try:
        spike_trains_list = list(recording.spike_trains.values())
        network_bursts = burst_detector.detect_network_bursts(spike_trains_list)
        if network_bursts:
            print(f"\nDetected {len(network_bursts)} network bursts")
            for i, nb in enumerate(network_bursts[:3]):  # Show first 3
                print(f"  Network burst {i+1}: {nb.start_time:.2f}-{nb.end_time:.2f}s, "
                      f"{len(nb.participating_channels)} channels")
    except Exception as e:
        print(f"Network burst detection failed: {e}")

Step 6: Basic Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create plots to visualize your results:

.. code-block:: python

    # Create output directory
    output_dir = Path("tutorial_output")
    output_dir.mkdir(exist_ok=True)

    # 1. Firing rate histogram
    plt.figure(figsize=(10, 6))
    plt.hist(list(firing_rates.values()), bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Number of Channels')
    plt.title('Distribution of Firing Rates Across Channels')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'firing_rate_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Raster plot of first few channels
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    for i, (channel_id, spike_data) in enumerate(list(recording.spike_trains.items())[:4]):
        spike_times = spike_data.spike_times
        y_positions = np.ones_like(spike_times) * i
        
        axes[i].scatter(spike_times, y_positions, s=1, alpha=0.7)
        axes[i].set_ylabel(channel_id)
        axes[i].set_ylim(i-0.4, i+0.4)
        axes[i].grid(True, alpha=0.3)
        
        # Highlight detected bursts if any
        if channel_id in all_bursts and all_bursts[channel_id]:
            for burst in all_bursts[channel_id]:
                axes[i].axvspan(burst.start_time, burst.end_time, 
                              alpha=0.3, color='red', label='Burst')
    
    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title('Spike Raster Plot (First 4 Channels)')
    plt.tight_layout()
    plt.savefig(output_dir / 'raster_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\\nPlots saved to {output_dir.absolute()}")

Step 7: Summary Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a summary of your analysis:

.. code-block:: python

    # Calculate summary statistics
    n_channels = len(recording.spike_trains)
    total_spikes = sum(len(spike_data.spike_times) 
                      for spike_data in recording.spike_trains.values())
    
    active_channels = sum(1 for rate in firing_rates.values() if rate > 0.1)
    bursting_channels = sum(1 for bursts in all_bursts.values() if len(bursts) > 0)
    
    print("\\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Recording duration: {recording.duration:.0f} seconds")
    print(f"Total channels: {n_channels}")
    print(f"Active channels (>0.1 Hz): {active_channels} ({100*active_channels/n_channels:.1f}%)")
    print(f"Total spikes: {total_spikes:,}")
    print(f"Overall spike rate: {total_spikes/recording.duration:.1f} spikes/second")
    print(f"Mean firing rate: {np.mean(list(firing_rates.values())):.2f} Hz")
    print(f"Bursting channels: {bursting_channels} ({100*bursting_channels/n_channels:.1f}%)")
    print("="*50)

Understanding the Results
-------------------------

Interpreting Firing Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Low activity (0-1 Hz)**: Minimal or sparse activity
- **Moderate activity (1-5 Hz)**: Typical for healthy cultures
- **High activity (>10 Hz)**: Very active, may indicate stimulation or pathology

Interpreting ISI Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Mean ISI**: Average time between spikes (inverse of firing rate)
- **CV (Coefficient of Variation)**: Regularity of firing
  
  - CV < 0.5: Regular firing
  - CV ≈ 1.0: Poisson-like (random) firing
  - CV > 1.5: Irregular or bursty firing

Interpreting Bursts
~~~~~~~~~~~~~~~~~~~

- **Individual bursts**: Periods of high-frequency firing within single channels
- **Network bursts**: Coordinated activity across multiple channels
- **Burst rate**: Frequency of burst events (typically 0.1-5 per minute)

Next Steps
----------

Now that you understand the basics, you can:

1. Try the analysis with your own MEA data
2. Explore :doc:`spike_analysis` for more detailed spike train analysis
3. Learn about :doc:`burst_detection` for advanced burst analysis
4. Check :doc:`manifold_learning` for population dynamics analysis

The complete code for this tutorial is available as ``examples/basic_analysis.py`` in the MEA-Flow repository.

Common Issues
-------------

**Import Errors**
   Make sure MEA-Flow is properly installed: ``pip install -e .``

**Empty Results**
   Check that your data has the correct format and contains actual spikes

**Memory Issues**
   For large datasets, consider analyzing subsets or reducing sampling rate

**Plotting Issues**
   Ensure matplotlib is installed: ``pip install matplotlib``