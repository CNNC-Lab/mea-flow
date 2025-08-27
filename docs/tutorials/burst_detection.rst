Burst Detection in MEA Data
===========================

This tutorial covers burst detection techniques in MEA-Flow, from basic single-channel bursts to complex network-wide burst analysis.

Introduction to Bursts
----------------------

A **burst** is a period of high-frequency neural activity that stands out from baseline firing. In MEA recordings, we distinguish between:

- **Single-channel bursts**: High-frequency firing within individual electrodes
- **Network bursts**: Coordinated bursting activity across multiple electrodes
- **Population bursts**: Large-scale activation involving most active channels

Understanding burst patterns is crucial for:

- Assessing network connectivity
- Detecting pathological activity
- Measuring drug effects
- Characterizing developmental changes

The Envelope Algorithm
----------------------

MEA-Flow uses the **envelope algorithm**, a robust method for burst detection that:

1. Calculates a smoothed firing rate envelope
2. Identifies periods above a dynamic threshold
3. Applies duration and spike count criteria
4. Merges nearby burst events

This approach is less sensitive to noise than simple threshold methods.

Basic Burst Detection
---------------------

Setting Up
~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from mea_flow.data import SpikeData, MEARecording
    from mea_flow.analysis import BurstDetector, ActivityAnalyzer
    from mea_flow.visualization import ActivityPlotter

Creating Synthetic Data with Bursts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's create data with clear burst patterns:

.. code-block:: python

    def create_bursting_data():
        """Create synthetic MEA data with defined burst patterns."""
        np.random.seed(42)
        
        spike_trains = {}
        duration = 300.0  # 5 minutes
        sampling_rate = 20000.0
        
        for channel_id in range(8):
            channel_name = f"Channel_{channel_id:02d}"
            spike_times = []
            t = 0.0
            
            # Parameters for this channel
            baseline_rate = np.random.uniform(0.5, 2.0)
            burst_rate = np.random.uniform(20, 50)  # High rate during bursts
            burst_probability = 0.02  # 2% chance per time step
            
            while t < duration:
                if np.random.random() < burst_probability:
                    # Generate a burst
                    burst_duration = np.random.uniform(0.2, 1.0)  # 200ms to 1s
                    burst_end = t + burst_duration
                    
                    print(f"Creating burst in {channel_name} at {t:.1f}s "
                          f"(duration: {burst_duration:.2f}s)")
                    
                    # High-frequency spikes during burst
                    while t < burst_end and t < duration:
                        t += np.random.exponential(1.0 / burst_rate)
                        if t < duration:
                            spike_times.append(t)
                    
                    # Brief refractory period after burst
                    t += np.random.uniform(0.5, 2.0)
                else:
                    # Baseline activity
                    t += np.random.exponential(1.0 / baseline_rate)
                    if t < duration:
                        spike_times.append(t)
            
            # Create SpikeData
            spike_times = np.array(sorted(spike_times))
            spike_samples = (spike_times * sampling_rate).astype(int)
            
            spike_trains[channel_name] = SpikeData(
                spike_times=spike_times,
                spike_samples=spike_samples,
                channel_id=channel_name,
                sampling_rate=sampling_rate
            )
        
        return MEARecording(
            spike_trains=spike_trains,
            duration=duration,
            sampling_rate=sampling_rate,
            metadata={"type": "bursting_simulation"}
        )

    # Create the data
    recording = create_bursting_data()
    print(f"Created bursting data with {len(recording.spike_trains)} channels")

Basic Burst Detection
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize burst detector with default parameters
    burst_detector = BurstDetector()

    # Detect bursts for each channel
    channel_bursts = {}
    
    for channel_id, spike_data in recording.spike_trains.items():
        try:
            bursts = burst_detector.detect_bursts(spike_data)
            channel_bursts[channel_id] = bursts
            
            if bursts:
                print(f"\\n{channel_id}:")
                print(f"  Detected {len(bursts)} bursts")
                
                # Analyze burst properties
                durations = [b.duration for b in bursts]
                spike_counts = [len(b.spike_times) for b in bursts]
                inter_burst_intervals = []
                
                for i in range(len(bursts) - 1):
                    ibi = bursts[i + 1].start_time - bursts[i].end_time
                    inter_burst_intervals.append(ibi)
                
                print(f"  Average duration: {np.mean(durations):.2f}s")
                print(f"  Average spikes per burst: {np.mean(spike_counts):.1f}")
                print(f"  Burst rate: {len(bursts) / recording.duration * 60:.1f} per minute")
                
                if inter_burst_intervals:
                    print(f"  Average inter-burst interval: {np.mean(inter_burst_intervals):.1f}s")
                    
        except Exception as e:
            print(f"Burst detection failed for {channel_id}: {e}")
            channel_bursts[channel_id] = []

Customizing Burst Detection Parameters
--------------------------------------

The burst detector has several tunable parameters:

.. code-block:: python

    # Create burst detector with custom parameters
    custom_detector = BurstDetector(
        envelope_cutoff=5.0,      # Smoothing frequency (Hz) - lower = more smoothing
        threshold_factor=3.0,     # Threshold multiplier - higher = stricter
        min_burst_duration=0.1,   # Minimum burst duration (seconds)
        max_burst_duration=10.0,  # Maximum burst duration (seconds)
        min_spikes_in_burst=5,    # Minimum spikes required
        min_channels_in_network_burst=2,  # For network burst detection
        max_network_isi=0.1       # Maximum ISI for network bursts (seconds)
    )

    print("\\nTesting different parameter sets:")
    
    # Test with strict parameters (fewer, stronger bursts)
    strict_detector = BurstDetector(
        threshold_factor=5.0,     # Higher threshold
        min_spikes_in_burst=10    # More spikes required
    )
    
    # Test with lenient parameters (more, weaker bursts)
    lenient_detector = BurstDetector(
        threshold_factor=2.0,     # Lower threshold
        min_spikes_in_burst=3     # Fewer spikes required
    )
    
    # Compare results for one channel
    test_channel = "Channel_00"
    test_spike_data = recording.spike_trains[test_channel]
    
    default_bursts = burst_detector.detect_bursts(test_spike_data)
    strict_bursts = strict_detector.detect_bursts(test_spike_data)
    lenient_bursts = lenient_detector.detect_bursts(test_spike_data)
    
    print(f"\\n{test_channel} burst detection comparison:")
    print(f"  Default parameters: {len(default_bursts)} bursts")
    print(f"  Strict parameters:  {len(strict_bursts)} bursts")
    print(f"  Lenient parameters: {len(lenient_bursts)} bursts")

Network Burst Detection
-----------------------

Network bursts represent coordinated activity across multiple channels:

.. code-block:: python

    # Detect network bursts
    spike_trains_list = list(recording.spike_trains.values())
    
    try:
        network_bursts = burst_detector.detect_network_bursts(spike_trains_list)
        
        if network_bursts:
            print(f"\\nDetected {len(network_bursts)} network bursts:")
            
            for i, nb in enumerate(network_bursts):
                participating_channels = nb.participating_channels
                duration = nb.end_time - nb.start_time
                
                print(f"\\n  Network Burst {i+1}:")
                print(f"    Time: {nb.start_time:.2f} - {nb.end_time:.2f}s (duration: {duration:.2f}s)")
                print(f"    Participating channels: {len(participating_channels)}")
                print(f"    Channel list: {', '.join(participating_channels[:5])}"
                      f"{'...' if len(participating_channels) > 5 else ''}")
                
                # Calculate network burst properties
                all_spikes_in_burst = []
                for channel_name in participating_channels:
                    channel_spikes = recording.spike_trains[channel_name].spike_times
                    burst_spikes = channel_spikes[
                        (channel_spikes >= nb.start_time) & (channel_spikes <= nb.end_time)
                    ]
                    all_spikes_in_burst.extend(burst_spikes)
                
                total_spikes = len(all_spikes_in_burst)
                spike_rate = total_spikes / duration if duration > 0 else 0
                
                print(f"    Total spikes: {total_spikes}")
                print(f"    Network spike rate: {spike_rate:.1f} Hz")
        else:
            print("\\nNo network bursts detected")
            
    except Exception as e:
        print(f"Network burst detection failed: {e}")

Advanced Burst Analysis
-----------------------

Burst Synchronization Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_burst_synchronization(channel_bursts, recording):
        """Analyze temporal synchronization of bursts across channels."""
        
        # Collect all burst times
        all_burst_starts = []
        burst_channel_map = {}
        
        for channel_id, bursts in channel_bursts.items():
            for burst in bursts:
                all_burst_starts.append(burst.start_time)
                if burst.start_time not in burst_channel_map:
                    burst_channel_map[burst.start_time] = []
                burst_channel_map[burst.start_time].append(channel_id)
        
        # Find synchronized bursts (within 1 second of each other)
        sync_threshold = 1.0  # seconds
        synchronized_events = []
        
        all_burst_starts = sorted(all_burst_starts)
        
        i = 0
        while i < len(all_burst_starts):
            event_start = all_burst_starts[i]
            synchronized_channels = set([])
            
            # Find all bursts within sync_threshold
            j = i
            while (j < len(all_burst_starts) and 
                   all_burst_starts[j] - event_start <= sync_threshold):
                synchronized_channels.update(burst_channel_map[all_burst_starts[j]])
                j += 1
            
            if len(synchronized_channels) >= 2:  # At least 2 channels
                synchronized_events.append({
                    'time': event_start,
                    'channels': list(synchronized_channels),
                    'n_channels': len(synchronized_channels)
                })
            
            i = j if j > i else i + 1
        
        return synchronized_events

    # Analyze synchronization
    sync_events = analyze_burst_synchronization(channel_bursts, recording)

    print(f"\\nBurst Synchronization Analysis:")
    print(f"Found {len(sync_events)} synchronized burst events")

    if sync_events:
        # Sort by number of participating channels
        sync_events.sort(key=lambda x: x['n_channels'], reverse=True)
        
        print("\\nTop synchronized events:")
        for i, event in enumerate(sync_events[:3]):
            print(f"  Event {i+1}: {event['n_channels']} channels at {event['time']:.1f}s")
            print(f"    Channels: {', '.join(event['channels'])}")

Burst Feature Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def extract_burst_features(channel_bursts):
        """Extract comprehensive features from detected bursts."""
        
        features = {
            'durations': [],
            'spike_counts': [],
            'peak_rates': [],
            'channels': [],
            'start_times': []
        }
        
        for channel_id, bursts in channel_bursts.items():
            for burst in bursts:
                features['durations'].append(burst.duration)
                features['spike_counts'].append(len(burst.spike_times))
                features['channels'].append(channel_id)
                features['start_times'].append(burst.start_time)
                
                # Calculate peak firing rate within burst
                if len(burst.spike_times) > 1:
                    isis = np.diff(burst.spike_times)
                    min_isi = np.min(isis)
                    peak_rate = 1.0 / min_isi if min_isi > 0 else 0
                else:
                    peak_rate = 0
                
                features['peak_rates'].append(peak_rate)
        
        return features

    # Extract features
    burst_features = extract_burst_features(channel_bursts)

    if burst_features['durations']:
        print(f"\\nBurst Feature Summary:")
        print(f"  Total bursts detected: {len(burst_features['durations'])}")
        print(f"  Duration range: {np.min(burst_features['durations']):.2f} - "
              f"{np.max(burst_features['durations']):.2f}s")
        print(f"  Average duration: {np.mean(burst_features['durations']):.2f}s")
        print(f"  Spike count range: {np.min(burst_features['spike_counts'])} - "
              f"{np.max(burst_features['spike_counts'])}")
        print(f"  Average spikes per burst: {np.mean(burst_features['spike_counts']):.1f}")
        print(f"  Peak rate range: {np.min(burst_features['peak_rates']):.1f} - "
              f"{np.max(burst_features['peak_rates']):.1f} Hz")

Burst Visualization
-------------------

Creating Comprehensive Burst Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create output directory
    output_dir = Path("burst_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # 1. Burst raster plot
    fig, axes = plt.subplots(len(recording.spike_trains), 1, 
                           figsize=(15, 2 * len(recording.spike_trains)), 
                           sharex=True)
    
    if len(recording.spike_trains) == 1:
        axes = [axes]
    
    for i, (channel_id, spike_data) in enumerate(recording.spike_trains.items()):
        # Plot spikes
        spike_times = spike_data.spike_times
        y_pos = np.ones_like(spike_times) * i
        axes[i].scatter(spike_times, y_pos, s=1, c='black', alpha=0.6)
        
        # Highlight bursts
        if channel_id in channel_bursts:
            for burst in channel_bursts[channel_id]:
                axes[i].axvspan(burst.start_time, burst.end_time, 
                              alpha=0.3, color='red')
                # Mark burst center
                center = (burst.start_time + burst.end_time) / 2
                axes[i].plot(center, i, 'rv', markersize=4)
        
        axes[i].set_ylabel(channel_id, rotation=0, ha='right')
        axes[i].set_ylim(i-0.4, i+0.4)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title('Spike Raster with Detected Bursts (Red = Bursts)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'burst_raster_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Burst feature histograms
    if burst_features['durations']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Duration histogram
        axes[0, 0].hist(burst_features['durations'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Burst Duration (s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Burst Duration Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Spike count histogram
        axes[0, 1].hist(burst_features['spike_counts'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Spikes per Burst')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Spikes per Burst Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Peak rate histogram
        axes[1, 0].hist(burst_features['peak_rates'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Peak Rate (Hz)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Peak Rate Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Burst timing over recording
        axes[1, 1].hist(burst_features['start_times'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Number of Bursts')
        axes[1, 1].set_title('Burst Timing Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'burst_feature_histograms.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\\nVisualization saved to: {output_dir.absolute()}")

Troubleshooting Burst Detection
-------------------------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**No Bursts Detected**

.. code-block:: python

    # Check if data has sufficient activity
    analyzer = ActivityAnalyzer()
    spike_data = recording.spike_trains["Channel_00"]
    firing_rate = analyzer.calculate_firing_rate(spike_data)
    
    if firing_rate < 0.5:
        print("Low firing rate - try more lenient parameters")
        lenient_detector = BurstDetector(threshold_factor=1.5, min_spikes_in_burst=3)
    
    # Check spike density over time
    time_bins = np.arange(0, recording.duration, 1.0)  # 1-second bins
    spike_counts, _ = np.histogram(spike_data.spike_times, bins=time_bins)
    max_rate = np.max(spike_counts)
    
    if max_rate < 5:
        print("Sparse spiking - bursts may not be detectable")

**Too Many Bursts Detected**

.. code-block:: python

    # Use stricter parameters
    strict_detector = BurstDetector(
        threshold_factor=4.0,
        min_burst_duration=0.2,
        min_spikes_in_burst=8
    )

**Inconsistent Results Across Channels**

.. code-block:: python

    # Analyze per-channel firing rates
    for channel_id, spike_data in recording.spike_trains.items():
        firing_rate = analyzer.calculate_firing_rate(spike_data)
        print(f"{channel_id}: {firing_rate:.2f} Hz")
        
        # Use adaptive parameters based on firing rate
        if firing_rate > 5.0:
            # High rate - use stricter detection
            detector = BurstDetector(threshold_factor=3.5)
        else:
            # Low rate - use lenient detection
            detector = BurstDetector(threshold_factor=2.0)

Best Practices
--------------

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with default parameters** for initial exploration
2. **Adjust threshold_factor**:
   
   - Lower (1.5-2.5): More sensitive, detects weaker bursts
   - Higher (3.5-6.0): More selective, only strong bursts

3. **Tune minimum requirements**:
   
   - ``min_spikes_in_burst``: 3-5 for weak cultures, 8-15 for active cultures
   - ``min_burst_duration``: 0.05-0.2s depending on burst characteristics

4. **Validate results** by visual inspection of raster plots

Quality Control
~~~~~~~~~~~~~~~

.. code-block:: python

    def assess_burst_quality(channel_bursts, recording):
        """Assess quality of burst detection results."""
        
        total_bursts = sum(len(bursts) for bursts in channel_bursts.values())
        channels_with_bursts = sum(1 for bursts in channel_bursts.values() if len(bursts) > 0)
        
        if total_bursts == 0:
            return "No bursts detected - check parameters or data quality"
        
        if channels_with_bursts / len(channel_bursts) > 0.8:
            return "High detection rate - may be too sensitive"
        
        if channels_with_bursts / len(channel_bursts) < 0.1:
            return "Low detection rate - may be too strict"
        
        return "Detection appears reasonable"

    quality_assessment = assess_burst_quality(channel_bursts, recording)
    print(f"\\nQuality Assessment: {quality_assessment}")

Next Steps
----------

- Explore :doc:`cross_condition_comparison` to compare bursting across experimental conditions
- Learn :doc:`manifold_learning` for analyzing population dynamics during bursts
- Try :doc:`well_plate_experiments` for high-throughput burst analysis

The complete burst detection example is available as ``examples/burst_analysis.py``.