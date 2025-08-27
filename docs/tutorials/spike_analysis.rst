Spike Train Analysis
===================

This tutorial covers detailed spike train analysis techniques available in MEA-Flow, including firing rate calculations, inter-spike interval (ISI) analysis, and spike pattern detection.

Understanding Spike Trains
--------------------------

A **spike train** is a sequence of action potentials (spikes) recorded from a single electrode over time. Key properties include:

- **Spike times**: Precise timing of each action potential
- **Firing rate**: Average frequency of spikes
- **ISI patterns**: Intervals between consecutive spikes
- **Regularity**: Consistency of spike timing
- **Burst patterns**: Periods of high-frequency activity

Basic Spike Analysis
-------------------

Setting Up
~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging

    from mea_flow.data import SpikeData, MEARecording
    from mea_flow.analysis import ActivityAnalyzer
    from mea_flow.visualization import ActivityPlotter

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

Creating Test Data
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def create_diverse_spike_data():
        """Create spike data with different activity patterns."""
        np.random.seed(42)
        
        patterns = {
            'regular': lambda t, rate: t + 1.0/rate,  # Regular intervals
            'poisson': lambda t, rate: t + np.random.exponential(1.0/rate),  # Random
            'bursting': lambda t, rate: generate_bursting_pattern(t, rate),  # Bursty
            'adapting': lambda t, rate: generate_adapting_pattern(t, rate),  # Rate adaptation
        }
        
        spike_trains = {}
        duration = 300.0
        sampling_rate = 20000.0
        
        for pattern_name, pattern_func in patterns.items():
            for rep in range(2):  # 2 channels per pattern
                channel_name = f"{pattern_name}_{rep:02d}"
                
                spike_times = []
                t = 0.0
                base_rate = np.random.uniform(2.0, 8.0)
                
                while t < duration:
                    if pattern_name == 'bursting':
                        t = generate_bursting_pattern(t, base_rate)
                    elif pattern_name == 'adapting':
                        t = generate_adapting_pattern(t, base_rate, len(spike_times))
                    else:
                        t = pattern_func(t, base_rate)
                    
                    if t < duration:
                        spike_times.append(t)
                
                spike_times = np.array(spike_times)
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
            metadata={"patterns": list(patterns.keys())}
        )

    def generate_bursting_pattern(t, rate):
        """Generate bursting spike pattern."""
        if np.random.random() < 0.02:  # 2% chance of burst
            # Generate burst
            burst_rate = rate * 10
            return t + np.random.exponential(1.0 / burst_rate)
        else:
            # Regular activity
            return t + np.random.exponential(1.0 / rate)

    def generate_adapting_pattern(t, initial_rate, spike_count):
        """Generate spike pattern with rate adaptation."""
        # Rate decreases with spike count (adaptation)
        adapted_rate = initial_rate * np.exp(-spike_count * 0.001)
        return t + np.random.exponential(1.0 / max(adapted_rate, 0.1))

    # Create the test data
    recording = create_diverse_spike_data()
    print(f"Created recording with {len(recording.spike_trains)} channels")

Firing Rate Analysis
-------------------

Basic Firing Rates
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    analyzer = ActivityAnalyzer()

    # Calculate basic firing rates
    print("\\nFiring Rate Analysis:")
    print("-" * 30)

    firing_rates = {}
    for channel_id, spike_data in recording.spike_trains.items():
        firing_rate = analyzer.calculate_firing_rate(spike_data)
        firing_rates[channel_id] = firing_rate
        
        pattern = channel_id.split('_')[0]
        print(f"{channel_id:12s} ({pattern:8s}): {firing_rate:6.2f} Hz")

Time-Varying Firing Rates
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def calculate_instantaneous_rate(spike_times, window_size=10.0, step_size=1.0):
        """Calculate instantaneous firing rate over time."""
        
        max_time = spike_times[-1] if len(spike_times) > 0 else 0
        time_points = np.arange(0, max_time, step_size)
        rates = []
        
        for t in time_points:
            # Count spikes in window around time point
            window_start = t - window_size / 2
            window_end = t + window_size / 2
            
            spikes_in_window = np.sum(
                (spike_times >= window_start) & (spike_times < window_end)
            )
            
            rate = spikes_in_window / window_size
            rates.append(rate)
        
        return time_points, np.array(rates)

    # Calculate time-varying rates for each channel
    print("\\nTime-Varying Rate Analysis:")
    print("-" * 35)

    for channel_id, spike_data in recording.spike_trains.items():
        time_points, rates = calculate_instantaneous_rate(
            spike_data.spike_times, window_size=30.0, step_size=5.0
        )
        
        mean_rate = np.mean(rates)
        std_rate = np.std(rates)
        cv_rate = std_rate / mean_rate if mean_rate > 0 else 0
        
        print(f"{channel_id}: mean={mean_rate:.2f} Hz, std={std_rate:.2f}, CV={cv_rate:.2f}")

Inter-Spike Interval Analysis
----------------------------

Basic ISI Statistics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    print("\\nInter-Spike Interval Analysis:")
    print("-" * 35)

    isi_results = {}
    for channel_id, spike_data in recording.spike_trains.items():
        if len(spike_data.spike_times) > 1:
            isi_stats = analyzer.calculate_isi_statistics(spike_data)
            isi_results[channel_id] = isi_stats
            
            pattern = channel_id.split('_')[0]
            print(f"\\n{channel_id} ({pattern}):")
            print(f"  Mean ISI:     {isi_stats['mean']:.3f} s")
            print(f"  Std ISI:      {isi_stats['std']:.3f} s")
            print(f"  CV:           {isi_stats['cv']:.3f}")
            print(f"  Median ISI:   {isi_stats['median']:.3f} s")
            print(f"  Min ISI:      {isi_stats['min']:.3f} s")
            print(f"  Max ISI:      {isi_stats['max']:.3f} s")

ISI Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def analyze_isi_distribution(spike_times):
        """Analyze ISI distribution characteristics."""
        
        if len(spike_times) < 2:
            return None
        
        isis = np.diff(spike_times)
        
        # Basic statistics
        results = {
            'isis': isis,
            'mean': np.mean(isis),
            'std': np.std(isis),
            'cv': np.std(isis) / np.mean(isis),
            'skewness': calculate_skewness(isis),
            'kurtosis': calculate_kurtosis(isis)
        }
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            results[f'p{p}'] = np.percentile(isis, p)
        
        # Mode approximation (most common ISI range)
        hist, bin_edges = np.histogram(isis, bins=50)
        mode_idx = np.argmax(hist)
        results['mode_approx'] = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        
        return results

    def calculate_skewness(data):
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def calculate_kurtosis(data):
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    # Analyze ISI distributions
    print("\\nISI Distribution Characteristics:")
    print("-" * 40)

    for channel_id, spike_data in recording.spike_trains.items():
        dist_results = analyze_isi_distribution(spike_data.spike_times)
        
        if dist_results:
            pattern = channel_id.split('_')[0]
            print(f"\\n{channel_id} ({pattern}):")
            print(f"  Skewness:     {dist_results['skewness']:.3f}")
            print(f"  Kurtosis:     {dist_results['kurtosis']:.3f}")
            print(f"  Mode (approx): {dist_results['mode_approx']:.3f} s")
            print(f"  P5-P95 range: {dist_results['p5']:.3f} - {dist_results['p95']:.3f} s")

Advanced Spike Pattern Analysis
------------------------------

Regularity Measures
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def calculate_advanced_regularity(spike_times):
        """Calculate advanced regularity measures."""
        
        if len(spike_times) < 3:
            return None
        
        isis = np.diff(spike_times)
        
        results = {}
        
        # Local variation (LV)
        lv_values = []
        for i in range(len(isis) - 1):
            if isis[i] + isis[i + 1] > 0:
                lv = 3 * (isis[i] - isis[i + 1])**2 / (isis[i] + isis[i + 1])**2
                lv_values.append(lv)
        
        results['local_variation'] = np.mean(lv_values) if lv_values else 0
        
        # Revised local variation (LvR)
        lvr_values = []
        for i in range(len(isis) - 1):
            if isis[i] + isis[i + 1] > 0:
                lvr = 3 * (isis[i] - isis[i + 1])**2 / (isis[i] + isis[i + 1])**2
                lvr *= (1 - 4 * abs(isis[i] - isis[i + 1]) / (isis[i] + isis[i + 1]))
                lvr_values.append(max(0, lvr))
        
        results['lvr'] = np.mean(lvr_values) if lvr_values else 0
        
        # Fano factor (requires binned spike counts)
        bin_size = 1.0  # 1 second bins
        max_time = spike_times[-1]
        n_bins = int(np.ceil(max_time / bin_size))
        
        spike_counts = np.histogram(spike_times, bins=n_bins, range=(0, n_bins * bin_size))[0]
        
        if np.mean(spike_counts) > 0:
            results['fano_factor'] = np.var(spike_counts) / np.mean(spike_counts)
        else:
            results['fano_factor'] = 0
        
        return results

    # Calculate advanced regularity measures
    print("\\nAdvanced Regularity Analysis:")
    print("-" * 35)

    for channel_id, spike_data in recording.spike_trains.items():
        reg_results = calculate_advanced_regularity(spike_data.spike_times)
        
        if reg_results:
            pattern = channel_id.split('_')[0]
            print(f"\\n{channel_id} ({pattern}):")
            print(f"  Local Variation (LV):  {reg_results['local_variation']:.3f}")
            print(f"  Revised LV (LvR):      {reg_results['lvr']:.3f}")
            print(f"  Fano Factor:           {reg_results['fano_factor']:.3f}")

Spike Pattern Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def classify_spike_pattern(spike_times, cv_threshold=0.5, lv_threshold=0.5):
        """Classify spike pattern based on regularity measures."""
        
        if len(spike_times) < 10:
            return "insufficient_data"
        
        isis = np.diff(spike_times)
        cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else float('inf')
        
        # Calculate local variation
        reg_results = calculate_advanced_regularity(spike_times)
        lv = reg_results['local_variation'] if reg_results else float('inf')
        
        # Classification rules
        if cv < cv_threshold and lv < lv_threshold:
            return "regular"
        elif cv > 1.5 or lv > 1.5:
            return "irregular_bursty"  
        elif cv > cv_threshold or lv > lv_threshold:
            return "irregular"
        else:
            return "semi_regular"

    # Classify all spike patterns
    print("\\nSpike Pattern Classification:")
    print("-" * 35)

    pattern_classifications = {}
    for channel_id, spike_data in recording.spike_trains.items():
        classification = classify_spike_pattern(spike_data.spike_times)
        pattern_classifications[channel_id] = classification
        
        true_pattern = channel_id.split('_')[0]
        print(f"{channel_id:12s}: {classification:15s} (true: {true_pattern})")

Visualization of Spike Analysis
------------------------------

Creating Comprehensive Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create output directory
    output_dir = Path("spike_analysis_output")
    output_dir.mkdir(exist_ok=True)

    # 1. ISI histograms by pattern
    patterns = list(set(ch.split('_')[0] for ch in recording.spike_trains.keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, pattern in enumerate(patterns[:4]):
        ax = axes[i]
        
        # Get all channels for this pattern
        pattern_channels = [ch for ch in recording.spike_trains.keys() 
                          if ch.startswith(pattern)]
        
        all_isis = []
        for ch in pattern_channels:
            spike_times = recording.spike_trains[ch].spike_times
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                all_isis.extend(isis)
        
        if all_isis:
            ax.hist(all_isis, bins=50, alpha=0.7, density=True)
            ax.set_xlabel('Inter-Spike Interval (s)')
            ax.set_ylabel('Density')
            ax.set_title(f'{pattern.title()} Pattern ISI Distribution')
            ax.set_xlim(0, np.percentile(all_isis, 95))  # Show 95th percentile
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'isi_distributions_by_pattern.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Rate vs Regularity scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for channel_id, spike_data in recording.spike_trains.items():
        firing_rate = firing_rates[channel_id]
        
        if len(spike_data.spike_times) > 1:
            isis = np.diff(spike_data.spike_times)
            cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
            
            pattern = channel_id.split('_')[0]
            color = {'regular': 'blue', 'poisson': 'green', 
                    'bursting': 'red', 'adapting': 'orange'}[pattern]
            
            ax.scatter(firing_rate, cv, c=color, label=pattern, 
                      s=60, alpha=0.7, edgecolors='black')
    
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Coefficient of Variation (CV)')
    ax.set_title('Firing Rate vs Regularity')
    ax.grid(True, alpha=0.3)
    
    # Add legend (unique patterns only)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, unique_labels)
    
    plt.savefig(output_dir / 'rate_vs_regularity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Detailed raster plot with ISI color coding
    fig, axes = plt.subplots(len(recording.spike_trains), 1, 
                           figsize=(15, 2 * len(recording.spike_trains)), 
                           sharex=True)
    
    if len(recording.spike_trains) == 1:
        axes = [axes]
    
    for i, (channel_id, spike_data) in enumerate(recording.spike_trains.items()):
        spike_times = spike_data.spike_times
        
        if len(spike_times) > 1:
            # Color spikes by preceding ISI
            isis = np.diff(spike_times)
            colors = plt.cm.viridis(np.log10(isis) / np.log10(np.max(isis)))
            
            # Plot spikes (skip first spike since it has no preceding ISI)
            for j in range(1, len(spike_times)):
                axes[i].scatter(spike_times[j], i, c=[colors[j-1]], s=2, alpha=0.8)
            
            # Plot first spike in black
            axes[i].scatter(spike_times[0], i, c='black', s=2)
        else:
            # Single spike or no spikes
            axes[i].scatter(spike_times, [i] * len(spike_times), c='black', s=2)
        
        pattern = channel_id.split('_')[0]
        axes[i].set_ylabel(f'{channel_id}\\n({pattern})', rotation=0, ha='right', va='center')
        axes[i].set_ylim(i-0.4, i+0.4)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title('Spike Raster Plot (Color = ISI Duration)')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02)
    cbar.set_label('Log ISI Duration', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'isi_colored_raster.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\\nSpike analysis visualizations saved to: {output_dir.absolute()}")

Summary and Interpretation
-------------------------

Pattern Interpretation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    print("\\n" + "="*60)
    print("SPIKE PATTERN ANALYSIS SUMMARY")
    print("="*60)

    # Summarize results by true pattern
    for pattern in patterns:
        pattern_channels = [ch for ch in recording.spike_trains.keys() 
                          if ch.startswith(pattern)]
        
        print(f"\\n{pattern.upper()} PATTERN:")
        print("-" * 20)
        
        rates = [firing_rates[ch] for ch in pattern_channels]
        classifications = [pattern_classifications[ch] for ch in pattern_channels]
        
        print(f"  Channels: {len(pattern_channels)}")
        print(f"  Firing rate: {np.mean(rates):.2f} ± {np.std(rates):.2f} Hz")
        print(f"  Classifications: {', '.join(set(classifications))}")
        
        # Calculate average CV for this pattern
        cvs = []
        for ch in pattern_channels:
            spike_times = recording.spike_trains[ch].spike_times
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
                cvs.append(cv)
        
        if cvs:
            print(f"  Average CV: {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")

    print("\\n" + "="*60)

**Understanding CV Values:**

- **CV < 0.5**: Regular firing (clock-like)
- **CV ≈ 1.0**: Poisson-like (random) firing  
- **CV > 1.5**: Irregular/bursty firing

**Understanding LV Values:**

- **LV < 0.5**: Local regularity (consistent ISIs)
- **LV > 1.5**: High local variability (burst-like)

Next Steps
----------

- Learn about :doc:`burst_detection` for analyzing burst patterns in detail
- Explore :doc:`cross_condition_comparison` for statistical comparisons
- Try :doc:`manifold_learning` for population-level analysis

The complete spike analysis code is available as part of the examples in the repository.