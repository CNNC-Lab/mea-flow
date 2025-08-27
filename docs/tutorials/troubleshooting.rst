Troubleshooting Guide
===================

This guide covers common issues you might encounter when using MEA-Flow and provides solutions and workarounds.

Installation Issues
------------------

PySpike Compilation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: PySpike fails to compile during installation.

**Symptoms**:
- Build errors mentioning C++ compilation
- "Microsoft Visual C++ 14.0 is required" on Windows
- "gcc" or "clang" compilation errors on Linux/macOS

**Solutions**:

1. **Use basic installation** (recommended):
   
   .. code-block:: bash
   
      uv pip install -e .  # Installs without PySpike
   
   MEA-Flow will work without PySpike using alternative distance measures.

2. **Install PySpike separately** (if needed):
   
   .. code-block:: bash
   
      # On Windows (install Visual Studio Build Tools first)
      pip install pyspike
      
      # On Linux (install build dependencies)
      sudo apt-get install python3-dev build-essential
      pip install pyspike
      
      # On macOS (install Xcode command line tools)
      xcode-select --install
      pip install pyspike

UMAP Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: UMAP-learn fails to install or causes conflicts.

**Solutions**:

1. **Install basic version first**:
   
   .. code-block:: bash
   
      uv pip install -e .
      pip install umap-learn

2. **Use alternative manifold methods**:
   
   .. code-block:: python
   
      # Use PCA, t-SNE, or MDS instead of UMAP
      manifold_analyzer = ManifoldAnalyzer()
      embedding = manifold_analyzer.apply_pca(data, n_components=2)

uv.lock Parse Errors
~~~~~~~~~~~~~~~~~~

**Problem**: "Failed to parse uv.lock" or lock file conflicts.

**Solutions**:

.. code-block:: bash

   # Remove lock file and regenerate
   rm uv.lock
   uv lock
   
   # Alternative: use pip directly
   pip install -e .

Import Errors
~~~~~~~~~~~~

**Problem**: Cannot import MEA-Flow modules.

**Check installation**:

.. code-block:: python

   try:
       import mea_flow
       print("MEA-Flow installed successfully!")
       print(f"Version: {mea_flow.__version__}")
   except ImportError as e:
       print(f"Import error: {e}")
       print("Try: pip install -e .")

**Check Python path**:

.. code-block:: bash

   # Make sure you're in the right environment
   which python
   python -c "import sys; print(sys.path)"

Data Loading Issues
------------------

File Format Problems
~~~~~~~~~~~~~~~~~~~

**Problem**: Cannot load MEA data files.

**Common file formats and solutions**:

1. **Axion .spk files**:
   
   .. code-block:: python
   
      # Requires MATLAB conversion (see documentation)
      # For now, convert to .mat format in MATLAB first
      
      from mea_flow.data import SpikeData
      # Load converted .mat file instead

2. **CSV format issues**:
   
   .. code-block:: python
   
      # Ensure CSV has correct columns: 'channel', 'time'
      import pandas as pd
      df = pd.read_csv('your_file.csv')
      print(df.columns)  # Check column names
      print(df.head())   # Check data format

3. **HDF5 access errors**:
   
   .. code-block:: bash
   
      # Install h5py if missing
      pip install h5py
   
   .. code-block:: python
   
      # Check HDF5 file structure
      import h5py
      with h5py.File('your_file.h5', 'r') as f:
          print(list(f.keys()))

Empty or Invalid Data
~~~~~~~~~~~~~~~~~~~~

**Problem**: No spikes detected or invalid spike times.

**Check data format**:

.. code-block:: python

   # Verify spike data
   print(f"Spike times range: {spike_times.min():.3f} - {spike_times.max():.3f}s")
   print(f"Number of spikes: {len(spike_times)}")
   print(f"Sampling rate: {sampling_rate} Hz")
   
   # Check for common issues
   if len(spike_times) == 0:
       print("WARNING: No spikes found")
   if np.any(spike_times < 0):
       print("WARNING: Negative spike times")
   if np.any(np.diff(spike_times) <= 0):
       print("WARNING: Spike times not sorted")

**Fix common issues**:

.. code-block:: python

   # Sort spike times
   spike_times = np.sort(spike_times)
   
   # Remove negative times
   spike_times = spike_times[spike_times >= 0]
   
   # Convert units if necessary
   if np.max(spike_times) > 10000:  # Probably in samples, not seconds
       spike_times = spike_times / sampling_rate

Analysis Issues
--------------

No Bursts Detected
~~~~~~~~~~~~~~~~~

**Problem**: Burst detection returns empty results.

**Diagnostic steps**:

.. code-block:: python

   from mea_flow.analysis import ActivityAnalyzer, BurstDetector
   
   analyzer = ActivityAnalyzer()
   
   # Check basic activity
   firing_rate = analyzer.calculate_firing_rate(spike_data)
   print(f"Firing rate: {firing_rate:.2f} Hz")
   
   if firing_rate < 0.5:
       print("Low firing rate - try lenient parameters")
   
   # Check spike density over time
   import matplotlib.pyplot as plt
   
   time_bins = np.arange(0, recording.duration, 1.0)
   spike_counts, _ = np.histogram(spike_data.spike_times, bins=time_bins)
   
   plt.figure(figsize=(10, 4))
   plt.plot(time_bins[:-1], spike_counts)
   plt.xlabel('Time (s)')
   plt.ylabel('Spikes per second')
   plt.title('Spike Density Over Time')
   plt.show()

**Solutions**:

1. **Use lenient parameters**:
   
   .. code-block:: python
   
      burst_detector = BurstDetector(
          threshold_factor=1.5,      # Lower threshold
          min_spikes_in_burst=3,     # Fewer spikes required
          min_burst_duration=0.05    # Shorter minimum duration
      )

2. **Check for very sparse data**:
   
   .. code-block:: python
   
      if firing_rate < 0.1:
          print("Data too sparse for burst detection")
          # Consider combining channels or longer recordings

Memory Issues
~~~~~~~~~~~~

**Problem**: Out of memory errors with large datasets.

**Solutions**:

1. **Process data in chunks**:
   
   .. code-block:: python
   
      # Analyze shorter time windows
      window_size = 60.0  # 1 minute windows
      n_windows = int(recording.duration / window_size)
      
      for i in range(n_windows):
          start_time = i * window_size
          end_time = (i + 1) * window_size
          
          # Extract window data
          window_spikes = spike_data.spike_times[
              (spike_data.spike_times >= start_time) & 
              (spike_data.spike_times < end_time)
          ]
          
          # Analyze window
          # ... process window_spikes

2. **Reduce sampling rate for analysis**:
   
   .. code-block:: python
   
      # Downsample spike times (keep every nth spike)
      downsampled_spikes = spike_times[::10]  # Keep every 10th spike

3. **Use subset of channels**:
   
   .. code-block:: python
   
      # Analyze only active channels
      active_channels = {}
      for channel_id, spike_data in recording.spike_trains.items():
          firing_rate = len(spike_data.spike_times) / recording.duration
          if firing_rate > 0.1:  # Only channels with >0.1 Hz
              active_channels[channel_id] = spike_data

Manifold Learning Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Manifold learning fails or produces poor results.

**Common issues and solutions**:

1. **Insufficient data**:
   
   .. code-block:: python
   
      # Check data size
      print(f"Feature matrix shape: {features.shape}")
      
      if features.shape[0] < 50:
           print("Warning: Few time points for manifold learning")
           print("Consider longer recordings or larger time windows")

2. **High dimensionality**:
   
   .. code-block:: python
   
      # Reduce features before manifold learning
      from sklearn.decomposition import PCA
      
      pca_pre = PCA(n_components=50)  # Reduce to 50 dimensions first
      features_reduced = pca_pre.fit_transform(features)
      
      # Then apply t-SNE or UMAP
      embedding = manifold_analyzer.apply_tsne(features_reduced)

3. **Parameter tuning**:
   
   .. code-block:: python
   
      # t-SNE parameters
      embedding = manifold_analyzer.apply_tsne(
          features, 
          perplexity=min(30, len(features)//4),  # Adjust perplexity
          n_iter=1000  # More iterations
      )

Visualization Issues
-------------------

Plot Display Problems
~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots don't display or look wrong.

**Solutions**:

.. code-block:: python

   # Set matplotlib backend
   import matplotlib
   matplotlib.use('Agg')  # For saving files
   # OR
   matplotlib.use('TkAgg')  # For interactive display
   
   # Check plot size and DPI
   plt.figure(figsize=(10, 6), dpi=300)
   
   # Ensure tight layout
   plt.tight_layout()
   
   # Save with high quality
   plt.savefig('plot.png', dpi=300, bbox_inches='tight')

Empty or Garbled Plots
~~~~~~~~~~~~~~~~~~~~~

**Problem**: Plots appear empty or with overlapping labels.

.. code-block:: python

   # Check data ranges
   print(f"X data range: {x_data.min()} - {x_data.max()}")
   print(f"Y data range: {y_data.min()} - {y_data.max()}")
   
   # Set explicit limits
   plt.xlim(x_data.min(), x_data.max())
   plt.ylim(y_data.min(), y_data.max())
   
   # Rotate labels if overlapping
   plt.xticks(rotation=45)
   
   # Adjust subplot spacing
   plt.subplots_adjust(bottom=0.2, left=0.15)

Performance Issues
-----------------

Slow Analysis
~~~~~~~~~~~~

**Problem**: Analysis takes too long to complete.

**Optimization strategies**:

1. **Use parallel processing**:
   
   .. code-block:: python
   
      from joblib import Parallel, delayed
      
      def analyze_channel(channel_data):
           # Your analysis function
           return results
      
      # Parallel processing
      results = Parallel(n_jobs=-1)(
           delayed(analyze_channel)(spike_data) 
           for spike_data in recording.spike_trains.values()
      )

2. **Reduce analysis resolution**:
   
   .. code-block:: python
   
      # Use larger time bins
      time_window = 30.0  # seconds (instead of 10.0)
      
      # Subsample data
      every_nth = 5
      subsampled_spikes = spike_times[::every_nth]

3. **Cache intermediate results**:
   
   .. code-block:: python
   
      import pickle
      
      # Save intermediate results
      with open('intermediate_results.pkl', 'wb') as f:
           pickle.dump(results, f)
      
      # Load cached results
      with open('intermediate_results.pkl', 'rb') as f:
           results = pickle.load(f)

Network Burst Detection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Network burst detection finds too many or too few events.

**Parameter tuning**:

.. code-block:: python

   burst_detector = BurstDetector(
       min_channels_in_network_burst=3,  # Require more channels
       max_network_isi=0.05,             # Stricter timing requirement
       network_burst_min_duration=0.1    # Minimum duration
   )
   
   # Check individual channel bursts first
   for channel_id, spike_data in recording.spike_trains.items():
       bursts = burst_detector.detect_bursts(spike_data)
       print(f"{channel_id}: {len(bursts)} bursts")
   
   # If few channels have bursts, network detection will fail
   if sum(len(bursts) for bursts in all_bursts.values()) < 10:
       print("Insufficient single-channel bursts for network analysis")

Getting Help
-----------

Debug Information
~~~~~~~~~~~~~~~~

When reporting issues, please include:

.. code-block:: python

   import sys
   import numpy as np
   import matplotlib
   import mea_flow
   
   print("System Information:")
   print(f"Python: {sys.version}")
   print(f"NumPy: {np.__version__}")
   print(f"Matplotlib: {matplotlib.__version__}")
   print(f"MEA-Flow: {mea_flow.__version__}")
   print(f"Platform: {sys.platform}")

Enable Detailed Logging
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   
   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)
   
   # MEA-Flow specific logging
   logger = logging.getLogger('mea_flow')
   logger.setLevel(logging.DEBUG)

Test Installation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Run this to test your installation
   from mea_flow.data import SpikeData, MEARecording
   from mea_flow.analysis import ActivityAnalyzer, BurstDetector
   from mea_flow.manifold import ManifoldAnalyzer
   from mea_flow.visualization import ActivityPlotter
   
   print("All modules imported successfully!")
   
   # Test with minimal synthetic data
   import numpy as np
   spike_times = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
   spike_data = SpikeData(
       spike_times=spike_times,
       spike_samples=(spike_times * 20000).astype(int),
       channel_id="test_channel",
       sampling_rate=20000.0
   )
   
   analyzer = ActivityAnalyzer()
   firing_rate = analyzer.calculate_firing_rate(spike_data)
   print(f"Test firing rate: {firing_rate:.2f} Hz")

Community Support
~~~~~~~~~~~~~~~~

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the API reference and tutorials
- **Examples**: Look at complete working examples in the repository

**Before opening an issue**:

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Try with the latest version
4. Provide minimal reproducible example
5. Include system information and error messages

Common Error Messages
--------------------

"ModuleNotFoundError: No module named 'mea_flow'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Install MEA-Flow properly:

.. code-block:: bash

   cd /path/to/mea-flow
   uv pip install -e .

"AttributeError: module 'mea_flow' has no attribute 'X'"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Check the correct import:

.. code-block:: python

   # Correct imports
   from mea_flow.data import SpikeData, MEARecording
   from mea_flow.analysis import ActivityAnalyzer
   
   # Not: from mea_flow import SpikeData  # Wrong!

"ValueError: Input contains NaN"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Clean your data:

.. code-block:: python

   # Remove NaN values
   spike_times = spike_times[~np.isnan(spike_times)]
   
   # Check for infinite values
   spike_times = spike_times[np.isfinite(spike_times)]

"MemoryError" or "Out of memory"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Solution**: Reduce data size or process in chunks (see Memory Issues above).

Still Having Issues?
-------------------

If this guide doesn't solve your problem:

1. **Update MEA-Flow**: Make sure you have the latest version
2. **Check Dependencies**: Verify all required packages are installed
3. **Minimal Example**: Create the smallest possible example that reproduces the issue
4. **Open GitHub Issue**: Provide detailed information and error messages

**Template for issue reports**:

```
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data with '...'
2. Run analysis with '...'
3. See error

**Expected behavior**
What you expected to happen.

**Error message**
```
Full error traceback here
```

**System information**
- OS: [e.g. Windows 10, Ubuntu 20.04, macOS 12.0]
- Python version: [e.g. 3.10.8]
- MEA-Flow version: [e.g. 0.1.0]
- Installation method: [e.g. uv, pip]

**Additional context**
Any other context about the problem.
```