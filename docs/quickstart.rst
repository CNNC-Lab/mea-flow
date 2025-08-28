Quick Start Guide
=================

This guide will get you up and running with MEA-Flow in just a few minutes.

Basic Workflow
--------------

MEA-Flow follows a simple workflow for analyzing MEA data:

1. **Load Data** → 2. **Compute Metrics** → 3. **Analyze Dynamics** → 4. **Visualize Results**

Loading MEA Data
----------------

MEA-Flow supports multiple data formats. Here are the most common ways to load your data:

**From MATLAB Files (.mat)**

.. code-block:: python

   from mea_flow import SpikeList
   
   # Load from MATLAB file (Axion format)
   spike_list = SpikeList.from_matlab('recording.mat')
   
   # Or specify custom field names
   spike_list = SpikeList.from_matlab(
       'recording.mat',
       channels_key='Channels',
       times_key='Times'
   )

**From CSV Files**

.. code-block:: python

   from mea_flow.data.loaders import load_csv_data
   
   spike_list = load_csv_data(
       'spikes.csv',
       time_col='spike_time',
       channel_col='electrode',
       recording_length=60.0  # seconds
   )

**From Pandas DataFrames**

.. code-block:: python

   import pandas as pd
   from mea_flow.data.loaders import load_dataframe_data
   
   # Your DataFrame with columns: 'time', 'channel'
   df = pd.read_csv('your_data.csv')
   
   spike_list = load_dataframe_data(
       df,
       time_col='time',
       channel_col='channel',
       recording_length=300.0
   )

**Manual Creation**

.. code-block:: python

   import numpy as np
   from mea_flow import SpikeList
   
   # Create from arrays
   spike_times = np.array([0.1, 0.5, 1.2, 2.3])
   channels = np.array([0, 1, 0, 1])
   
   spike_list = SpikeList(
       spike_data={'times': spike_times, 'channels': channels},
       recording_length=5.0
   )

Computing Metrics
-----------------

Use the MEAMetrics class to compute comprehensive analysis metrics:

.. code-block:: python

   from mea_flow import MEAMetrics
   
   # Initialize metrics calculator
   metrics = MEAMetrics()
   
   # Compute all metrics globally
   results = metrics.compute_all_metrics(spike_list, grouping='global')
   
   # Results contain:
   print(results['activity'])    # Firing rates, burst detection
   print(results['regularity'])  # CV-ISI, entropy measures
   print(results['synchrony'])   # Correlations, distances
   print(results['bursts'])      # Network burst analysis

**Different Grouping Options**

.. code-block:: python

   # Global analysis (all channels together)
   global_results = metrics.compute_all_metrics(spike_list, grouping='global')
   
   # Per-channel analysis
   channel_results = metrics.compute_all_metrics(spike_list, grouping='channel')
   
   # Per-well analysis (if well mapping available)
   well_results = metrics.compute_all_metrics(spike_list, grouping='well')

**Selective Metric Computation**

.. code-block:: python

   # Compute only specific metric categories
   activity_metrics = metrics.compute_activity_metrics(spike_list)
   sync_metrics = metrics.compute_synchrony_metrics(spike_list)

Manifold Learning Analysis
--------------------------

Analyze population dynamics using manifold learning techniques:

.. code-block:: python

   from mea_flow import ManifoldAnalysis
   
   # Initialize manifold analyzer
   manifold = ManifoldAnalysis()
   
   # Analyze population dynamics
   manifold_results = manifold.analyze_population_dynamics(spike_list)
   
   # Results contain embeddings for multiple methods
   embeddings = manifold_results['embeddings']
   
   # Access specific methods
   pca_embedding = embeddings['PCA']['embedding']
   tsne_embedding = embeddings['t-SNE']['embedding']
   
   # Evaluation metrics
   evaluation = manifold_results['evaluation']
   dimensionality = manifold_results['dimensionality']

**Custom Configuration**

.. code-block:: python

   from mea_flow.manifold.analysis import ManifoldConfig
   
   # Create custom configuration
   config = ManifoldConfig(
       methods=['PCA', 'UMAP', 't-SNE'],  # Only these methods
       max_components=3,
       tau=0.02,  # Exponential filter time constant
       dt=0.001   # Sampling interval
   )
   
   manifold = ManifoldAnalysis(config=config)
   results = manifold.analyze_population_dynamics(spike_list)

Creating Visualizations
-----------------------

Generate publication-ready plots with the MEAPlotter class:

.. code-block:: python

   from mea_flow import MEAPlotter
   
   # Initialize plotter
   plotter = MEAPlotter()
   
   # Basic raster plot
   fig = plotter.plot_spike_raster(spike_list)
   
   # Activity summary
   fig = plotter.plot_activity_summary(spike_list, results['activity'])
   
   # Synchrony analysis
   fig = plotter.plot_synchrony_analysis(spike_list, results['synchrony'])
   
   # Manifold results
   fig = plotter.plot_manifold_results(manifold_results)
   
   # Comprehensive summary report
   fig = plotter.create_summary_report([spike_list], results)

**Customization Options**

.. code-block:: python

   # Custom styling
   plotter = MEAPlotter(
       style='seaborn-v0_8',
       color_palette='viridis',
       figsize=(12, 8)
   )
   
   # Raster plot with customization
   fig = plotter.plot_spike_raster(
       spike_list,
       time_range=(10, 30),  # Show only 10-30 seconds
       channels=[0, 1, 2, 3],  # Specific channels
       color_by_well=True,  # Color by well if available
       title="My MEA Recording"
   )

Complete Example
----------------

Here's a complete analysis workflow:

.. code-block:: python

   import numpy as np
   from mea_flow import SpikeList, MEAMetrics, ManifoldAnalysis, MEAPlotter
   
   # 1. Load or create data
   spike_list = SpikeList.from_matlab('my_recording.mat')
   
   # 2. Compute comprehensive metrics
   metrics = MEAMetrics()
   results = metrics.compute_all_metrics(spike_list, grouping='global')
   
   print(f"Active channels: {results['activity']['active_channels']}")
   print(f"Mean firing rate: {results['activity']['mean_firing_rate']:.2f} Hz")
   print(f"Mean correlation: {results['synchrony']['pearson_cc_mean']:.3f}")
   
   # 3. Analyze population dynamics
   manifold = ManifoldAnalysis()
   manifold_results = manifold.analyze_population_dynamics(spike_list)
   
   print(f"Effective dimensionality: {manifold_results['dimensionality']}")
   
   # 4. Create visualizations
   plotter = MEAPlotter()
   
   # Individual plots
   raster_fig = plotter.plot_spike_raster(spike_list)
   manifold_fig = plotter.plot_manifold_results(manifold_results)
   
   # Comprehensive summary
   summary_fig = plotter.create_summary_report([spike_list], results)
   
   # Save figures
   plotter.save_figure(summary_fig, 'mea_analysis_summary.pdf', dpi=300)

Cross-Condition Analysis
------------------------

Compare multiple experimental conditions:

.. code-block:: python

   # Load multiple conditions
   control_data = SpikeList.from_matlab('control.mat')
   treatment_data = SpikeList.from_matlab('treatment.mat')
   
   spike_lists = [control_data, treatment_data]
   condition_names = ['Control', 'Treatment']
   
   # Analyze all conditions
   all_results = {}
   for name, data in zip(condition_names, spike_lists):
       results = metrics.compute_all_metrics(data, grouping='global')
       all_results[name] = results
   
   # Compare conditions
   comparison_fig = plotter.plot_condition_comparison(all_results)

Well-Based Analysis
-------------------

For multi-well MEA plates:

.. code-block:: python

   # Create well mapping (example for 2x2 well plate)
   well_map = {
       0: [0, 1, 2, 3],      # Well 0: channels 0-3
       1: [4, 5, 6, 7],      # Well 1: channels 4-7
       2: [8, 9, 10, 11],    # Well 2: channels 8-11
       3: [12, 13, 14, 15]   # Well 3: channels 12-15
   }
   
   # Create SpikeList with well mapping
   spike_list = SpikeList(
       spike_data={'times': spike_times, 'channels': channels},
       recording_length=recording_length,
       well_map=well_map
   )
   
   # Analyze by wells
   well_results = metrics.compute_all_metrics(spike_list, grouping='well')
   
   # Visualize electrode map
   activity_data = {ch: spike_list.spike_trains[ch].n_spikes 
                   for ch in spike_list.spike_trains.keys()}
   electrode_fig = plotter.plot_electrode_map(activity_data, well_map)

Next Steps
----------

- **Detailed Tutorial**: Work through the :doc:`notebooks/01_mea_flow_tutorial`
- **Examples**: Explore specific use cases in :doc:`examples/index`
- **API Reference**: Deep dive into the :doc:`api/data` documentation
- **Advanced Topics**: Learn about custom configurations and extensions