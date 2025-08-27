Analysis Module
===============

The analysis module provides comprehensive metrics and statistical analysis tools for MEA data, including activity, regularity, synchrony, and burst analysis.

Main Classes
------------

.. currentmodule:: mea_flow.analysis

MEAMetrics
~~~~~~~~~~

The main orchestrator class for computing comprehensive MEA analysis metrics.

.. autoclass:: MEAMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Activity Analysis
-----------------

.. currentmodule:: mea_flow.analysis.activity

Functions for analyzing neural activity patterns, firing rates, and burst detection.

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: compute_activity_metrics

.. autofunction:: firing_rate

.. autofunction:: burst_detection

.. autofunction:: population_activity

Burst Analysis
~~~~~~~~~~~~~~

.. autofunction:: detect_bursts_envelope

.. autofunction:: burst_statistics

.. autofunction:: interburst_intervals

Regularity Analysis
-------------------

.. currentmodule:: mea_flow.analysis.regularity

Functions for analyzing spike timing regularity and variability measures.

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: compute_regularity_metrics

.. autofunction:: cv_isi

.. autofunction:: local_variation

.. autofunction:: entropy_isi

.. autofunction:: fano_factor

Advanced Measures
~~~~~~~~~~~~~~~~~

.. autofunction:: spike_time_tiling_coefficient

.. autofunction:: victor_purpura_distance

.. autofunction:: regularity_index

Synchrony Analysis
------------------

.. currentmodule:: mea_flow.analysis.synchrony

Functions for analyzing synchronization and correlation between channels.

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: compute_synchrony_metrics

.. autofunction:: pairwise_correlations

.. autofunction:: spike_distance_measures

.. autofunction:: van_rossum_distance

Population Synchrony
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: population_synchrony_measures

.. autofunction:: chi_square_distance

.. autofunction:: cosyne_similarity

.. autofunction:: synchrony_index

PySpike Integration
~~~~~~~~~~~~~~~~~~~

When PySpike is available, additional distance measures are provided:

.. autofunction:: pyspike_isi_distance

.. autofunction:: pyspike_spike_distance

.. autofunction:: pyspike_spike_sync

Burst Analysis
--------------

.. currentmodule:: mea_flow.analysis.burst_analysis

Functions for network-level burst detection and analysis.

Network Bursts
~~~~~~~~~~~~~~

.. autofunction:: network_burst_analysis

.. autofunction:: detect_network_bursts

.. autofunction:: network_burst_statistics

Burst Characterization
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: burst_participation

.. autofunction:: burst_synchrony

.. autofunction:: burst_propagation

Configuration
-------------

.. currentmodule:: mea_flow.analysis.config

Analysis configuration and parameter management.

.. autoclass:: AnalysisConfig
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Basic Metrics Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mea_flow import SpikeList, MEAMetrics
   
   # Load your data
   spike_list = SpikeList.from_matlab('recording.mat')
   
   # Compute all metrics
   metrics = MEAMetrics()
   results = metrics.compute_all_metrics(spike_list, grouping='global')
   
   # Access different metric categories
   activity = results['activity']
   regularity = results['regularity']
   synchrony = results['synchrony']
   bursts = results['bursts']
   
   print(f"Mean firing rate: {activity['mean_firing_rate']:.2f} Hz")
   print(f"Mean CV-ISI: {regularity['mean_cv_isi']:.3f}")
   print(f"Mean correlation: {synchrony['pearson_cc_mean']:.3f}")

Per-Channel Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze each channel separately
   channel_results = metrics.compute_all_metrics(spike_list, grouping='channel')
   
   # Access results for specific channel
   channel_5_metrics = channel_results[5]
   print(f"Channel 5 firing rate: {channel_5_metrics['firing_rate']:.2f} Hz")

Well-Based Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze by wells (requires well mapping)
   well_results = metrics.compute_all_metrics(spike_list, grouping='well')
   
   # Compare wells
   for well_id, well_metrics in well_results.items():
       print(f"Well {well_id} firing rate: {well_metrics['mean_firing_rate']:.2f} Hz")

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mea_flow.analysis.config import AnalysisConfig
   
   # Create custom configuration
   config = AnalysisConfig(
       min_spikes=5,           # Minimum spikes for analysis
       burst_threshold=0.1,    # Burst detection threshold
       sync_time_bin=0.01,     # Synchrony analysis bin size
       max_isi=2.0            # Maximum ISI to consider
   )
   
   metrics = MEAMetrics(config=config)
   results = metrics.compute_all_metrics(spike_list)

Selective Metric Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute only specific metrics
   activity_only = metrics.compute_activity_metrics(spike_list)
   sync_only = metrics.compute_synchrony_metrics(spike_list)
   
   # Custom burst analysis
   from mea_flow.analysis.burst_analysis import network_burst_analysis
   
   network_bursts = network_burst_analysis(spike_list, config)

Advanced Synchrony Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mea_flow.analysis.synchrony import pairwise_correlations, van_rossum_distance
   
   # Get active channels
   active_channels = spike_list.get_active_channels(min_spikes=10)
   
   # Compute pairwise correlations
   correlations = pairwise_correlations(
       spike_list, 
       channels=active_channels,
       bin_size=0.01
   )
   
   # Van Rossum distances
   distances = van_rossum_distance(
       spike_list,
       channels=active_channels,
       tau=0.02  # Time constant
   )

Cross-Condition Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare conditions
   control_results = metrics.compute_all_metrics(control_data, grouping='global')
   treatment_results = metrics.compute_all_metrics(treatment_data, grouping='global')
   
   # Statistical comparison
   from scipy import stats
   
   control_rates = [control_results['activity']['mean_firing_rate']]
   treatment_rates = [treatment_results['activity']['mean_firing_rate']]
   
   t_stat, p_value = stats.ttest_ind(control_rates, treatment_rates)
   print(f"Firing rate difference: t={t_stat:.3f}, p={p_value:.3f}")

See Also
--------

- :doc:`data`: Data structures used by analysis functions
- :doc:`manifold`: Population dynamics and manifold learning
- :doc:`visualization`: Plotting analysis results
- :doc:`../tutorials/index`: Detailed analysis tutorials