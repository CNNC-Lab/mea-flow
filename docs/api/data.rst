Data Module
===========

The data module provides core classes and functions for handling MEA spike data, including loading from various formats and organizing data for analysis.

Core Classes
------------

.. currentmodule:: mea_flow.data

SpikeList
~~~~~~~~~

The main class for managing MEA spike data with well organization and temporal analysis capabilities.

.. autoclass:: SpikeList
   :members:
   :undoc-members:
   :show-inheritance:

SpikeTrain
~~~~~~~~~~

Individual spike train representation for single electrode data.

.. autoclass:: spike_list.SpikeTrain
   :members:
   :undoc-members:
   :show-inheritance:

Data Loaders
------------

.. currentmodule:: mea_flow.data.loaders

Functions for loading MEA data from various file formats.

File Format Loaders
~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_matlab_file

.. autofunction:: load_csv_data

.. autofunction:: load_dataframe_data

.. autofunction:: load_hdf5_data

Format Detection
~~~~~~~~~~~~~~~~

.. autofunction:: detect_file_format

.. autofunction:: load_auto_format

Utility Functions
-----------------

.. currentmodule:: mea_flow.data.utils

Helper functions for data processing and validation.

.. autofunction:: validate_spike_data

.. autofunction:: create_well_mapping

.. autofunction:: channel_to_well_position

Examples
--------

Basic Usage
~~~~~~~~~~~

Creating a SpikeList from arrays:

.. code-block:: python

   import numpy as np
   from mea_flow.data import SpikeList
   
   # Create spike data
   spike_times = np.array([0.1, 0.5, 1.2, 2.3])
   channels = np.array([0, 1, 0, 1])
   
   # Create SpikeList
   spike_list = SpikeList(
       spike_data={'times': spike_times, 'channels': channels},
       recording_length=5.0
   )
   
   # Access spike trains
   channel_0_train = spike_list.spike_trains[0]
   print(f"Channel 0 has {channel_0_train.n_spikes} spikes")

Loading from MATLAB Files
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from mea_flow.data.loaders import load_matlab_file
   
   # Load Axion format
   spike_list = load_matlab_file(
       'recording.mat',
       channels_key='Channels',
       times_key='Times',
       recording_length=60.0
   )

Well-Based Organization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create well mapping for 2x2 well plate
   well_map = {
       0: [0, 1, 2, 3],      # Well 0
       1: [4, 5, 6, 7],      # Well 1
       2: [8, 9, 10, 11],    # Well 2
       3: [12, 13, 14, 15]   # Well 3
   }
   
   spike_list = SpikeList(
       spike_data=data,
       recording_length=recording_length,
       well_map=well_map
   )
   
   # Access well data
   well_0_channels = spike_list.get_well_channels(0)
   well_filtered = spike_list.select_wells([0, 1])

Time Analysis
~~~~~~~~~~~~~

.. code-block:: python

   # Time slicing
   time_slice = spike_list.time_slice(10.0, 30.0)  # 10-30 seconds
   
   # Binning for analysis
   spike_matrix, time_bins = spike_list.bin_spikes(bin_size=0.1)
   
   # Convert to continuous signals
   signals, time_vector = spike_list.to_continuous_signal(
       tau=0.02,  # Time constant
       dt=0.001   # Sampling rate
   )

Data Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   from mea_flow.data.utils import validate_spike_data
   
   # Validate spike data format
   is_valid, errors = validate_spike_data(
       {'times': spike_times, 'channels': channels}
   )
   
   if not is_valid:
       print(f"Data validation errors: {errors}")

See Also
--------

- :doc:`../quickstart`: Quick start guide with data loading examples
- :doc:`../tutorials/index`: Detailed tutorials on data handling
- :doc:`analysis`: Analysis functions that work with SpikeList objects