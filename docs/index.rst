MEA-Flow: Neural Population Dynamics Analysis
=============================================

MEA-Flow is a comprehensive Python package for analyzing multi-electrode array (MEA) data with a focus on neural population dynamics, manifold learning, and comparative analysis across experimental conditions.

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/github/license/CNNC-Lab/mea-flow
   :target: https://github.com/CNNC-Lab/mea-flow/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/docs-sphinx-blue.svg
   :target: https://mea-flow.readthedocs.io
   :alt: Documentation Status

Key Features
============

🔬 **Comprehensive Analysis Suite**
   - Activity metrics: firing rates, burst detection, network analysis
   - Regularity metrics: CV-ISI, entropy measures, Fano factors
   - Synchrony metrics: correlations, spike distances, population measures

📊 **Advanced Manifold Learning**
   - Multiple methods: PCA, MDS, Isomap, LLE, UMAP, t-SNE
   - Population geometry and state space analysis
   - Cross-condition manifold comparison

📁 **Flexible Data Input**
   - Multiple formats: .spk, .mat, CSV, HDF5, pandas DataFrames
   - Well-based organization for multi-well plates
   - Time window and channel selection

📈 **Publication-Ready Visualizations**
   - Raster plots with well coloring
   - Electrode maps and activity patterns
   - Statistical comparisons and manifold embeddings

Quick Start
===========

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install mea-flow
   
   # Or with uv
   uv add mea-flow
   
   # Development installation
   git clone https://github.com/CNNC-Lab/mea-flow.git
   cd mea-flow
   uv sync

Basic Usage
-----------

.. code-block:: python

   from mea_flow import SpikeList, MEAMetrics, ManifoldAnalysis, MEAPlotter

   # Load MEA data
   spike_list = SpikeList.from_matlab('your_data.mat')

   # Compute comprehensive metrics
   metrics = MEAMetrics()
   results = metrics.compute_all_metrics(spike_list, grouping='well')

   # Analyze population dynamics
   manifold = ManifoldAnalysis()
   pop_results = manifold.analyze_population_dynamics(spike_list)

   # Create publication-ready plots
   plotter = MEAPlotter()
   plotter.create_summary_report([spike_list], results)

Documentation Structure
=======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data
   api/analysis
   api/manifold
   api/visualization
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

.. toctree::
   :maxdepth: 1
   :caption: Tutorials & Examples

   notebooks/01_mea_flow_tutorial

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`