Tutorials
=========

This section contains comprehensive tutorials for using MEA-Flow to analyze multi-electrode array data. Each tutorial is designed to guide you through specific aspects of MEA data analysis, from basic workflows to advanced techniques.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   spike_analysis
   burst_detection
   manifold_learning
   cross_condition_comparison
   well_plate_experiments
   advanced_workflows
   troubleshooting

Overview
--------

The tutorials are organized from basic to advanced topics:

**Getting Started**
   Learn the fundamentals of MEA-Flow, including data loading, basic analysis, and visualization.

**Spike Analysis**
   Deep dive into spike train analysis, including firing rate calculations, ISI analysis, and spike pattern detection.

**Burst Detection**
   Understand burst detection algorithms, parameter tuning, and network burst analysis.

**Manifold Learning**
   Explore dimensionality reduction techniques for analyzing neural population dynamics.

**Cross-Condition Comparison**
   Learn statistical methods for comparing neural activity across experimental conditions.

**Well Plate Experiments**
   Analyze multi-well MEA plate data, including dose-response experiments and cross-well comparisons.

**Advanced Workflows**
   Combine multiple analysis techniques for comprehensive MEA data analysis.

**Troubleshooting**
   Common issues and solutions when working with MEA-Flow.

Prerequisites
-------------

Before starting these tutorials, you should:

1. Have MEA-Flow installed (see :doc:`../installation`)
2. Be familiar with Python basics
3. Have basic understanding of neuroscience concepts (spikes, bursts, neural activity)
4. Understand matplotlib for plotting (helpful but not required)

Example Data
------------

Many tutorials use synthetic data generated within the examples. This ensures:

- Reproducible results across different environments
- No need to download large datasets
- Focus on analysis techniques rather than data preprocessing
- Demonstrations of expected data formats

For real data analysis, adapt the loading sections to your specific data format.

Getting Help
------------

If you encounter issues while following these tutorials:

1. Check the :doc:`troubleshooting` guide
2. Review the :doc:`../api/index` documentation
3. Look at the complete examples in the ``examples/`` directory
4. Open an issue on GitHub for bugs or feature requests

Next Steps
----------

Start with :doc:`getting_started` for a general introduction to MEA-Flow, or jump directly to a specific topic that interests you.

All tutorial code is also available as runnable Python scripts in the ``examples/`` directory of the repository.