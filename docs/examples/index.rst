Examples
========

This section contains complete, runnable examples demonstrating various aspects of MEA-Flow functionality. Each example is available as both documentation and as a standalone Python script in the ``examples/`` directory.

Complete Workflows
------------------

**Basic MEA Analysis** (:doc:`../examples/basic_analysis`)
   Introduction to MEA-Flow with synthetic data generation, activity analysis, and visualization.
   
   *Script: examples/basic_analysis.py*

**Cross-Condition Comparison** (:doc:`../examples/cross_condition_comparison`)
   Statistical comparison of neural activity across multiple experimental conditions using manifold learning.
   
   *Script: examples/cross_condition_comparison.py*

**Manifold Learning Workflow** (:doc:`../examples/manifold_learning_workflow`)
   Advanced population dynamics analysis with dimensionality reduction and trajectory analysis.
   
   *Script: examples/manifold_learning_workflow.py*

**Well Plate Analysis** (:doc:`../examples/well_plate_analysis`)
   Multi-well MEA plate analysis including dose-response experiments and cross-well comparisons.
   
   *Script: examples/well_plate_analysis.py*

Running the Examples
--------------------

All examples are designed to be self-contained and runnable:

.. code-block:: bash

   # Navigate to the MEA-Flow directory
   cd mea-flow
   
   # Run any example
   python examples/basic_analysis.py
   python examples/cross_condition_comparison.py
   python examples/manifold_learning_workflow.py
   python examples/well_plate_analysis.py

Each example will:

- Generate synthetic MEA data (no external data required)
- Perform the complete analysis workflow
- Create visualizations and save them to an output directory
- Print summary statistics and results

Output Structure
---------------

Examples create organized output directories:

.. code-block::

   output/
   ├── basic_analysis/
   │   ├── activity_summary.png
   │   ├── raster_plots.png
   │   └── analysis_results.csv
   ├── cross_condition_comparison/
   │   ├── firing_rate_comparison.png
   │   ├── manifold_comparison.png
   │   └── statistical_summary.txt
   ├── manifold_learning_workflow/
   │   ├── manifold_overview.png
   │   ├── pca_detailed_analysis.png
   │   └── manifold_analysis_report.txt
   └── well_plate_analysis/
       ├── well_plate_heatmaps.png
       ├── dose_response_curves.png
       └── detailed_well_analysis.csv

Customizing Examples
-------------------

The examples are designed to be easily customized:

**Data Parameters**
   Modify duration, number of channels, sampling rate, and activity patterns.

**Analysis Parameters**
   Adjust burst detection thresholds, manifold learning methods, and statistical tests.

**Visualization Options**
   Change plot styles, colors, and output formats.

**Output Options**
   Modify save locations and file formats.

Example Template
---------------

Use this template to create your own analysis workflows:

.. code-block:: python

   #!/usr/bin/env python3
   """
   Custom MEA Analysis Example
   ===========================
   
   Description of your analysis workflow.
   """
   
   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   import logging
   
   from mea_flow.data import SpikeData, MEARecording
   from mea_flow.analysis import ActivityAnalyzer, BurstDetector
   from mea_flow.visualization import ActivityPlotter
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def main():
       """Main analysis workflow."""
       logger.info("Starting custom MEA analysis")
       
       # Create output directory
       output_dir = Path("output/custom_analysis")
       output_dir.mkdir(parents=True, exist_ok=True)
       
       # Your analysis code here...
       
       print("Analysis complete!")
   
   if __name__ == "__main__":
       main()

Integration with Tutorials
-------------------------

These examples complement the step-by-step tutorials:

- **Examples** provide complete, working code for specific use cases
- **Tutorials** provide detailed explanations and educational content
- Both use the same MEA-Flow API and follow similar patterns

For learning MEA-Flow, we recommend:

1. Start with :doc:`../tutorials/getting_started`
2. Run :doc:`../examples/basic_analysis` 
3. Follow specific tutorials for your analysis needs
4. Adapt the relevant examples for your own data

Next Steps
----------

- Browse the :doc:`../tutorials/index` for detailed explanations
- Check the :doc:`../api/index` for complete API documentation
- Join the community discussions on GitHub for questions and contributions

.. toctree::
   :maxdepth: 1
   :hidden:

   basic_analysis
   cross_condition_comparison  
   manifold_learning_workflow
   well_plate_analysis