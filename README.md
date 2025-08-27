# MEA-Flow: Neural Population Dynamics Analysis for Multi-Electrode Arrays

MEA-Flow is a comprehensive Python package for analyzing multi-electrode array (MEA) data with a focus on neural population dynamics, manifold learning, and comparative analysis across experimental conditions.

## 🚀 Key Features

### 📊 **Comprehensive Metrics Analysis**
- **Activity Metrics**: Firing rates, spike counts, burst detection, network burst analysis
- **Regularity Metrics**: CV-ISI, Local Variation (LV), entropy measures, Fano factors
- **Synchrony Metrics**: Pairwise correlations, PySpike distances, population synchrony measures

### 🔬 **Advanced Manifold Learning**
- **Multiple Methods**: PCA, MDS, Isomap, LLE, UMAP, t-SNE, Spectral Embedding
- **Population Geometry**: State space analysis of neural population dynamics
- **Embedding Evaluation**: Trustworthiness, continuity, reconstruction error metrics
- **Cross-Condition Comparison**: Procrustes alignment, classification analysis

### 📁 **Flexible Data Input**
- **Multiple Formats**: Axion .spk files (via MATLAB), .mat files, CSV, HDF5, pandas DataFrames
- **Well-Based Organization**: Automatic channel-to-well mapping for multi-well plates
- **Time Window Analysis**: Temporal segmentation and sliding window analysis

### 📈 **Publication-Ready Visualizations**
- **Raster Plots**: Static and animated spike visualizations with well coloring
- **Electrode Maps**: Spatial activity patterns and electrode layouts
- **Metrics Comparisons**: Statistical plots with significance testing
- **Manifold Visualizations**: 2D/3D embeddings, trajectories, dimensionality analysis

### 🔧 **Analysis Workflows**
- **Multi-Level Analysis**: Global, well-based, time-resolved, and channel-level grouping
- **Comparative Studies**: Cross-condition statistical analysis and feature importance
- **Configurable Pipelines**: Preset configurations for different experimental paradigms

## 🛠️ Installation

MEA-Flow uses `uv` for fast and reliable dependency management.

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up MEA-Flow
git clone https://github.com/username/mea-flow.git
cd mea-flow

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
uv pip install -e .

# Verify installation
python -c "import mea_flow; print('MEA-Flow installed successfully!')"
```

### Dependencies

MEA-Flow automatically handles the following key dependencies:
- **Core**: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`
- **Analysis**: `scikit-learn`, `pyspike` (for spike train analysis)
- **Manifold Learning**: `umap-learn` (optional, for UMAP)
- **Data I/O**: `h5py` (for HDF5 support)

## 🚀 Quick Example

```python
import numpy as np
from mea_flow import SpikeList, MEAMetrics, MEAPlotter

# Load your MEA data
spike_list = SpikeList(spike_data=your_spike_data, 
                      recording_length=300.0)

# Compute comprehensive metrics
analyzer = MEAMetrics()
metrics = analyzer.compute_all_metrics(spike_list)

# Create visualizations
plotter = MEAPlotter()
fig = plotter.plot_raster(spike_list, time_range=(0, 30))
```

## 📖 Documentation & Tutorials

### Getting Started
1. **[Tutorial Notebook](notebooks/01_mea_flow_tutorial.ipynb)** - Comprehensive walkthrough
2. **[API Reference](docs/)** - Detailed function documentation  
3. **[Examples](examples/)** - Specific use cases and workflows

### Key Workflows

**Basic Analysis Pipeline:**
```python
# 1. Load data
from mea_flow import load_data
spike_list = load_data('your_data.mat')

# 2. Compute metrics
from mea_flow import MEAMetrics
analyzer = MEAMetrics()
metrics = analyzer.compare_conditions(spike_lists_dict)

# 3. Manifold analysis
from mea_flow import ManifoldAnalysis  
manifold_analyzer = ManifoldAnalysis()
results = manifold_analyzer.analyze_population_dynamics(spike_list)

# 4. Visualization
from mea_flow import MEAPlotter
plotter = MEAPlotter()
plotter.create_summary_report(spike_lists_dict, metrics)
```

**Cross-Condition Comparison:**
```python
# Load multiple conditions
conditions = {'control': spike_list1, 'treatment': spike_list2}

# Comprehensive comparison
metrics_comparison = analyzer.compare_conditions(conditions)
manifold_comparison = manifold_analyzer.compare_conditions(conditions) 

# Statistical visualization
plotter.plot_metrics_comparison(metrics_comparison, grouping_col='condition')
```

