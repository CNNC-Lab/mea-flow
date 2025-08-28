# MEA-Flow: Neural Population Dynamics Analysis for Multi-Electrode Arrays

<p align="center">
  <img src="docs/assets/logo.png" alt="MEA-Flow Logo" width="400"/>
</p>

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/CNNC-Lab/mea-flow)](https://github.com/CNNC-Lab/mea-flow/blob/main/LICENSE)
[![CI](https://github.com/CNNC-Lab/mea-flow/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/CNNC-Lab/mea-flow/actions)
[![Coverage](https://codecov.io/gh/CNNC-Lab/mea-flow/branch/main/graph/badge.svg)](https://codecov.io/gh/CNNC-Lab/mea-flow)
[![PyPI](https://img.shields.io/pypi/v/mea-flow)](https://pypi.org/project/mea-flow/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
- **Multiple Formats**: Axion .spk files (native Python), .mat files, CSV, HDF5, pandas DataFrames
- **Well-Based Organization**: Automatic channel-to-well mapping for multi-well plates
- **Time Window Analysis**: Temporal segmentation and sliding window analysis
- **Native .spk Loading**: No MATLAB dependency, 2-5x faster performance

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

MEA-Flow uses `uv` for fast and reliable dependency management with optional dependencies for enhanced functionality.

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up MEA-Flow
git clone https://github.com/CNNC-Lab/mea-flow.git
cd mea-flow

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Basic installation (core functionality)
uv pip install -e .

# OR install with all optional dependencies
uv pip install -e ".[full]"

# Verify installation
python -c "import mea_flow; print('MEA-Flow installed successfully!')"
```

### Installation Options

**Basic Installation** (recommended for most users):
```bash
uv pip install -e .
```
Installs core functionality with all essential features.

**Full Installation** (for advanced users):
```bash
uv pip install -e ".[full]"
```
Includes optional dependencies like PySpike and UMAP-learn.

**Development Installation**:
```bash
uv pip install -e ".[dev,notebooks]"
# OR with new uv dependency groups:
uv sync --group dev
```
Includes testing, documentation, and Jupyter notebook support.

### Dependencies

**Core Dependencies** (always installed):
- **Computation**: `numpy`, `scipy`, `pandas`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn` 
- **Data Handling**: `h5py`, `tqdm`, `joblib`

**Optional Dependencies** (install with `[full]`):
- **PySpike**: Advanced spike train distance measures
- **UMAP-learn**: UMAP manifold learning method

### Using uv (Recommended)

If you have uv installed, you can use the modern dependency management:

```bash
# Install and sync all dependencies
uv sync

# Run with uv (automatically manages virtual environment)
uv run python verify_install.py

# Run Jupyter with all dependencies
uv run jupyter lab

# Add new dependencies
uv add package-name
uv add --dev dev-package-name
```

### Troubleshooting

If you encounter installation issues with optional dependencies:

1. **PySpike build errors**: PySpike requires compilation. Use the basic installation:
   ```bash
   uv pip install -e .
   ```
   The library will work without PySpike, using alternative distance measures.

2. **UMAP issues**: Install basic version and add UMAP separately if needed:
   ```bash
   uv pip install -e .
   pip install umap-learn
   ```

3. **uv.lock parse errors**: Remove the lock file and regenerate:
   ```bash
   rm uv.lock
   uv lock
   ```
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

## 📖 Documentation & Examples

### Getting Started
1. **[Tutorial Notebook](notebooks/01_mea_flow_tutorial.ipynb)** - Interactive walkthrough
2. **[Working Examples](examples/)** - Real analysis workflows
3. **[Installation Guide](INSTALL_GUIDE.md)** - Detailed setup instructions

### Example Scripts
- **[Basic Analysis](examples/basic_analysis.py)** - Core metrics and visualization
- **[Cross-Condition Comparison](examples/cross_condition_comparison.py)** - Statistical comparisons
- **[Manifold Learning](examples/manifold_learning_workflow.py)** - Population dynamics
- **[Well Plate Analysis](examples/well_plate_analysis.py)** - Multi-well experiments
- **[Native .spk Loading](examples/spk_loader_demo.py)** - File format examples

### Key Workflows

**Basic Analysis Pipeline:**
```python
# 1. Load data (automatic format detection)
from mea_flow.data import load_data
spike_list = load_data('recording.spk')  # Native Python loader

# 2. Compute metrics
from mea_flow.analysis import MEAMetrics
analyzer = MEAMetrics()
metrics = analyzer.compute_all_metrics(spike_list)

# 3. Manifold analysis
from mea_flow.manifold import ManifoldAnalysis  
manifold_analyzer = ManifoldAnalysis()
results = manifold_analyzer.analyze_population_dynamics(spike_list)

# 4. Visualization
from mea_flow.visualization import MEAPlotter
plotter = MEAPlotter()
fig = plotter.plot_raster(spike_list, time_range=(0, 60))
plotter.save_figure(fig, 'examples/output/raster_plot.png')
```

**Cross-Condition Comparison:**
```python
# Load multiple conditions
control_data = load_data('control.spk')
treatment_data = load_data('treatment.spk')
conditions = {'control': control_data, 'treatment': treatment_data}

# Comprehensive comparison
metrics_comparison = analyzer.compare_conditions(conditions)
manifold_comparison = manifold_analyzer.compare_conditions(conditions) 

# Statistical visualization
fig = plotter.plot_metrics_comparison(metrics_comparison, grouping_col='condition')
plotter.save_figure(fig, 'examples/output/condition_comparison.png')
```

