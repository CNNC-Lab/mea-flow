# **MEA-Flow**: Neural Population Dynamics Analysis for Multi-Electrode Arrays

<p align="center">
  <img src="docs/logo.png" alt="MEA-Flow Logo" width="400"/>
</p>

MEA-Flow is a comprehensive Python package for analyzing multi-electrode array (MEA) data with a focus on population dynamics, feature analysis, and comparative studies across experimental conditions. The package provides a complete pipeline from data loading to advanced statistical analysis and visualization.

## 🚀 Key Features

### 📊 **Comprehensive Neural Activity Analysis**
- **Activity Metrics**: Firing rates, spike counts, burst detection, network burst analysis
- **Regularity Metrics**: CV-ISI, Local Variation (LV), entropy measures, Fano factors
- **Synchrony Metrics**: Pairwise correlations, spike train distances, population synchrony measures
- **Burst Analysis**: Individual electrode and network-level burst detection with adaptive thresholds

### 🔬 **Advanced Feature Analysis**
- **Feature Selection**: 11 complementary methods including mutual information, LASSO, Random Forest, XGBoost
- **Redundancy Detection**: Variance Inflation Factor (VIF) and correlation analysis
- **Consensus Ranking**: Multi-method feature importance with Borda count aggregation
- **Machine Learning**: Classification analysis for experimental condition prediction

### 🌐 **Manifold Learning & Dimensionality Reduction**
- **Multiple Methods**: PCA, MDS, Isomap, LLE, UMAP, t-SNE, Spectral Embedding
- **Population Dynamics**: State space analysis of neural population geometry
- **Embedding Evaluation**: Trustworthiness, continuity, and reconstruction error metrics
- **Cross-Condition Comparison**: Procrustes alignment and geometric analysis

### 📁 **Native Data Loading**
- **Axion .spk Files**: Native Python loader with 100% MATLAB compatibility
- **Multiple Formats**: .mat files, CSV, HDF5, pandas DataFrames
- **No Dependencies**: Eliminates MATLAB requirement with 2-5x faster performance
- **Universal Compatibility**: Handles all .spk file variants with robust error handling

### 📈 **Publication-Ready Visualizations**
- **Raster Plots**: Spike train visualizations with electrode mapping
- **Feature Analysis**: Importance rankings, consensus plots, method comparisons
- **Manifold Visualizations**: 2D/3D embeddings with condition coloring
- **Statistical Plots**: Comprehensive analysis dashboards and comparison plots

## 🛠️ Installation

MEA-Flow uses modern Python packaging with `uv` for fast dependency management.

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Quick Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install MEA-Flow
git clone https://github.com/CNNC-Lab/mea-flow.git
cd mea-flow

# Install with all dependencies
uv sync

# Verify installation
uv run python -c "import mea_flow; print('MEA-Flow installed successfully!')"
```

### Alternative Installation (pip)

```bash
# Clone repository
git clone https://github.com/CNNC-Lab/mea-flow.git
cd mea-flow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install package
pip install -e .
```

### Core Dependencies
- **Scientific Computing**: numpy, scipy, pandas, scikit-learn
- **Machine Learning**: xgboost, lightgbm, imbalanced-learn, statsmodels
- **Feature Selection**: mlxtend, skrebate, boruta
- **Visualization**: matplotlib, seaborn
- **Manifold Learning**: umap-learn
- **Data I/O**: h5py, tqdm, joblib

### Optional Dependencies
- **PySpike**: Advanced spike train distance measures (install with `[full]`)

For detailed installation instructions, see [`INSTALLATION_GUIDE.md`](INSTALLATION_GUIDE.md).

## 🚀 Quick Example

```python
from mea_flow.data import load_data
from mea_flow.analysis import MEAMetrics
from mea_flow.visualization import MEAPlotter

# Load Axion .spk file (native Python, no MATLAB required)
spike_list = load_data('recording.spk')

# Compute comprehensive neural activity metrics
analyzer = MEAMetrics()
metrics = analyzer.compute_all_metrics(spike_list)

# Create publication-ready visualizations
plotter = MEAPlotter()
fig = plotter.plot_raster(spike_list, time_range=(0, 30))
```

## 📖 Documentation & Tutorials

### Interactive Notebooks

1. **[Data Loading](notebooks/01_data_loading.ipynb)** - Native .spk file loading and format conversion
2. **[Activity Visualization](notebooks/02_activity_visualization.ipynb)** - Raster plots and electrode mapping
3. **[Activity Analysis](notebooks/03_activity_analysis.ipynb)** - Comprehensive neural metrics computation
4. **[Feature-Based Analysis](notebooks/04_feature_based_analysis.ipynb)** - Feature selection and machine learning
5. **[Manifold Analysis](notebooks/05_manifold_analysis.ipynb)** - Population dynamics and dimensionality reduction

### Method References

- **[Activity Analysis Methods](notebooks/ACTIVITY_ANALYSIS_METHODS_REFERENCE.md)** - Mathematical descriptions of neural metrics
- **[Feature Analysis Methods](notebooks/FEATURE_ANALYSIS_METHODS_REFERENCE.md)** - Feature selection and importance algorithms
- **[Manifold Methods](notebooks/MANIFOLD_METHODS_DOCUMENTATION.md)** - Dimensionality reduction techniques
- **[Clustering Methods](notebooks/CLUSTERING_METHODS_DOCUMENTATION.md)** - Clustering algorithms for neural data

### Installation & Setup

- **[Installation Guide](INSTALLATION_GUIDE.md)** - Detailed setup instructions
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Production deployment information

## 🔬 Analysis Workflows

### 1. Basic Neural Activity Analysis

```python
# Load and analyze neural activity
from mea_flow.data import load_data
from mea_flow.analysis import MEAMetrics

spike_list = load_data('recording.spk')
analyzer = MEAMetrics()
metrics = analyzer.compute_all_metrics(spike_list)
```

### 2. Feature-Based Condition Comparison

```python
# Multi-condition feature analysis
from mea_flow.analysis import comprehensive_feature_analysis

# Load pre-computed metrics DataFrame
results = comprehensive_feature_analysis(
    data=metrics_df,
    target_column='condition',
    methods=['mutual_info', 'random_forest', 'lasso']
)
```

### 3. Population Dynamics Analysis

```python
# Manifold learning for population geometry
from mea_flow.manifold import analyze_feature_space

embedding_results = analyze_feature_space(
    data=metrics_df,
    methods=['pca', 'umap', 'tsne'],
    color_column='condition'
)
```

## 📊 Project Scope & Applications

MEA-Flow is designed for neuroscience researchers working with multi-electrode array recordings. The package addresses key challenges in neural population analysis:

- **Data Format Compatibility**: Native Python loading of Axion .spk files eliminates MATLAB dependencies
- **Comprehensive Metrics**: 70+ neural activity measures covering activity, regularity, synchrony, and burst dynamics
- **Feature Analysis**: Advanced machine learning approaches for identifying discriminative neural features
- **Population Dynamics**: Manifold learning techniques for understanding high-dimensional neural state spaces
- **Statistical Rigor**: Multiple validation methods and consensus approaches for robust scientific conclusions

### Experimental Applications
- Cross-condition comparisons (e.g., control vs. treatment)
- Time-course analysis of neural development or plasticity
- Drug screening and pharmacological studies
- Network connectivity and synchronization analysis
- Population-level neural dynamics characterization

## 📚 Citation & Publication

### Zenodo DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

*Note: DOI will be assigned upon first Zenodo release*

### How to Cite

If you use MEA-Flow in your research, please cite:

```bibtex
@software{mea_flow_2024,
  title={MEA-Flow: Neural Population Dynamics Analysis for Multi-Electrode Arrays},
  author={MEA-Flow Development Team},
  year={2024},
  url={https://github.com/CNNC-Lab/mea-flow},
  doi={10.5281/zenodo.XXXXXXX},
  version={0.1.0}
}
```

### Publications Using MEA-Flow

*This section will be updated as research using MEA-Flow is published.*

## 🤝 Contributing

MEA-Flow is an open-source project welcoming contributions from the neuroscience community. Please see our contribution guidelines for details on:

- Bug reports and feature requests
- Code contributions and pull requests
- Documentation improvements
- Method validation and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

MEA-Flow development was supported by advances in computational neuroscience and the open-source scientific Python ecosystem. Special thanks to the developers of scikit-learn, matplotlib, and other foundational libraries that make this work possible.

---

**MEA-Flow**: Empowering neuroscience research through comprehensive multi-electrode array analysis.
