# MEA-Flow Notebooks

This directory contains interactive Jupyter notebooks demonstrating the complete MEA-Flow analysis pipeline, from data loading to advanced feature analysis.

## Notebook Overview

### [01_data_loading.ipynb](01_data_loading.ipynb)
**Data Loading and Format Conversion**

Learn how to load MEA data from various formats into MEA-Flow:
- **Native Axion .spk loader**: Reverse-engineered Python implementation that replaces MATLAB dependencies
- **MATLAB .mat files**: Load pre-processed spike data
- **CSV files**: Import data from other platforms
- **Batch loading**: Efficiently load multiple files at once

This notebook includes technical details about the reverse-engineering process that achieved 100% accuracy with MATLAB outputs.

---

### [02_activity_visualization.ipynb](02_activity_visualization.ipynb)
**Population Activity Visualization**

Explore comprehensive visualization methods for MEA data:
- **Raster plots**: Spike timing visualization with time windows
- **Well activity plots (PSTH)**: Population firing rate over time
- **Electrode spatial maps**: Activity distribution across the electrode grid
- **Animated visualizations**: Dynamic raster plots and electrode maps
- **Multi-condition comparisons**: Side-by-side well grid comparisons
- **Manifold embeddings**: PCA and other dimensionality reduction visualizations

Perfect for initial data exploration and presentation-quality figures.

---

### [03_activity_analysis.ipynb](03_activity_analysis.ipynb)
**Comprehensive MEA Metrics Analysis**

Compute and compare 70+ statistical metrics across experimental conditions:

**Metric Categories:**
- **Activity metrics**: Firing rates, spike counts, network activity
- **Regularity metrics**: CV-ISI, local variation, entropy
- **Synchrony metrics**: Cross-correlations, van Rossum distance, PySpike measures
- **Burst metrics**: Individual and network bursts, burst participation

**Analysis Groupings:**
- **Global**: Entire recording per condition
- **Well-based**: Individual well comparisons
- **Time-windowed**: Temporal dynamics analysis
- **Channel-level**: Single electrode statistics

Includes statistical comparisons and automated CSV export for all results.

---

### [04_feature_based_analysis.ipynb](04_feature_based_analysis.ipynb)
**Feature Importance and Machine Learning**

Advanced analysis to identify discriminant features and data structure:

**1. Exploratory Analysis:**
- **Dimensionality reduction**: PCA, MDS, t-SNE, UMAP, Spectral Embedding
- **Clustering analysis**: K-means, GMM, DBSCAN, Hierarchical clustering
- **Data structure visualization**: Manifold projections and cluster validation

**2. Feature Importance Pipeline:**
- **Redundancy detection**: VIF, correlation analysis, variance thresholding
- **Feature selection**: 11 different methods (ANOVA, Random Forest, LASSO, LDA, SVM, XGBoost, Ridge, Elastic Net, mRMR, ReliefF)
- **Consensus ranking**: Multi-method agreement and stability analysis
- **Critical feature identification**: Data-driven selection of most informative metrics

**3. Classification Analysis:**
- **Model training**: Predict experimental conditions from metrics
- **Feature validation**: Stability testing and cross-validation
- **Dimension reduction**: Focus analysis on critical features only

Includes complete visualization suite and export functionality.

---

### [05_manifold_analysis.ipynb](05_manifold_analysis.ipynb)
**Population Dynamics and Manifold Learning**

Analyze the geometric structure of neural population activity in state space:

**Methods Implemented:**
- Principal Component Analysis (PCA)
- Multidimensional Scaling (MDS)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Spectral Embedding
- Isomap
- Locally Linear Embedding (LLE)

**Quality Metrics:**
- Trustworthiness and continuity (local/global structure preservation)
- Stress and reconstruction error
- Distance correlation

**Cross-Condition Comparisons:**
- Classification accuracy in embedding space
- Procrustes analysis (manifold alignment)
- Effective dimensionality estimation
- Population statistics comparison

**Signal Preprocessing:**
- Spike train to continuous signal conversion
- Exponential filtering with configurable time constants
- Downsampling strategies for computational efficiency

---

## Recommended Workflow

1. **Start with [01_data_loading.ipynb](01_data_loading.ipynb)** to load your data
2. **Use [02_activity_visualization.ipynb](02_activity_visualization.ipynb)** for initial exploration
3. **Run [03_activity_analysis.ipynb](03_activity_analysis.ipynb)** to compute comprehensive metrics
4. **Apply [04_feature_based_analysis.ipynb](04_feature_based_analysis.ipynb)** to identify key discriminant features
5. **Explore [05_manifold_analysis.ipynb](05_manifold_analysis.ipynb)** for population dynamics insights

## Requirements

All notebooks require MEA-Flow to be installed. See the main project README for installation instructions.

Optional dependencies for full functionality:
- PySpike (for spike distance metrics)
- UMAP-learn (for UMAP embeddings)
- XGBoost (for feature selection)

## Data

Example notebooks use MEA recordings from three experimental conditions, obtained from the study cited below:
- Control
- Chronic stress
- miR-186-5p inhibition

Files referenced: `n1-DIV17-01`, `n2-DIV17-01`, `n3-DIV17-01` (in .spk, .mat, or .csv formats). Note that we do not provide
the reference data, only the notebooks as examples. To obtain the original datasets, please contact the study's authors.

### Citation

The example data used in these notebooks is from the following published research:

> Rodrigues, B., Leitão, R.A., Santos, M. et al. MiR-186-5p inhibition restores synaptic transmission and neuronal network activity in a model of chronic stress. *Molecular Psychiatry* **30**, 1034–1046 (2025). https://doi.org/10.1038/s41380-024-02715-1

This study examines how upregulation of miR-186-5p triggered by chronic stress may be a key mediator of changes leading to synaptic dysfunction in hippocampal neurons.

---