# MEA-Flow Installation Guide

## 🛠️ Fixed Installation (No More PySpike Issues!)

### Problem Solved ✅
The PySpike compilation issue has been resolved by making it an optional dependency. MEA-Flow now works perfectly without PySpike!

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/CNNC-Lab/mea-flow.git
cd mea-flow

# Method 1: Using uv (Recommended)
uv sync
uv run python verify_install.py

# Method 2: Using pip
pip install -e .
python verify_install.py
```

### Verification

After installation, run the verification script:
```bash
python verify_install.py
```

You should see:
```
🎉 MEA-Flow installation verified successfully!
```

### What Works Without PySpike

✅ **All Core Features**:
- Load MEA data (.spk, .mat, CSV, HDF5)
- Activity metrics (firing rates, burst detection)
- Regularity metrics (CV-ISI, entropy, Fano factors)
- Synchrony metrics (correlations, Van Rossum distances)
- Manifold learning (PCA, MDS, Isomap, LLE, t-SNE)
- Publication-ready visualizations
- Cross-condition statistical analysis

⚠️ **Only Missing** (optional):
- PySpike-specific distance measures (ISI distance, SPIKE distance)
- UMAP manifold learning (unless installed separately)

### Development Setup

For development with Jupyter support:

```bash
# Using uv (creates proper virtual environment)
uv sync --group dev
uv run jupyter lab

# Using pip
pip install -e ".[dev,notebooks]"
jupyter lab
```

### Optional: Install PySpike Later

If you need PySpike-specific features and have build tools available:

```bash
# Try installing PySpike separately
pip install pyspike

# Or install full version
uv sync --extra full
```

### Usage Example

```python
from mea_flow import SpikeList, MEAMetrics, ManifoldAnalysis, MEAPlotter

# Load your MEA data
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
```

### Next Steps

1. ✅ Run `python verify_install.py` to confirm installation
2. 📓 Check out `notebooks/01_mea_flow_tutorial.ipynb` for comprehensive tutorial
3. 📊 Start analyzing your MEA data!
4. 📚 Read the documentation in `docs/` folder

### Support

If you encounter any issues:
1. Check this installation guide
2. Review the main README.md
3. Run the verification script for diagnostics
4. Check the GitHub issues page

**The library is now fully functional for MEA analysis!** 🚀