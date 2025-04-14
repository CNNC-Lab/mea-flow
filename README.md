<img src="docs/assets/logo.png" width="500" height="600">

**Latent State Space Analysis of Micro-Electrode Array Data**

MEA Flow is a Python package for analyzing multi-electrode array (MEA) data with a focus on neural population dynamics through manifold learning techniques, dimensionality reduction, and state space analysis.

## Features

- MEA-specific data loading and preprocessing for various recording systems (Axion, MultiChannel Systems, etc.)
- Advanced manifold learning and dimensionality reduction techniques
- State space analysis of neural population dynamics
- Specialized visualizations for MEA data and latent space representations
- Pipelines for comparing experimental conditions

## Installation

MEA Flow can be installed using `uv`, a fast Python package installer and environment manager.

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv

If you don't have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### Option 1: Install from PyPI
```bash
# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
# .venv\Scripts\activate   # On Windows

# Install MEA Flow
uv pip install mea-flow
```

#### Option 2: Install from source (for development)
```bash
# Clone the repository
git clone https://github.com/username/mea-flow.git
cd mea-flow

# Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
# .venv\Scripts\activate   # On Windows

# Install dependencies and development packages
uv pip install -r requirements.txt

# Install the package in development mode
uv pip install -e .
```

### Dependencies
MEA Flow depends on [neurolytics](https://github.com/CoNiC-project/neurolytics), a general-purpose neural data analysis package. It will be automatically installed when you install MEA Flow.

## Example Usage

```python
import neurolytics as nl
import mea_flow as mf

# Load MEA data
mea_data = mf.mea_io.load_axion_file("path/to/data.raw")

# Preprocess
filtered_data = nl.signals.filter(mea_data, low_cutoff=200, high_cutoff=3000)
spike_data = nl.signals.detect_spikes(filtered_data, threshold="auto")

# Separate wells
wells = mf.mea_io.segregate_wells(spike_data)

# Create population objects
populations = [nl.signals.Population(well) for well in wells]

# Extract manifolds
manifolds = [
    mf.manifold.extract_manifold(pop, method="umap", n_components=3)
    for pop in populations
]

# Visualize
mf.mea_viz.plot_manifold_comparison(manifolds, labels=["Control", "Treatment"])

# Analyze state dynamics
dynamics = mf.state_space.analyze_dynamics(manifolds[0])
```


## Documentation
Full documentation is available at https://mea-flow.readthedocs.io.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Install development dependencies (uv pip install -r requirements.txt)
4. Make your changes
5. Run tests (pytest)
6. Commit your changes (git commit -m 'Add some amazing feature')
7. Push to the branch (git push origin feature/amazing-feature)
8. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The neurolytics team for providing the core neural data analysis framework
The CoNiC project for inspiration and guidance
