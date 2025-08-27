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

