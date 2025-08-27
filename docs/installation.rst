Installation Guide
==================

MEA-Flow supports multiple installation methods to accommodate different user preferences and system configurations.

Prerequisites
-------------

- Python 3.10 or higher
- pip or uv package manager (uv recommended for development)

Quick Installation
------------------

**Option 1: Basic Installation (Recommended)**

For most users, the basic installation provides all core functionality:

.. code-block:: bash

   pip install mea-flow

**Option 2: Using uv (Modern Python Package Manager)**

.. code-block:: bash

   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install MEA-Flow
   uv add mea-flow

**Option 3: Development Installation**

For development or to get the latest features:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/CNNC-Lab/mea-flow.git
   cd mea-flow
   
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .

Installation Options
--------------------

**Core Dependencies (Always Installed)**
   - numpy, scipy, pandas
   - matplotlib, seaborn
   - scikit-learn
   - h5py, tqdm, joblib

**Optional Dependencies**
   
   For enhanced functionality, install optional dependencies:

   .. code-block:: bash

      # Install with all optional features
      pip install mea-flow[full]
      
      # Or with uv
      uv add mea-flow --extra full

   Optional features include:
   
   - **PySpike**: Advanced spike train distance measures
   - **UMAP**: UMAP manifold learning method

**Development Dependencies**

   For development, testing, and documentation:

   .. code-block:: bash

      # Using pip
      pip install mea-flow[dev,notebooks]
      
      # Using uv dependency groups
      uv sync --group dev

Verification
------------

Verify your installation by running the verification script:

.. code-block:: bash

   python -c "import mea_flow; print('MEA-Flow installed successfully!')"

Or use the comprehensive verification script:

.. code-block:: bash

   # In the cloned repository
   python verify_install.py

The verification script will:

- Check all core imports
- Verify dependencies
- Test basic functionality
- Report optional dependency status

Troubleshooting
---------------

**PySpike Installation Issues**

PySpike requires compilation and may fail on some systems. If you encounter build errors:

1. **Use basic installation** (PySpike is optional):
   
   .. code-block:: bash

      pip install mea-flow

2. **Alternative: Install PySpike separately**:
   
   .. code-block:: bash

      pip install mea-flow
      # Try PySpike installation separately
      pip install pyspike

**UMAP Installation Issues**

If UMAP installation fails:

.. code-block:: bash

   pip install mea-flow
   # Try UMAP separately with conda if pip fails
   conda install umap-learn

**uv.lock Parse Errors**

If you encounter uv.lock parsing errors:

.. code-block:: bash

   rm uv.lock
   uv lock

**Virtual Environment Issues**

Create a clean virtual environment:

.. code-block:: bash

   # Using venv
   python -m venv mea_env
   source mea_env/bin/activate  # Linux/macOS
   # mea_env\\Scripts\\activate  # Windows
   
   # Using uv
   uv venv
   source .venv/bin/activate

System-Specific Notes
---------------------

**Linux**
   
   Install system dependencies for compilation (if using optional dependencies):
   
   .. code-block:: bash

      sudo apt-get install build-essential python3-dev

**macOS**
   
   Install Xcode command line tools:
   
   .. code-block:: bash

      xcode-select --install

**Windows**
   
   Install Microsoft Visual C++ Build Tools if compilation is needed.

Docker Installation
-------------------

For reproducible environments, use the provided Docker configuration:

.. code-block:: bash

   # Build the Docker image
   docker build -t mea-flow .
   
   # Run with Jupyter
   docker run -p 8888:8888 mea-flow jupyter lab --ip=0.0.0.0

Next Steps
----------

After successful installation:

1. **Quick Start**: Follow the :doc:`quickstart` guide
2. **Tutorial**: Work through the :doc:`notebooks/01_mea_flow_tutorial`
3. **Examples**: Explore the :doc:`examples/index`
4. **API Reference**: Browse the :doc:`api/data` documentation