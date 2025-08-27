#!/usr/bin/env python3
"""
Installation verification script for MEA-Flow.

This script checks that all core components are working correctly
and reports the status of optional dependencies.
"""

import sys
import warnings

def check_core_imports():
    """Check core MEA-Flow imports."""
    print("🔍 Checking core imports...")
    
    try:
        import mea_flow
        print("✅ mea_flow imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import mea_flow: {e}")
        return False
    
    try:
        from mea_flow import SpikeList, MEAMetrics, ManifoldAnalysis, MEAPlotter
        print("✅ Core classes imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core classes: {e}")
        return False
    
    try:
        from mea_flow.data import loaders
        from mea_flow.analysis import metrics, activity, synchrony, regularity, burst_analysis
        from mea_flow.manifold import analysis, embedding
        from mea_flow.visualization import plotter
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import modules: {e}")
        return False
    
    return True

def check_dependencies():
    """Check status of core and optional dependencies."""
    print("\n📦 Checking dependencies...")
    
    # Core dependencies
    core_deps = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'sklearn',
        'h5py': 'h5py',
        'tqdm': 'tqdm',
        'joblib': 'joblib'
    }
    
    all_core_ok = True
    for name, import_name in core_deps.items():
        try:
            __import__(import_name)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            all_core_ok = False
    
    # Optional dependencies
    print("\n🔧 Optional dependencies:")
    
    try:
        import pyspike
        print("✅ PySpike - Advanced spike train distance measures available")
    except ImportError:
        print("⚠️  PySpike - Not installed (spike distances will use fallback methods)")
    
    try:
        import umap
        print("✅ UMAP - UMAP manifold learning available")
    except ImportError:
        print("⚠️  UMAP - Not installed (UMAP embedding will be disabled)")
    
    return all_core_ok

def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import numpy as np
        from mea_flow import SpikeList, MEAMetrics
        
        # Create simple synthetic data
        np.random.seed(42)
        n_channels = 16
        n_spikes_per_channel = 50
        recording_length = 10.0  # seconds
        
        spike_data = []
        channel_ids = []
        
        for ch in range(n_channels):
            # Generate random spike times
            spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_per_channel))
            spike_data.extend(spike_times)
            channel_ids.extend([ch] * len(spike_times))
        
        # Create SpikeList
        spike_list = SpikeList(
            spike_data={'times': spike_data, 'channels': channel_ids},
            recording_length=recording_length
        )
        print("✅ SpikeList creation successful")
        
        # Test metrics calculation
        metrics = MEAMetrics()
        results = metrics.compute_all_metrics(spike_list, grouping='global')
        print("✅ Metrics calculation successful")
        
        # Test manifold analysis
        from mea_flow import ManifoldAnalysis
        manifold = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pop_results = manifold.analyze_population_dynamics(spike_list)
        print("✅ Manifold analysis successful")
        
        print("🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 MEA-Flow Installation Verification")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.10+ required.")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Run checks
    core_ok = check_core_imports()
    deps_ok = check_dependencies()
    func_ok = test_basic_functionality() if core_ok else False
    
    print("\n" + "=" * 40)
    if core_ok and deps_ok and func_ok:
        print("🎉 MEA-Flow installation verified successfully!")
        print("\n💡 Next steps:")
        print("   - Check out the tutorial: notebooks/01_mea_flow_tutorial.ipynb")
        print("   - Read the documentation in docs/")
        print("   - Try analyzing your MEA data!")
        return True
    else:
        print("❌ Installation verification failed.")
        print("\n🔧 Troubleshooting:")
        if not core_ok:
            print("   - Reinstall MEA-Flow: uv pip install -e .")
        if not deps_ok:
            print("   - Check missing dependencies and install if needed")
        print("   - Check the README.md for installation instructions")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)