"""
Pytest configuration and fixtures for MEA-Flow tests.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

from mea_flow.data import SpikeList


@pytest.fixture
def random_seed():
    """Fix random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def simple_spike_data(random_seed):
    """Create simple synthetic spike data."""
    n_channels = 8
    n_spikes_per_channel = 20
    recording_length = 5.0  # seconds
    
    spike_times = []
    channel_ids = []
    
    for ch in range(n_channels):
        # Generate random spike times
        times = np.sort(np.random.uniform(0, recording_length, n_spikes_per_channel))
        spike_times.extend(times)
        channel_ids.extend([ch] * len(times))
    
    return {
        'times': np.array(spike_times),
        'channels': np.array(channel_ids),
        'recording_length': recording_length,
        'n_channels': n_channels
    }


@pytest.fixture
def complex_spike_data(random_seed):
    """Create more complex synthetic spike data with bursts."""
    n_channels = 16
    recording_length = 10.0
    
    spike_times = []
    channel_ids = []
    
    for ch in range(n_channels):
        # Generate baseline activity
        n_baseline = np.random.poisson(30)  # 30 spikes baseline
        baseline_times = np.sort(np.random.uniform(0, recording_length, n_baseline))
        
        # Add burst activity for some channels
        if ch < 8:  # First 8 channels have bursts
            n_bursts = np.random.randint(2, 5)
            for _ in range(n_bursts):
                burst_start = np.random.uniform(1, recording_length - 1)
                burst_duration = np.random.uniform(0.1, 0.3)
                n_burst_spikes = np.random.poisson(15)
                
                burst_times = np.sort(np.random.uniform(
                    burst_start, 
                    burst_start + burst_duration, 
                    n_burst_spikes
                ))
                baseline_times = np.concatenate([baseline_times, burst_times])
        
        baseline_times = np.sort(baseline_times)
        spike_times.extend(baseline_times)
        channel_ids.extend([ch] * len(baseline_times))
    
    return {
        'times': np.array(spike_times),
        'channels': np.array(channel_ids),
        'recording_length': recording_length,
        'n_channels': n_channels
    }


@pytest.fixture
def multi_well_spike_data(random_seed):
    """Create spike data with well organization (4x4 grid)."""
    n_wells = 4  # 2x2 grid for testing
    channels_per_well = 4  # 2x2 electrodes per well
    recording_length = 8.0
    
    spike_times = []
    channel_ids = []
    
    # Create well mapping (channels 0-3 = well 0, 4-7 = well 1, etc.)
    well_map = {}
    for well_id in range(n_wells):
        well_channels = list(range(well_id * channels_per_well, (well_id + 1) * channels_per_well))
        well_map[well_id] = well_channels
    
    total_channels = n_wells * channels_per_well
    
    for ch in range(total_channels):
        well_id = ch // channels_per_well
        
        # Different activity levels per well
        if well_id == 0:  # High activity well
            n_spikes = np.random.poisson(80)
        elif well_id == 1:  # Medium activity well
            n_spikes = np.random.poisson(40)
        elif well_id == 2:  # Low activity well
            n_spikes = np.random.poisson(10)
        else:  # Very low activity well
            n_spikes = np.random.poisson(5)
        
        times = np.sort(np.random.uniform(0, recording_length, n_spikes))
        spike_times.extend(times)
        channel_ids.extend([ch] * len(times))
    
    return {
        'times': np.array(spike_times),
        'channels': np.array(channel_ids),
        'recording_length': recording_length,
        'n_channels': total_channels,
        'well_map': well_map
    }


@pytest.fixture
def spike_list_simple(simple_spike_data):
    """Create SpikeList from simple spike data."""
    return SpikeList(
        spike_data=simple_spike_data,
        recording_length=simple_spike_data['recording_length']
    )


@pytest.fixture
def spike_list_complex(complex_spike_data):
    """Create SpikeList from complex spike data."""
    return SpikeList(
        spike_data=complex_spike_data,
        recording_length=complex_spike_data['recording_length']
    )


@pytest.fixture
def spike_list_multi_well(multi_well_spike_data):
    """Create SpikeList with well mapping."""
    return SpikeList(
        spike_data=multi_well_spike_data,
        recording_length=multi_well_spike_data['recording_length'],
        well_map=multi_well_spike_data['well_map']
    )


@pytest.fixture
def empty_spike_list():
    """Create empty SpikeList for testing edge cases."""
    return SpikeList(
        spike_data={'times': np.array([]), 'channels': np.array([])},
        recording_length=1.0
    )


@pytest.fixture
def matlab_test_data(temp_dir):
    """Create mock MATLAB data files for testing loaders."""
    # This would normally require scipy.io.savemat, but we'll create
    # a dictionary structure that matches what loadmat returns
    data = {
        'Channels': np.array([0, 0, 1, 1, 2, 2]),
        'Times': np.array([0.1, 0.5, 0.2, 0.8, 0.3, 0.9])
    }
    return data


@pytest.fixture
def csv_test_data(temp_dir):
    """Create CSV test data file."""
    csv_file = temp_dir / "test_spikes.csv"
    df = pd.DataFrame({
        'channel': [0, 0, 1, 1, 2, 2],
        'time': [0.1, 0.5, 0.2, 0.8, 0.3, 0.9]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def analysis_config():
    """Create standard analysis configuration for testing."""
    return type('Config', (), {
        'min_spikes': 10,
        'max_isi': 2.0,
        'burst_threshold': 0.1,
        'sync_time_bin': 0.01,
        'tau_van_rossum': 0.02,
        'n_pairs_sync': 100,
        'methods': ['PCA', 'MDS'],
        'max_components': 3,
        'dt': 0.001
    })()


# Test data categories for parameterized tests
@pytest.fixture(params=['simple', 'complex', 'multi_well'])
def all_spike_lists(request, spike_list_simple, spike_list_complex, spike_list_multi_well):
    """Parameterized fixture to test all spike list types."""
    spike_lists = {
        'simple': spike_list_simple,
        'complex': spike_list_complex,
        'multi_well': spike_list_multi_well
    }
    return spike_lists[request.param]


@pytest.fixture(params=['global', 'channel', 'well'])
def grouping_methods(request):
    """Parameterized fixture for different grouping methods.""" 
    return request.param


# Custom markers for test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "data: marks tests that test data handling")
    config.addinivalue_line("markers", "analysis: marks tests that test analysis functions")
    config.addinivalue_line("markers", "visualization: marks tests that test plotting")
    config.addinivalue_line("markers", "optional: marks tests that require optional dependencies")