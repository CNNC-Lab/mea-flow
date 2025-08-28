"""
Tests for MEA-Flow data handling modules.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from mea_flow.data import SpikeList
from mea_flow.data.loaders import load_csv_file, load_from_dataframe

# Create aliases for consistency
load_csv_data = load_csv_file
load_dataframe_data = load_from_dataframe
from mea_flow.data.spike_list import SpikeTrain


class TestSpikeTrain:
    """Test SpikeTrain class functionality."""
    
    @pytest.mark.unit
    def test_spike_train_creation(self):
        """Test SpikeTrain creation and basic properties."""
        spike_times = np.array([0.1, 0.5, 1.2, 2.3])
        train = SpikeTrain(channel_id=0, spike_times=spike_times, recording_length=5.0)
        
        assert len(train.spike_times) == 4
        assert train.n_spikes == 4
        assert np.allclose(train.spike_times, spike_times)
        assert train.channel_id == 0
        assert train.recording_length == 5.0
    
    @pytest.mark.unit
    def test_spike_train_empty(self):
        """Test SpikeTrain with no spikes."""
        train = SpikeTrain(channel_id=0, spike_times=np.array([]), recording_length=1.0)
        
        assert train.n_spikes == 0
        assert len(train.spike_times) == 0
    
    @pytest.mark.unit
    def test_spike_train_sorting(self):
        """Test that spike times are automatically sorted."""
        unsorted_times = np.array([1.5, 0.3, 2.1, 0.8])
        train = SpikeTrain(channel_id=0, spike_times=unsorted_times, recording_length=3.0)
        
        assert np.array_equal(train.spike_times, np.array([0.3, 0.8, 1.5, 2.1]))
    
    @pytest.mark.unit
    def test_spike_train_isi(self):
        """Test inter-spike interval calculation."""
        spike_times = np.array([0.0, 0.1, 0.3, 0.7])
        train = SpikeTrain(channel_id=0, spike_times=spike_times, recording_length=1.0)
        
        isis = train.get_isi()
        expected_isis = np.array([0.1, 0.2, 0.4])
        assert np.allclose(isis, expected_isis)
    
    @pytest.mark.unit
    def test_spike_train_isi_empty(self):
        """Test ISI calculation with no or single spike."""
        # Empty train
        train_empty = SpikeTrain(channel_id=0, spike_times=np.array([]), recording_length=1.0)
        assert len(train_empty.get_isi()) == 0
        
        # Single spike
        train_single = SpikeTrain(channel_id=0, spike_times=np.array([1.0]), recording_length=2.0)
        assert len(train_single.get_isi()) == 0
    
    @pytest.mark.unit
    def test_spike_train_firing_rate(self):
        """Test firing rate calculation."""
        spike_times = np.array([0.1, 0.5, 1.2, 2.3])  # 4 spikes
        train = SpikeTrain(channel_id=0, spike_times=spike_times, recording_length=3.0)
        
        # Test firing rate property
        rate = train.firing_rate
        assert np.isclose(rate, 4.0/3.0)
    



class TestSpikeList:
    """Test SpikeList class functionality."""
    
    @pytest.mark.unit
    def test_spike_list_creation_dict(self, simple_spike_data):
        """Test SpikeList creation from dictionary."""
        spike_list = SpikeList(
            spike_data=simple_spike_data,
            recording_length=simple_spike_data['recording_length']
        )
        
        assert spike_list.recording_length == simple_spike_data['recording_length']
        assert len(spike_list.spike_trains) == simple_spike_data['n_channels']
        assert spike_list.n_channels == simple_spike_data['n_channels']
    
    @pytest.mark.unit
    def test_spike_list_creation_arrays(self):
        """Test SpikeList creation from separate arrays."""
        times = np.array([0.1, 0.5, 1.2])
        channels = np.array([0, 1, 0])
        
        spike_list = SpikeList(
            spike_data={'times': times, 'channels': channels},
            recording_length=2.0
        )
        
        assert spike_list.n_channels == 2
        assert 0 in spike_list.spike_trains
        assert 1 in spike_list.spike_trains
        assert spike_list.spike_trains[0].n_spikes == 2  # Channel 0 has 2 spikes
        assert spike_list.spike_trains[1].n_spikes == 1  # Channel 1 has 1 spike
    
    @pytest.mark.unit
    def test_spike_list_well_mapping(self, multi_well_spike_data):
        """Test SpikeList with well mapping."""
        spike_list = SpikeList(
            spike_data=multi_well_spike_data,
            recording_length=multi_well_spike_data['recording_length'],
            well_map=multi_well_spike_data['well_map']
        )
        
        assert spike_list.well_map is not None
        assert len(spike_list.well_map) == 4  # 4 wells
        
        # Test well access
        well_0_channels = spike_list.get_well_channels(0)
        expected_channels = [0, 1, 2, 3]
        assert well_0_channels == expected_channels
    
    @pytest.mark.unit
    def test_spike_list_active_channels(self, spike_list_complex):
        """Test active channel identification."""
        active_channels = spike_list_complex.get_active_channels(min_spikes=5)
        
        assert isinstance(active_channels, list)
        assert len(active_channels) > 0
        
        # All returned channels should have at least min_spikes
        for ch in active_channels:
            assert spike_list_complex.spike_trains[ch].n_spikes >= 5
    
    @pytest.mark.unit
    def test_spike_list_time_slice(self, spike_list_complex):
        """Test time slicing of entire spike list."""
        start_time = 2.0
        end_time = 6.0
        
        sliced = spike_list_complex.time_slice(start_time, end_time)
        
        assert sliced.recording_length == end_time - start_time
        
        # Check that all spikes are within the time range
        for ch_id, train in sliced.spike_trains.items():
            if train.n_spikes > 0:
                assert np.all(train.spike_times >= 0)  # Relative to slice start
                assert np.all(train.spike_times <= end_time - start_time)
    
    @pytest.mark.unit
    def test_spike_list_bin_spikes(self, spike_list_simple):
        """Test spike binning functionality."""
        bin_size = 0.1  # 100ms bins
        spike_matrix, time_bins = spike_list_simple.bin_spikes(bin_size)
        
        n_channels = len(spike_list_simple.get_active_channels())
        expected_n_bins = int(np.ceil(spike_list_simple.recording_length / bin_size))
        
        assert spike_matrix.shape[0] == n_channels
        assert spike_matrix.shape[1] == expected_n_bins
        assert len(time_bins) == expected_n_bins + 1
        
        # Check that spike counts are non-negative integers
        assert np.all(spike_matrix >= 0)
        assert np.all(spike_matrix == spike_matrix.astype(int))
    
    @pytest.mark.unit
    def test_spike_list_continuous_signal(self, spike_list_simple):
        """Test conversion to continuous signal."""
        tau = 0.02
        dt = 0.001
        
        signals, time_vector = spike_list_simple.to_continuous_signal(tau=tau, dt=dt)
        
        n_channels = len(spike_list_simple.get_active_channels())
        expected_n_points = int(spike_list_simple.recording_length / dt) + 1
        
        assert signals.shape[0] == n_channels
        assert signals.shape[1] == expected_n_points
        assert len(time_vector) == expected_n_points
        
        # Check that signals are non-negative (exponential decay)
        assert np.all(signals >= 0)
    
    @pytest.mark.unit
    def test_spike_list_empty(self, empty_spike_list):
        """Test SpikeList with no spikes."""
        assert empty_spike_list.n_channels == 0
        assert len(empty_spike_list.spike_trains) == 0
        assert len(empty_spike_list.get_active_channels()) == 0
    
    @pytest.mark.unit
    def test_spike_list_statistics(self, spike_list_complex):
        """Test basic statistics calculation."""
        stats = spike_list_complex.get_statistics()
        
        assert 'total_spikes' in stats
        assert 'n_channels' in stats
        assert 'recording_length' in stats
        assert 'mean_firing_rate' in stats
        
        assert stats['total_spikes'] > 0
        assert stats['n_channels'] > 0
        assert stats['recording_length'] > 0


class TestDataLoaders:
    """Test data loading functions."""
    
    @pytest.mark.unit
    def test_load_csv_data(self, csv_test_data):
        """Test CSV data loading."""
        spike_list = load_csv_data(
            csv_test_data,
            time_col='time',
            channel_col='channel',
            recording_length=1.0
        )
        
        assert isinstance(spike_list, SpikeList)
        assert spike_list.n_channels == 3  # Channels 0, 1, 2
        assert spike_list.recording_length == 1.0
    
    @pytest.mark.unit
    def test_load_dataframe_data(self):
        """Test DataFrame data loading."""
        df = pd.DataFrame({
            'channel': [0, 0, 1, 2],
            'time': [0.1, 0.5, 0.2, 0.8]
        })
        
        spike_list = load_dataframe_data(
            df,
            time_col='time',
            channel_col='channel',
            recording_length=1.0
        )
        
        assert isinstance(spike_list, SpikeList)
        assert spike_list.n_channels == 3
        assert spike_list.recording_length == 1.0
    
    @pytest.mark.unit
    @patch('mea_flow.data.loaders.loadmat')
    def test_load_matlab_file(self, mock_loadmat, matlab_test_data):
        """Test MATLAB file loading."""
        mock_loadmat.return_value = matlab_test_data
        
        from mea_flow.data.loaders import load_matlab_file
        
        spike_list = load_matlab_file(
            'dummy_file.mat',
            channels_key='Channels',
            times_key='Times',
            recording_length=1.0
        )
        
        assert isinstance(spike_list, SpikeList)
        mock_loadmat.assert_called_once()
    
    @pytest.mark.unit
    def test_invalid_data_handling(self):
        """Test handling of invalid input data."""
        # Test with mismatched array lengths
        with pytest.raises(ValueError):
            SpikeList(
                spike_data={
                    'times': np.array([0.1, 0.2]),
                    'channels': np.array([0])  # Wrong length
                },
                recording_length=1.0
            )
        
        # Test with negative recording length
        with pytest.raises(ValueError):
            SpikeList(
                spike_data={'times': np.array([]), 'channels': np.array([])},
                recording_length=-1.0
            )


class TestSpikeListMethods:
    """Test advanced SpikeList methods."""
    
    @pytest.mark.unit
    def test_channel_selection(self, spike_list_multi_well):
        """Test channel selection functionality."""
        selected_channels = [0, 1, 4, 5]
        filtered = spike_list_multi_well.select_channels(selected_channels)
        
        assert set(filtered.spike_trains.keys()).issubset(set(selected_channels))
    
    @pytest.mark.unit
    def test_well_selection(self, spike_list_multi_well):
        """Test well-based selection."""
        well_ids = [0, 1]
        filtered = spike_list_multi_well.select_wells(well_ids)
        
        expected_channels = []
        for well_id in well_ids:
            expected_channels.extend(spike_list_multi_well.get_well_channels(well_id))
        
        assert set(filtered.spike_trains.keys()).issubset(set(expected_channels))
    
    @pytest.mark.unit
    @pytest.mark.parametrize("grouping", ["global", "channel", "well"])
    def test_grouping_methods(self, spike_list_multi_well, grouping):
        """Test different grouping methods."""
        if grouping == "well" and spike_list_multi_well.well_map is None:
            pytest.skip("Well mapping not available")
        
        groups = spike_list_multi_well.get_analysis_groups(grouping)
        
        assert isinstance(groups, dict)
        assert len(groups) > 0
        
        for group_id, channels in groups.items():
            assert isinstance(channels, list)
            assert len(channels) > 0


# Integration tests
class TestDataIntegration:
    """Integration tests for data handling."""
    
    @pytest.mark.integration
    def test_full_data_pipeline(self, complex_spike_data, temp_dir):
        """Test complete data processing pipeline."""
        # Create SpikeList
        spike_list = SpikeList(
            spike_data=complex_spike_data,
            recording_length=complex_spike_data['recording_length']
        )
        
        # Test various operations
        active_channels = spike_list.get_active_channels(min_spikes=10)
        assert len(active_channels) > 0
        
        # Test time slicing
        sliced = spike_list.time_slice(1.0, 5.0)
        assert sliced.recording_length == 4.0
        
        # Test binning
        spike_matrix, _ = spike_list.bin_spikes(0.1)
        assert spike_matrix.shape[0] == len(active_channels)
        
        # Test continuous conversion
        signals, _ = spike_list.to_continuous_signal(tau=0.02, dt=0.001)
        assert signals.shape[0] == len(active_channels)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_handling(self, random_seed):
        """Test handling of large datasets."""
        # Create large dataset
        n_channels = 64
        n_spikes_total = 50000
        recording_length = 60.0
        
        # Generate random data
        spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_total))
        channels = np.random.randint(0, n_channels, n_spikes_total)
        
        spike_list = SpikeList(
            spike_data={'times': spike_times, 'channels': channels},
            recording_length=recording_length
        )
        
        assert spike_list.n_channels == n_channels
        
        # Test that operations complete without error
        active_channels = spike_list.get_active_channels()
        spike_matrix, _ = spike_list.bin_spikes(0.1)
        
        assert len(active_channels) <= n_channels
        assert spike_matrix.shape[0] == len(active_channels)