"""
Tests for MEA-Flow analysis modules.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch

from mea_flow.analysis import MEAMetrics
from mea_flow.analysis.activity import compute_activity_metrics, firing_rate, burst_detection
from mea_flow.analysis.regularity import compute_regularity_metrics, cv_isi, entropy_isi
from mea_flow.analysis.synchrony import compute_synchrony_metrics, pairwise_correlations
from mea_flow.analysis.burst_analysis import network_burst_analysis, burst_statistics


class TestActivityMetrics:
    """Test activity analysis functions."""
    
    @pytest.mark.unit
    def test_firing_rate_calculation(self, spike_list_simple):
        """Test basic firing rate calculation."""
        channels = spike_list_simple.get_active_channels()
        if len(channels) == 0:
            pytest.skip("No active channels in test data")
        
        channel = channels[0]
        rate = firing_rate(spike_list_simple, channel)
        
        assert rate >= 0
        assert isinstance(rate, (int, float))
        
        # Test with specific recording length
        rate_custom = firing_rate(spike_list_simple, channel, recording_length=10.0)
        assert rate_custom >= 0
    
    @pytest.mark.unit
    def test_firing_rate_empty_channel(self, spike_list_simple):
        """Test firing rate with empty channel."""
        # Use a channel that doesn't exist
        rate = firing_rate(spike_list_simple, 999)
        assert rate == 0
    
    @pytest.mark.unit
    def test_burst_detection(self, spike_list_complex):
        """Test burst detection algorithm."""
        channels = spike_list_complex.get_active_channels(min_spikes=20)
        if len(channels) == 0:
            pytest.skip("No channels with sufficient spikes")
        
        channel = channels[0]
        bursts = burst_detection(spike_list_complex, channel, min_spikes=3, max_isi=0.1)
        
        assert isinstance(bursts, list)
        # Each burst should be a dictionary with start, end, spikes, etc.
        for burst in bursts:
            assert isinstance(burst, dict)
            assert 'start_time' in burst
            assert 'end_time' in burst
            assert 'n_spikes' in burst
            assert burst['start_time'] <= burst['end_time']
            assert burst['n_spikes'] >= 3
    
    @pytest.mark.unit
    def test_activity_metrics_computation(self, spike_list_complex, analysis_config):
        """Test complete activity metrics computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_activity_metrics(spike_list_complex, analysis_config)
        
        assert isinstance(metrics, dict)
        
        # Check for expected keys (may be NaN if no active channels)
        expected_keys = [
            'mean_firing_rate', 'std_firing_rate', 'total_spikes',
            'active_channels', 'burst_rate', 'spikes_in_bursts_ratio'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    @pytest.mark.unit
    def test_activity_metrics_empty_data(self, empty_spike_list, analysis_config):
        """Test activity metrics with empty data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_activity_metrics(empty_spike_list, analysis_config)
        
        assert isinstance(metrics, dict)
        assert metrics['total_spikes'] == 0
        assert metrics['active_channels'] == 0
        assert np.isnan(metrics['mean_firing_rate'])


class TestRegularityMetrics:
    """Test regularity analysis functions."""
    
    @pytest.mark.unit
    def test_cv_isi_calculation(self, spike_list_complex):
        """Test coefficient of variation of ISI."""
        channels = spike_list_complex.get_active_channels(min_spikes=10)
        if len(channels) == 0:
            pytest.skip("No channels with sufficient spikes")
        
        channel = channels[0]
        cv = cv_isi(spike_list_complex, channel)
        
        assert cv >= 0  # CV is always non-negative
        assert isinstance(cv, (int, float))
    
    @pytest.mark.unit
    def test_cv_isi_few_spikes(self, spike_list_simple):
        """Test CV-ISI with few spikes."""
        channels = spike_list_simple.get_active_channels()
        if len(channels) == 0:
            pytest.skip("No active channels")
        
        channel = channels[0]
        cv = cv_isi(spike_list_simple, channel)
        
        # Should return valid number or NaN for insufficient spikes
        assert isinstance(cv, (int, float)) or np.isnan(cv)
    
    @pytest.mark.unit
    def test_entropy_isi_calculation(self, spike_list_complex):
        """Test ISI entropy calculation."""
        channels = spike_list_complex.get_active_channels(min_spikes=10)
        if len(channels) == 0:
            pytest.skip("No channels with sufficient spikes")
        
        channel = channels[0]
        entropy = entropy_isi(spike_list_complex, channel, n_bins=10)
        
        assert entropy >= 0  # Entropy is non-negative
        assert isinstance(entropy, (int, float))
    
    @pytest.mark.unit
    def test_regularity_metrics_computation(self, spike_list_complex, analysis_config):
        """Test complete regularity metrics computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_regularity_metrics(spike_list_complex, analysis_config)
        
        assert isinstance(metrics, dict)
        
        expected_keys = [
            'mean_cv_isi', 'std_cv_isi', 'mean_entropy_isi', 
            'mean_lv', 'mean_fano_factor'
        ]
        
        for key in expected_keys:
            assert key in metrics


class TestSynchronyMetrics:
    """Test synchrony analysis functions."""
    
    @pytest.mark.unit
    def test_pairwise_correlations(self, spike_list_complex):
        """Test pairwise correlation calculation."""
        channels = spike_list_complex.get_active_channels(min_spikes=10)
        if len(channels) < 2:
            pytest.skip("Need at least 2 active channels")
        
        correlations = pairwise_correlations(
            spike_list_complex,
            channels=channels[:4],  # Limit to first 4 channels
            bin_size=0.1
        )
        
        assert isinstance(correlations, list)
        
        # All correlations should be between -1 and 1
        for corr in correlations:
            assert -1 <= corr <= 1
    
    @pytest.mark.unit
    def test_pairwise_correlations_insufficient_channels(self, spike_list_simple):
        """Test pairwise correlations with insufficient channels."""
        correlations = pairwise_correlations(
            spike_list_simple,
            channels=[0],  # Only one channel
            bin_size=0.1
        )
        
        assert isinstance(correlations, list)
        assert len(correlations) == 0
    
    @pytest.mark.unit
    def test_synchrony_metrics_computation(self, spike_list_complex, analysis_config):
        """Test complete synchrony metrics computation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_synchrony_metrics(spike_list_complex, analysis_config)
        
        assert isinstance(metrics, dict)
        
        expected_keys = [
            'pearson_cc_mean', 'pearson_cc_std', 'n_pairs_analyzed',
            'van_rossum_distance_mean', 'chi_square_distance',
            'cosyne_similarity', 'synchrony_index'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    @pytest.mark.optional
    @patch('mea_flow.analysis.synchrony.PYSPIKE_AVAILABLE', True)
    @patch('mea_flow.analysis.synchrony.pyspike')
    def test_pyspike_integration(self, mock_pyspike, spike_list_complex, analysis_config):
        """Test PySpike integration when available."""
        # Mock PySpike functions
        mock_pyspike.SpikeTrain.return_value = Mock()
        mock_pyspike.isi_distance.return_value = 0.5
        mock_pyspike.spike_distance.return_value = 0.3
        mock_pyspike.spike_sync.return_value = 0.7
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = compute_synchrony_metrics(spike_list_complex, analysis_config)
        
        # Should include PySpike metrics
        assert 'isi_distance' in metrics
        assert 'spike_distance' in metrics
        assert 'spike_sync_distance' in metrics


class TestBurstAnalysis:
    """Test burst analysis functions."""
    
    @pytest.mark.unit
    def test_burst_statistics(self, spike_list_complex):
        """Test burst statistics calculation."""
        channels = spike_list_complex.get_active_channels(min_spikes=20)
        if len(channels) == 0:
            pytest.skip("No channels with sufficient spikes")
        
        channel = channels[0]
        bursts = burst_detection(spike_list_complex, channel, min_spikes=3)
        
        if len(bursts) > 0:
            stats = burst_statistics(bursts)
            
            assert isinstance(stats, dict)
            assert 'n_bursts' in stats
            assert 'mean_burst_duration' in stats
            assert 'mean_interburst_interval' in stats
            assert stats['n_bursts'] == len(bursts)
        else:
            # If no bursts detected, should handle gracefully
            stats = burst_statistics([])
            assert stats['n_bursts'] == 0
    
    @pytest.mark.unit
    def test_network_burst_analysis(self, spike_list_complex, analysis_config):
        """Test network burst detection and analysis."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            network_bursts = network_burst_analysis(spike_list_complex, analysis_config)
        
        assert isinstance(network_bursts, dict)
        
        expected_keys = [
            'n_network_bursts', 'network_burst_rate', 
            'mean_network_burst_duration', 'network_burst_synchrony'
        ]
        
        for key in expected_keys:
            assert key in network_bursts


class TestMEAMetrics:
    """Test the main MEAMetrics orchestrator class."""
    
    @pytest.mark.unit
    def test_mea_metrics_initialization(self):
        """Test MEAMetrics class initialization."""
        metrics = MEAMetrics()
        assert metrics is not None
        
        # Test with custom config
        custom_config = type('Config', (), {'min_spikes': 5})()
        metrics_custom = MEAMetrics(config=custom_config)
        assert metrics_custom.config.min_spikes == 5
    
    @pytest.mark.unit
    @pytest.mark.parametrize("grouping", ["global", "channel"])
    def test_compute_all_metrics(self, spike_list_complex, grouping):
        """Test complete metrics computation with different groupings."""
        metrics = MEAMetrics()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(spike_list_complex, grouping=grouping)
        
        assert isinstance(results, dict)
        
        if grouping == "global":
            # Should return single set of metrics
            expected_sections = ['activity', 'regularity', 'synchrony', 'bursts']
            for section in expected_sections:
                assert section in results
        elif grouping == "channel":
            # Should return per-channel metrics
            assert isinstance(results, dict)
            # Results should be organized by channel
    
    @pytest.mark.unit
    def test_compute_all_metrics_well_grouping(self, spike_list_multi_well):
        """Test metrics computation with well grouping."""
        if spike_list_multi_well.well_map is None:
            pytest.skip("Well mapping not available")
        
        metrics = MEAMetrics()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(spike_list_multi_well, grouping="well")
        
        assert isinstance(results, dict)
        # Results should be organized by well
        for well_id in spike_list_multi_well.well_map.keys():
            # May or may not have results depending on activity
            pass
    
    @pytest.mark.unit
    def test_compute_metrics_empty_data(self, empty_spike_list):
        """Test metrics computation with empty data."""
        metrics = MEAMetrics()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(empty_spike_list, grouping="global")
        
        assert isinstance(results, dict)
        # Should handle empty data gracefully
        assert results['activity']['total_spikes'] == 0
    
    @pytest.mark.unit
    def test_compute_selective_metrics(self, spike_list_complex):
        """Test computing only specific metric categories."""
        metrics = MEAMetrics()
        
        # Test activity only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            activity_only = metrics.compute_activity_metrics(spike_list_complex)
        
        assert isinstance(activity_only, dict)
        assert 'mean_firing_rate' in activity_only
        
        # Test synchrony only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sync_only = metrics.compute_synchrony_metrics(spike_list_complex)
        
        assert isinstance(sync_only, dict)
        assert 'pearson_cc_mean' in sync_only


class TestMetricsIntegration:
    """Integration tests for analysis modules."""
    
    @pytest.mark.integration
    def test_full_analysis_pipeline(self, spike_list_complex):
        """Test complete analysis pipeline."""
        metrics = MEAMetrics()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test all grouping methods
            global_results = metrics.compute_all_metrics(spike_list_complex, grouping="global")
            channel_results = metrics.compute_all_metrics(spike_list_complex, grouping="channel")
            
        assert isinstance(global_results, dict)
        assert isinstance(channel_results, dict)
        
        # Global results should have metric categories
        for category in ['activity', 'regularity', 'synchrony', 'bursts']:
            assert category in global_results
            assert isinstance(global_results[category], dict)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_metrics_consistency(self, all_spike_lists):
        """Test that metrics are consistent across different data types."""
        metrics = MEAMetrics()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(all_spike_lists, grouping="global")
        
        # Basic consistency checks
        assert isinstance(results, dict)
        
        # Activity metrics should be consistent
        if results['activity']['active_channels'] > 0:
            assert results['activity']['total_spikes'] > 0
            assert results['activity']['mean_firing_rate'] >= 0
        
        # Synchrony metrics should be in valid ranges
        if not np.isnan(results['synchrony']['pearson_cc_mean']):
            assert -1 <= results['synchrony']['pearson_cc_mean'] <= 1
    
    @pytest.mark.integration
    def test_error_handling(self, spike_list_simple):
        """Test error handling in analysis functions."""
        metrics = MEAMetrics()
        
        # Test with invalid grouping
        with pytest.raises(ValueError):
            metrics.compute_all_metrics(spike_list_simple, grouping="invalid")
        
        # Test with None input
        with pytest.raises((ValueError, AttributeError)):
            metrics.compute_all_metrics(None, grouping="global")


# Performance tests
class TestAnalysisPerformance:
    """Performance tests for analysis modules."""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self, random_seed):
        """Test performance with large datasets."""
        # Create large dataset
        n_channels = 32
        n_spikes_total = 20000
        recording_length = 30.0
        
        spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_total))
        channels = np.random.randint(0, n_channels, n_spikes_total)
        
        from mea_flow.data import SpikeList
        spike_list = SpikeList(
            spike_data={'times': spike_times, 'channels': channels},
            recording_length=recording_length
        )
        
        metrics = MEAMetrics()
        
        # Time the computation
        import time
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(spike_list, grouping="global")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert computation_time < 30.0  # 30 seconds max
        assert isinstance(results, dict)