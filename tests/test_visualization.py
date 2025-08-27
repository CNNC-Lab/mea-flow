"""
Tests for MEA-Flow visualization modules.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Use Agg backend for testing (no display required)
matplotlib.use('Agg')

from mea_flow.visualization import MEAPlotter
from mea_flow.visualization.plotter import (
    plot_raster, plot_electrode_map, plot_metrics_comparison,
    plot_manifold_embedding, plot_dimensionality_analysis
)


class TestBasicPlotting:
    """Test basic plotting functions."""
    
    @pytest.mark.visualization
    def test_plot_raster_basic(self, spike_list_simple):
        """Test basic raster plot functionality."""
        fig, ax = plot_raster(spike_list_simple)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that plot has data
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_raster_with_channels(self, spike_list_complex):
        """Test raster plot with specific channels."""
        active_channels = spike_list_complex.get_active_channels()[:5]  # First 5 active channels
        
        if len(active_channels) == 0:
            pytest.skip("No active channels in test data")
        
        fig, ax = plot_raster(spike_list_complex, channels=active_channels)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_raster_time_range(self, spike_list_complex):
        """Test raster plot with time range."""
        time_range = (1.0, 5.0)
        
        fig, ax = plot_raster(spike_list_complex, time_range=time_range)
        
        assert isinstance(fig, plt.Figure)
        
        # Check that x-axis limits match time range
        xlim = ax.get_xlim()
        assert xlim[0] >= time_range[0]
        assert xlim[1] <= time_range[1]
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_raster_well_colors(self, spike_list_multi_well):
        """Test raster plot with well-based coloring."""
        if spike_list_multi_well.well_map is None:
            pytest.skip("Well mapping not available")
        
        fig, ax = plot_raster(spike_list_multi_well, color_by_well=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_electrode_map(self, spike_list_multi_well):
        """Test electrode map visualization."""
        if spike_list_multi_well.well_map is None:
            pytest.skip("Well mapping not available")
        
        # Calculate some activity data
        activity_data = {}
        for ch in spike_list_multi_well.spike_trains.keys():
            activity_data[ch] = spike_list_multi_well.spike_trains[ch].n_spikes
        
        fig, ax = plot_electrode_map(activity_data, spike_list_multi_well.well_map)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_electrode_map_no_well_map(self, spike_list_simple):
        """Test electrode map without well mapping (should create default layout)."""
        activity_data = {}
        for ch in spike_list_simple.spike_trains.keys():
            activity_data[ch] = spike_list_simple.spike_trains[ch].n_spikes
        
        fig, ax = plot_electrode_map(activity_data)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestMetricsVisualization:
    """Test metrics visualization functions."""
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Create sample metrics data for testing."""
        np.random.seed(42)
        
        condition1_data = {
            'activity': {
                'mean_firing_rate': np.random.normal(10, 2, 20),
                'burst_rate': np.random.normal(0.5, 0.1, 20),
            },
            'synchrony': {
                'pearson_cc_mean': np.random.normal(0.3, 0.1, 20),
            }
        }
        
        condition2_data = {
            'activity': {
                'mean_firing_rate': np.random.normal(12, 2, 20),
                'burst_rate': np.random.normal(0.7, 0.1, 20),
            },
            'synchrony': {
                'pearson_cc_mean': np.random.normal(0.4, 0.1, 20),
            }
        }
        
        return {
            'Condition1': condition1_data,
            'Condition2': condition2_data
        }
    
    @pytest.mark.visualization
    def test_plot_metrics_comparison(self, sample_metrics_data):
        """Test metrics comparison plotting."""
        fig = plot_metrics_comparison(
            sample_metrics_data,
            metric_category='activity',
            metric_name='mean_firing_rate'
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have at least one axes with data
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_metrics_comparison_multiple_metrics(self, sample_metrics_data):
        """Test comparison plotting with multiple metrics."""
        metrics_to_plot = ['mean_firing_rate', 'burst_rate']
        
        fig = plot_metrics_comparison(
            sample_metrics_data,
            metric_category='activity',
            metric_name=metrics_to_plot
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_metrics_statistical_tests(self, sample_metrics_data):
        """Test metrics plotting with statistical tests."""
        fig = plot_metrics_comparison(
            sample_metrics_data,
            metric_category='activity', 
            metric_name='mean_firing_rate',
            statistical_test='t-test'
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestManifoldVisualization:
    """Test manifold visualization functions."""
    
    @pytest.fixture
    def sample_embedding_data(self, random_seed):
        """Create sample embedding data."""
        n_points = 100
        
        # Create 2D and 3D embeddings
        embedding_2d = np.random.rand(n_points, 2)
        embedding_3d = np.random.rand(n_points, 3)
        
        # Create time labels
        time_labels = np.linspace(0, 10, n_points)
        
        return {
            '2D': {'embedding': embedding_2d, 'time_labels': time_labels},
            '3D': {'embedding': embedding_3d, 'time_labels': time_labels}
        }
    
    @pytest.mark.visualization
    def test_plot_manifold_embedding_2d(self, sample_embedding_data):
        """Test 2D manifold embedding visualization."""
        embedding_data = sample_embedding_data['2D']
        
        fig, ax = plot_manifold_embedding(
            embedding_data['embedding'],
            title="Test 2D Embedding",
            labels=embedding_data['time_labels']
        )
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that scatter plot was created
        assert len(ax.collections) > 0
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_manifold_embedding_3d(self, sample_embedding_data):
        """Test 3D manifold embedding visualization."""
        embedding_data = sample_embedding_data['3D']
        
        fig, ax = plot_manifold_embedding(
            embedding_data['embedding'],
            title="Test 3D Embedding",
            labels=embedding_data['time_labels']
        )
        
        assert isinstance(fig, plt.Figure)
        # 3D plots use Axes3D
        assert hasattr(ax, 'zaxis')
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_manifold_trajectory(self, sample_embedding_data):
        """Test manifold trajectory visualization."""
        embedding_data = sample_embedding_data['2D']
        
        fig, ax = plot_manifold_embedding(
            embedding_data['embedding'],
            title="Test Trajectory",
            labels=embedding_data['time_labels'],
            show_trajectory=True
        )
        
        assert isinstance(fig, plt.Figure)
        
        # Should have both scatter points and trajectory lines
        assert len(ax.collections) > 0 or len(ax.lines) > 0
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_dimensionality_analysis(self, random_seed):
        """Test dimensionality analysis visualization."""
        # Create sample PCA results
        n_components = 10
        explained_variance_ratio = np.exp(-np.arange(n_components) * 0.5)
        explained_variance_ratio /= np.sum(explained_variance_ratio)
        
        pca_results = {
            'explained_variance_ratio': explained_variance_ratio,
            'explained_variance': explained_variance_ratio * 100
        }
        
        fig = plot_dimensionality_analysis(pca_results)
        
        assert isinstance(fig, plt.Figure)
        
        # Should have subplots
        axes = fig.get_axes()
        assert len(axes) >= 2  # At least 2 subplots expected
        
        plt.close(fig)


class TestMEAPlotter:
    """Test the main MEAPlotter class."""
    
    @pytest.mark.visualization
    def test_mea_plotter_initialization(self):
        """Test MEAPlotter initialization."""
        plotter = MEAPlotter()
        assert plotter is not None
        
        # Test with custom style
        plotter_custom = MEAPlotter(style='seaborn-v0_8', figsize=(12, 8))
        assert plotter_custom.figsize == (12, 8)
    
    @pytest.mark.visualization
    def test_plot_spike_raster(self, spike_list_simple):
        """Test raster plotting through MEAPlotter."""
        plotter = MEAPlotter()
        
        fig = plotter.plot_spike_raster(spike_list_simple)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_activity_summary(self, spike_list_complex):
        """Test activity summary plotting."""
        from mea_flow.analysis import MEAMetrics
        
        metrics = MEAMetrics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_activity_metrics(spike_list_complex)
        
        plotter = MEAPlotter()
        fig = plotter.plot_activity_summary(spike_list_complex, results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_synchrony_analysis(self, spike_list_complex):
        """Test synchrony analysis plotting."""
        from mea_flow.analysis import MEAMetrics
        
        metrics = MEAMetrics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_synchrony_metrics(spike_list_complex)
        
        plotter = MEAPlotter()
        fig = plotter.plot_synchrony_analysis(spike_list_complex, results)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_plot_manifold_results(self, spike_list_complex):
        """Test manifold results plotting."""
        from mea_flow.manifold import ManifoldAnalysis
        
        analysis = ManifoldAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(spike_list_complex)
        
        if results.get('embeddings'):
            plotter = MEAPlotter()
            fig = plotter.plot_manifold_results(results)
            
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        else:
            pytest.skip("No valid embeddings generated for test data")
    
    @pytest.mark.visualization
    def test_create_summary_report(self, spike_list_complex):
        """Test comprehensive summary report creation."""
        from mea_flow.analysis import MEAMetrics
        
        # Compute metrics
        metrics = MEAMetrics()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = metrics.compute_all_metrics(spike_list_complex, grouping='global')
        
        plotter = MEAPlotter()
        
        # Create summary for single condition
        fig = plotter.create_summary_report([spike_list_complex], results)
        
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_save_figure(self, spike_list_simple, temp_dir):
        """Test figure saving functionality."""
        plotter = MEAPlotter()
        fig = plotter.plot_spike_raster(spike_list_simple)
        
        # Test saving with different formats
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            output_file = temp_dir / f"test_plot.{fmt}"
            plotter.save_figure(fig, output_file, dpi=100)
            
            assert output_file.exists()
            assert output_file.stat().st_size > 0
        
        plt.close(fig)


class TestVisualizationIntegration:
    """Integration tests for visualization modules."""
    
    @pytest.mark.integration
    @pytest.mark.visualization
    def test_full_visualization_pipeline(self, spike_list_complex):
        """Test complete visualization pipeline."""
        from mea_flow.analysis import MEAMetrics
        from mea_flow.manifold import ManifoldAnalysis
        
        # Compute all analyses
        metrics = MEAMetrics()
        manifold = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get all metrics
            results = metrics.compute_all_metrics(spike_list_complex, grouping='global')
            
            # Get manifold analysis
            manifold_results = manifold.analyze_population_dynamics(spike_list_complex)
        
        # Create all visualizations
        plotter = MEAPlotter()
        
        figures = []
        
        # Basic raster plot
        fig1 = plotter.plot_spike_raster(spike_list_complex)
        figures.append(fig1)
        
        # Activity analysis
        if results.get('activity'):
            fig2 = plotter.plot_activity_summary(spike_list_complex, results['activity'])
            figures.append(fig2)
        
        # Synchrony analysis
        if results.get('synchrony'):
            fig3 = plotter.plot_synchrony_analysis(spike_list_complex, results['synchrony'])
            figures.append(fig3)
        
        # Manifold analysis
        if manifold_results.get('embeddings'):
            fig4 = plotter.plot_manifold_results(manifold_results)
            figures.append(fig4)
        
        # Summary report
        fig5 = plotter.create_summary_report([spike_list_complex], results)
        figures.append(fig5)
        
        # All figures should be valid
        for fig in figures:
            assert isinstance(fig, plt.Figure)
            assert len(fig.get_axes()) > 0
        
        # Close all figures
        for fig in figures:
            plt.close(fig)
    
    @pytest.mark.integration
    @pytest.mark.visualization
    def test_cross_condition_visualization(self, spike_list_simple, spike_list_complex):
        """Test visualization across multiple conditions."""
        from mea_flow.analysis import MEAMetrics
        
        spike_lists = [spike_list_simple, spike_list_complex]
        condition_names = ['Simple', 'Complex']
        
        # Compute metrics for both conditions
        metrics = MEAMetrics()
        all_results = {}
        
        for i, spike_list in enumerate(spike_lists):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = metrics.compute_all_metrics(spike_list, grouping='global')
                all_results[condition_names[i]] = results
        
        plotter = MEAPlotter()
        
        # Test cross-condition comparison
        if len(all_results) > 1:
            fig = plotter.plot_condition_comparison(all_results)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    @pytest.mark.integration
    @pytest.mark.visualization
    @pytest.mark.slow
    def test_large_dataset_visualization(self, random_seed):
        """Test visualization performance with larger datasets."""
        from mea_flow.data import SpikeList
        
        # Create larger dataset
        n_channels = 32
        n_spikes_total = 5000
        recording_length = 20.0
        
        spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_total))
        channels = np.random.randint(0, n_channels, n_spikes_total)
        
        spike_list = SpikeList(
            spike_data={'times': spike_times, 'channels': channels},
            recording_length=recording_length
        )
        
        plotter = MEAPlotter()
        
        # Test that raster plot handles large data efficiently
        import time
        start_time = time.time()
        
        fig = plotter.plot_spike_raster(spike_list)
        
        end_time = time.time()
        plotting_time = end_time - start_time
        
        # Should complete within reasonable time
        assert plotting_time < 10.0  # 10 seconds max
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


class TestVisualizationErrorHandling:
    """Test error handling in visualization modules."""
    
    @pytest.mark.visualization
    def test_empty_data_visualization(self, empty_spike_list):
        """Test visualization with empty data."""
        plotter = MEAPlotter()
        
        # Should handle empty data gracefully
        fig = plotter.plot_spike_raster(empty_spike_list)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_invalid_parameters(self, spike_list_simple):
        """Test handling of invalid visualization parameters."""
        plotter = MEAPlotter()
        
        # Test with invalid time range
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plotter.plot_spike_raster(
                spike_list_simple, 
                time_range=(-1, 1000)  # Invalid range
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        
        # Test with invalid channels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = plotter.plot_spike_raster(
                spike_list_simple,
                channels=[999, 1000]  # Non-existent channels
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    @pytest.mark.visualization
    def test_missing_data_handling(self):
        """Test handling of missing or None data."""
        plotter = MEAPlotter()
        
        # Test with None input
        with pytest.raises((ValueError, AttributeError)):
            plotter.plot_spike_raster(None)


# Custom fixtures for visualization tests
@pytest.fixture
def mock_matplotlib_backend():
    """Mock matplotlib backend for testing."""
    with patch('matplotlib.pyplot.show') as mock_show:
        yield mock_show


class TestPlotCustomization:
    """Test plot customization options."""
    
    @pytest.mark.visualization
    def test_plot_styling_options(self, spike_list_simple):
        """Test various plot styling options."""
        plotter = MEAPlotter(
            style='seaborn-v0_8',
            color_palette='viridis',
            figsize=(10, 6)
        )
        
        fig = plotter.plot_spike_raster(
            spike_list_simple,
            title="Custom Title",
            xlabel="Custom X Label",
            ylabel="Custom Y Label"
        )
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 10  # Width
        assert fig.get_size_inches()[1] == 6   # Height
        
        plt.close(fig)
    
    @pytest.mark.visualization
    def test_color_customization(self, spike_list_multi_well):
        """Test color customization options."""
        if spike_list_multi_well.well_map is None:
            pytest.skip("Well mapping not available")
        
        plotter = MEAPlotter(color_palette='Set1')
        
        fig = plotter.plot_spike_raster(
            spike_list_multi_well,
            color_by_well=True,
            colors=['red', 'blue', 'green', 'orange']
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)