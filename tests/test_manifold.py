"""
Tests for MEA-Flow manifold learning modules.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import Mock, patch

from mea_flow.manifold import ManifoldAnalysis
from mea_flow.manifold.embedding import embed_population_dynamics, apply_dimensionality_reduction
from mea_flow.manifold.evaluation import evaluate_embedding, effective_dimensionality
from mea_flow.manifold.comparison import compare_manifolds


class TestEmbedding:
    """Test manifold embedding functions."""
    
    @pytest.fixture
    def sample_data(self, random_seed):
        """Create sample high-dimensional data."""
        n_timepoints = 200
        n_channels = 16
        
        # Create data with some structure
        t = np.linspace(0, 10, n_timepoints)
        signal_matrix = np.zeros((n_channels, n_timepoints))
        
        for i in range(n_channels):
            # Add some structured patterns
            signal_matrix[i, :] = np.sin(2 * np.pi * (i + 1) * t / 10) + 0.1 * np.random.randn(n_timepoints)
        
        return signal_matrix
    
    @pytest.mark.unit
    def test_pca_embedding(self, sample_data):
        """Test PCA embedding."""
        result = embed_population_dynamics(
            sample_data, 
            method='PCA', 
            n_components=3
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert 'explained_variance_ratio' in result
        assert 'method' in result
        
        assert result['embedding'].shape == (sample_data.shape[1], 3)
        assert result['method'] == 'PCA'
        assert len(result['explained_variance_ratio']) == 3
        assert np.sum(result['explained_variance_ratio']) <= 1.0
    
    @pytest.mark.unit
    def test_mds_embedding(self, sample_data):
        """Test MDS embedding."""
        result = embed_population_dynamics(
            sample_data,
            method='MDS',
            n_components=2
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert 'stress' in result
        assert result['embedding'].shape == (sample_data.shape[1], 2)
        assert result['stress'] >= 0
    
    @pytest.mark.unit
    def test_isomap_embedding(self, sample_data):
        """Test Isomap embedding.""" 
        result = embed_population_dynamics(
            sample_data,
            method='Isomap',
            n_components=2,
            n_neighbors=10
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert 'reconstruction_error' in result
        assert result['embedding'].shape == (sample_data.shape[1], 2)
    
    @pytest.mark.unit
    def test_lle_embedding(self, sample_data):
        """Test Locally Linear Embedding."""
        result = embed_population_dynamics(
            sample_data,
            method='LLE',
            n_components=2,
            n_neighbors=10
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert 'reconstruction_error' in result
        assert result['embedding'].shape == (sample_data.shape[1], 2)
    
    @pytest.mark.unit
    def test_tsne_embedding(self, sample_data):
        """Test t-SNE embedding."""
        # Use smaller dataset for t-SNE (it can be slow)
        small_data = sample_data[:, ::10]  # Subsample
        
        result = embed_population_dynamics(
            small_data,
            method='t-SNE',
            n_components=2,
            perplexity=5  # Small perplexity for small dataset
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert 'kl_divergence' in result
        assert result['embedding'].shape == (small_data.shape[1], 2)
    
    @pytest.mark.optional
    @patch('mea_flow.manifold.embedding.UMAP_AVAILABLE', True)
    @patch('mea_flow.manifold.embedding.umap')
    def test_umap_embedding(self, mock_umap, sample_data):
        """Test UMAP embedding when available."""
        # Mock UMAP
        mock_reducer = Mock()
        mock_reducer.fit_transform.return_value = np.random.rand(sample_data.shape[1], 2)
        mock_umap.UMAP.return_value = mock_reducer
        
        result = embed_population_dynamics(
            sample_data,
            method='UMAP',
            n_components=2
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert result['embedding'].shape == (sample_data.shape[1], 2)
    
    @pytest.mark.unit
    def test_spectral_embedding(self, sample_data):
        """Test Spectral embedding."""
        result = embed_population_dynamics(
            sample_data,
            method='Spectral',
            n_components=2,
            n_neighbors=10
        )
        
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert result['embedding'].shape == (sample_data.shape[1], 2)
    
    @pytest.mark.unit
    def test_invalid_method(self, sample_data):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError):
            embed_population_dynamics(
                sample_data,
                method='InvalidMethod',
                n_components=2
            )
    
    @pytest.mark.unit
    def test_apply_multiple_methods(self, sample_data):
        """Test applying multiple dimensionality reduction methods."""
        methods = ['PCA', 'MDS']
        results = apply_dimensionality_reduction(
            sample_data,
            methods=methods,
            n_components=2
        )
        
        assert isinstance(results, dict)
        assert len(results) == len(methods)
        
        for method in methods:
            assert method in results
            assert 'embedding' in results[method]
    
    @pytest.mark.unit
    def test_embedding_with_small_dataset(self):
        """Test embedding with very small dataset."""
        # Create minimal dataset
        small_data = np.random.rand(3, 10)  # Only 3 features, 10 timepoints
        
        result = embed_population_dynamics(
            small_data,
            method='PCA',
            n_components=2
        )
        
        assert isinstance(result, dict)
        # Should automatically limit n_components to available dimensions
        assert result['embedding'].shape[1] <= min(small_data.shape)


class TestEmbeddingEvaluation:
    """Test embedding evaluation metrics."""
    
    @pytest.fixture
    def embedding_data(self, random_seed):
        """Create sample embedding and original data."""
        n_points = 100
        original_dim = 10
        embedding_dim = 2
        
        # Create structured high-dimensional data
        original = np.random.rand(n_points, original_dim)
        # Create simple 2D embedding
        embedding = original[:, :embedding_dim] + 0.1 * np.random.rand(n_points, embedding_dim)
        
        return original, embedding
    
    @pytest.mark.unit
    def test_evaluate_embedding(self, embedding_data):
        """Test embedding evaluation metrics."""
        original, embedding = embedding_data
        
        metrics = evaluate_embedding(original, embedding, k_neighbors=5)
        
        assert isinstance(metrics, dict)
        expected_keys = ['trustworthiness', 'continuity', 'shepard_correlation']
        
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1  # These metrics are typically in [0, 1]
    
    @pytest.mark.unit
    def test_effective_dimensionality(self, random_seed):
        """Test effective dimensionality estimation."""
        # Create data with known effective dimensionality
        n_samples = 200
        true_dim = 3
        noise_dim = 7
        
        # Generate data that lives in 3D subspace
        true_data = np.random.rand(n_samples, true_dim)
        noise_data = 0.1 * np.random.rand(n_samples, noise_dim)
        
        # Random projection to higher dimensions
        projection_matrix = np.random.rand(true_dim + noise_dim, 15)
        high_dim_data = np.column_stack([true_data, noise_data]) @ projection_matrix
        
        # Estimate dimensionality
        est_dim = effective_dimensionality(high_dim_data, method='pca_variance')
        
        assert isinstance(est_dim, (int, float))
        assert est_dim > 0
        # Should be reasonably close to true dimensionality (allowing for some error)
        assert est_dim <= high_dim_data.shape[1]
    
    @pytest.mark.unit
    def test_participation_ratio(self, random_seed):
        """Test participation ratio method."""
        # Create data with clear dimensionality structure
        n_samples = 100
        data = np.random.rand(n_samples, 10)
        
        pr = effective_dimensionality(data, method='participation_ratio')
        
        assert isinstance(pr, (int, float))
        assert pr > 0
        assert pr <= data.shape[1]


class TestManifoldComparison:
    """Test manifold comparison functions."""
    
    @pytest.fixture
    def comparison_data(self, random_seed):
        """Create data for manifold comparison."""
        n_points = 50
        dim = 3
        
        # Create two similar but different embeddings
        base_embedding = np.random.rand(n_points, dim)
        
        # Second embedding with some transformation
        embedding1 = base_embedding + 0.1 * np.random.rand(n_points, dim)
        embedding2 = base_embedding + 0.2 * np.random.rand(n_points, dim)
        
        return embedding1, embedding2
    
    @pytest.mark.unit
    def test_compare_manifolds(self, comparison_data):
        """Test manifold comparison functionality."""
        embedding1, embedding2 = comparison_data
        
        comparison = compare_manifolds(embedding1, embedding2)
        
        assert isinstance(comparison, dict)
        expected_keys = ['procrustes_distance', 'alignment_similarity']
        
        for key in expected_keys:
            assert key in comparison
            assert isinstance(comparison[key], (int, float))
    
    @pytest.mark.unit
    def test_procrustes_analysis(self, comparison_data):
        """Test Procrustes analysis component."""
        from mea_flow.manifold.comparison import procrustes_analysis
        
        embedding1, embedding2 = comparison_data
        
        result = procrustes_analysis(embedding1, embedding2)
        
        assert isinstance(result, dict)
        assert 'distance' in result
        assert 'transformed' in result
        assert result['transformed'].shape == embedding2.shape


class TestManifoldAnalysis:
    """Test the main ManifoldAnalysis class."""
    
    @pytest.mark.unit
    def test_manifold_analysis_initialization(self):
        """Test ManifoldAnalysis initialization."""
        analysis = ManifoldAnalysis()
        assert analysis is not None
        
        # Test with custom config
        from mea_flow.manifold.analysis import ManifoldConfig
        config = ManifoldConfig(max_components=5, methods=['PCA', 'MDS'])
        analysis_custom = ManifoldAnalysis(config=config)
        assert analysis_custom.config.max_components == 5
    
    @pytest.mark.unit
    def test_analyze_population_dynamics(self, spike_list_complex):
        """Test complete population dynamics analysis."""
        analysis = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(spike_list_complex)
        
        assert isinstance(results, dict)
        
        expected_keys = [
            'embeddings', 'evaluation', 'dimensionality', 'config_used'
        ]
        
        for key in expected_keys:
            assert key in results
    
    @pytest.mark.unit
    def test_analyze_population_dynamics_channel_selection(self, spike_list_complex):
        """Test analysis with specific channel selection."""
        analysis = ManifoldAnalysis()
        
        active_channels = spike_list_complex.get_active_channels(min_spikes=10)
        if len(active_channels) < 4:
            pytest.skip("Need at least 4 active channels")
        
        selected_channels = active_channels[:4]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(
                spike_list_complex, 
                channels=selected_channels
            )
        
        assert isinstance(results, dict)
        assert 'embeddings' in results
    
    @pytest.mark.unit
    def test_analyze_population_dynamics_time_slice(self, spike_list_complex):
        """Test analysis with time range selection."""
        analysis = ManifoldAnalysis()
        
        time_range = (1.0, 5.0)  # Analyze middle portion
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(
                spike_list_complex,
                time_range=time_range
            )
        
        assert isinstance(results, dict)
        # Should still return valid results for the time slice
    
    @pytest.mark.unit
    def test_insufficient_data_handling(self, empty_spike_list):
        """Test handling of insufficient data."""
        analysis = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(empty_spike_list)
        
        # Should handle gracefully and return appropriate error/empty results
        assert isinstance(results, dict)
    
    @pytest.mark.unit
    def test_custom_methods_selection(self, spike_list_complex):
        """Test analysis with custom method selection."""
        from mea_flow.manifold.analysis import ManifoldConfig
        
        config = ManifoldConfig(
            methods=['PCA', 'MDS'],  # Only use these methods
            max_components=2
        )
        analysis = ManifoldAnalysis(config=config)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(spike_list_complex)
        
        if 'embeddings' in results and results['embeddings']:
            # Should only have the requested methods
            available_methods = set(results['embeddings'].keys())
            requested_methods = set(config.methods)
            assert available_methods.issubset(requested_methods)


class TestManifoldIntegration:
    """Integration tests for manifold learning."""
    
    @pytest.mark.integration
    def test_full_manifold_pipeline(self, spike_list_complex):
        """Test complete manifold analysis pipeline."""
        analysis = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Run full analysis
            results = analysis.analyze_population_dynamics(spike_list_complex)
            
            # Test that results are comprehensive
            assert isinstance(results, dict)
            
            if 'embeddings' in results and results['embeddings']:
                # Test that embeddings are valid
                for method, embedding_result in results['embeddings'].items():
                    assert 'embedding' in embedding_result
                    assert isinstance(embedding_result['embedding'], np.ndarray)
                    assert embedding_result['embedding'].ndim == 2
                    assert embedding_result['embedding'].shape[1] <= embedding_result.get('n_components', 3)
            
            # Test evaluation metrics
            if 'evaluation' in results:
                assert isinstance(results['evaluation'], dict)
            
            # Test dimensionality estimation
            if 'dimensionality' in results:
                assert isinstance(results['dimensionality'], dict)
    
    @pytest.mark.integration
    def test_multiple_conditions_comparison(self, spike_list_simple, spike_list_complex):
        """Test comparing manifolds across conditions."""
        analysis = ManifoldAnalysis()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Analyze both conditions
            results1 = analysis.analyze_population_dynamics(spike_list_simple)
            results2 = analysis.analyze_population_dynamics(spike_list_complex)
            
            # If both have valid embeddings, test comparison
            if (results1.get('embeddings') and results2.get('embeddings') and
                'PCA' in results1['embeddings'] and 'PCA' in results2['embeddings']):
                
                emb1 = results1['embeddings']['PCA']['embedding']
                emb2 = results2['embeddings']['PCA']['embedding']
                
                # Make sure they have the same number of components for comparison
                min_components = min(emb1.shape[1], emb2.shape[1])
                emb1_comp = emb1[:, :min_components]
                emb2_comp = emb2[:, :min_components]
                
                # Ensure same number of points (pad with zeros if needed)
                min_points = min(emb1_comp.shape[0], emb2_comp.shape[0])
                emb1_comp = emb1_comp[:min_points]
                emb2_comp = emb2_comp[:min_points]
                
                if min_points > 0 and min_components > 0:
                    comparison = compare_manifolds(emb1_comp, emb2_comp)
                    assert isinstance(comparison, dict)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_manifold_analysis_robustness(self, random_seed):
        """Test robustness of manifold analysis with various data conditions."""
        from mea_flow.data import SpikeList
        
        analysis = ManifoldAnalysis()
        
        # Test with different data characteristics
        test_conditions = [
            # (n_channels, n_spikes_total, recording_length, description)
            (8, 500, 5.0, "small_dataset"),
            (32, 5000, 20.0, "medium_dataset"),
            (16, 100, 2.0, "sparse_data"),
        ]
        
        for n_channels, n_spikes_total, recording_length, description in test_conditions:
            # Generate test data
            spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_total))
            channels = np.random.randint(0, n_channels, n_spikes_total)
            
            spike_list = SpikeList(
                spike_data={'times': spike_times, 'channels': channels},
                recording_length=recording_length
            )
            
            # Should handle all conditions gracefully
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = analysis.analyze_population_dynamics(spike_list)
            
            assert isinstance(results, dict), f"Failed for {description}"


# Performance tests
class TestManifoldPerformance:
    """Performance tests for manifold learning."""
    
    @pytest.mark.slow
    def test_manifold_performance(self, random_seed):
        """Test performance with moderately large datasets."""
        from mea_flow.data import SpikeList
        
        # Create moderately large dataset
        n_channels = 32
        n_spikes_total = 10000
        recording_length = 30.0
        
        spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes_total))
        channels = np.random.randint(0, n_channels, n_spikes_total)
        
        spike_list = SpikeList(
            spike_data={'times': spike_times, 'channels': channels},
            recording_length=recording_length
        )
        
        analysis = ManifoldAnalysis()
        
        # Time the computation
        import time
        start_time = time.time()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analysis.analyze_population_dynamics(spike_list)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert computation_time < 60.0  # 1 minute max
        assert isinstance(results, dict)