#!/usr/bin/env python3
"""
Manifold Learning Workflow Example
==================================

This example demonstrates advanced manifold learning techniques for analyzing
neural population dynamics in MEA data, including dimensionality reduction,
trajectory analysis, and state space exploration.

Authors: MEA-Flow Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mea_flow import SpikeList, MEAMetrics, ManifoldAnalysis, MEAPlotter
    from mea_flow.data import load_data
    from mea_flow.analysis import compute_activity_metrics
    from mea_flow.manifold import embed_population_dynamics
except ImportError as e:
    logger.error(f"Failed to import MEA-Flow modules: {e}")
    logger.error("Please ensure MEA-Flow is installed: pip install -e .")
    exit(1)

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


def create_dynamic_mea_data(n_channels: int = 4, duration: float = 10.0, 
                           state_transitions: bool = True, seed: int = 42) -> SpikeList:
    """
    Create synthetic MEA data with dynamic population activity patterns.
    
    Parameters
    ----------
    n_channels : int
        Number of electrode channels (e.g., 64 for 8x8 MEA)
    duration : float
        Recording duration in seconds
    state_transitions : bool
        Whether to include distinct network states
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    MEARecording
        Synthetic MEA recording with population dynamics
    """
    np.random.seed(seed)
    logger.info(f"Creating dynamic MEA data: {n_channels} channels, {duration}s duration (demo)")
    
    sampling_rate = 20000.0
    spike_trains = {}
    
    # Define network states if enabled
    if state_transitions:
        # Three distinct states: low, medium, high activity
        state_durations = [120, 180, 150, 150]  # seconds
        state_types = ['low', 'medium', 'high', 'burst']
        
        state_params = {
            'low': {'base_rate': 1.0, 'synchrony': 0.1, 'burst_prob': 0.01},
            'medium': {'base_rate': 5.0, 'synchrony': 0.3, 'burst_prob': 0.05},
            'high': {'base_rate': 12.0, 'synchrony': 0.6, 'burst_prob': 0.1},
            'burst': {'base_rate': 8.0, 'synchrony': 0.8, 'burst_prob': 0.15}
        }
        
        # Create state timeline
        state_timeline = []
        current_time = 0
        for state_type, state_dur in zip(state_types, state_durations):
            if current_time + state_dur <= duration:
                state_timeline.append({
                    'start': current_time,
                    'end': current_time + state_dur,
                    'type': state_type,
                    'params': state_params[state_type]
                })
                current_time += state_dur
    
    # Generate electrode layout (8x8 grid)
    grid_size = int(np.sqrt(n_channels))
    electrode_positions = {}
    for i in range(n_channels):
        row = i // grid_size
        col = i % grid_size
        electrode_positions[f"Channel_{i:02d}"] = (row, col)
    
    # Generate spike data for each channel
    for channel_id in range(n_channels):
        channel_name = f"Channel_{channel_id:02d}"
        row, col = electrode_positions[channel_name]
        
        # Channel-specific baseline properties
        position_factor = 1.0 + 0.3 * np.sin(row * np.pi / grid_size) * np.cos(col * np.pi / grid_size)
        channel_base_rate = 3.0 * position_factor
        
        spike_times = []
        t = 0.0
        
        while t < duration:
            # Determine current state parameters
            if state_transitions:
                current_state = None
                for state in state_timeline:
                    if state['start'] <= t < state['end']:
                        current_state = state
                        break
                
                if current_state:
                    state_params_current = current_state['params']
                    rate_multiplier = state_params_current['base_rate'] / 3.0
                    sync_level = state_params_current['synchrony']
                    burst_prob = state_params_current['burst_prob']
                else:
                    rate_multiplier = 1.0
                    sync_level = 0.2
                    burst_prob = 0.02
            else:
                rate_multiplier = 1.0
                sync_level = 0.2
                burst_prob = 0.02
            
            # Calculate current firing rate
            current_rate = channel_base_rate * rate_multiplier
            
            # Add temporal modulation (slow oscillations)
            oscillation = 1.0 + 0.2 * np.sin(2 * np.pi * t / 30.0)  # 30s period
            current_rate *= oscillation
            
            # Generate next spike
            if np.random.random() < burst_prob:
                # Generate baseline activity
                base_rate = np.random.uniform(2.0, 5.0)  # 2-5 Hz baseline
                burst_rate = current_rate * np.random.uniform(5, 15)
                burst_duration = np.random.uniform(0.1, 0.8)
                burst_end = min(t + burst_duration, duration)
                
                while t < burst_end:
                    t += np.random.exponential(1.0 / burst_rate)
                    if t < duration:
                        # Add synchrony-based jitter
                        if np.random.random() < sync_level:
                            # Synchronize with nearby electrodes
                            sync_jitter = np.random.normal(0, 0.005)  # 5ms jitter
                            spike_times.append(max(0, t + sync_jitter))
                        else:
                            spike_times.append(t)
            else:
                # Regular spike
                t += np.random.exponential(1.0 / current_rate)
                if t < duration:
                    spike_times.append(t)
        
        # Add cross-channel synchronous events
        if np.random.random() < 0.3:  # 30% of channels participate in sync events
            n_sync_events = int(duration * 0.1)
            sync_times = np.random.uniform(0, duration, max(1, n_sync_events))
            for sync_time in sync_times:
                if np.random.random() < 0.7:  # 70% probability to participate
                    # Add some jitter to make it realistic
                    jittered_time = sync_time + np.random.normal(0, 0.01)  # 10ms jitter
                    spike_times.append(max(0, jittered_time))
        
        # Sort spike times
        spike_times = np.array(sorted(spike_times))
        spike_trains[channel_id] = spike_times
    
    # Create SpikeList object
    spike_list = SpikeList(
        spike_data=spike_trains,
        recording_length=duration,
        sampling_rate=sampling_rate
    )
    
    # Store metadata as attributes
    spike_list.metadata = {
        'electrode_layout': f"{grid_size}x{grid_size}",
        'state_transitions': state_transitions,
        'created_by': 'manifold_learning_workflow.py'
    }
    
    if state_transitions:
        spike_list.metadata['state_timeline'] = state_timeline
    
    logger.info(f"Created MEA data with {len(spike_trains)} channels, "
                f"{sum(len(times) for times in spike_trains.values())} total spikes")
    
    return spike_list


def extract_population_features(spike_list: SpikeList, 
                               time_window: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract population-level features for manifold analysis.
    
    Parameters
    ----------
    spike_list : SpikeList
        MEA spike data
    time_window : float
        Time window for feature extraction (seconds)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix and time points
    """
    logger.info(f"Extracting population features with {time_window}s windows")
    
    # Create time bins
    duration = spike_list.recording_length
    n_bins = int(duration / time_window)
    time_points = np.linspace(0, duration, n_bins + 1)
    bin_centers = (time_points[:-1] + time_points[1:]) / 2
    
    # Initialize feature matrix
    n_channels = len(spike_list.spike_trains)
    n_features_per_channel = 5  # firing rate, ISI mean/std, burst activity, local sync
    feature_matrix = np.zeros((n_bins, n_channels * n_features_per_channel))
    
    channel_ids = list(spike_list.spike_trains.keys())
    
    for bin_idx in range(n_bins):
        start_time = time_points[bin_idx]
        end_time = time_points[bin_idx + 1]
        
        bin_features = []
        
        for channel_id in channel_ids:
            spike_train = spike_list.spike_trains[channel_id]
            
            # Get spike times for this channel
            spike_times = spike_train.spike_times
            
            # Get spikes in this time window
            mask = (spike_times >= start_time) & (spike_times < end_time)
            window_spikes = spike_times[mask]
            
            # Feature 1: Firing rate
            firing_rate = len(window_spikes) / time_window
            
            # Feature 2-3: ISI statistics
            if len(window_spikes) > 1:
                isis = np.diff(window_spikes)
                isi_mean = np.mean(isis)
                isi_std = np.std(isis)
            else:
                isi_mean = isi_std = 0
            
            # Feature 4: Burst activity (local spike density)
            burst_activity = 0
            if len(window_spikes) > 2:
                # Look for clusters of spikes (simple burst detection)
                for i in range(len(window_spikes) - 1):
                    if window_spikes[i + 1] - window_spikes[i] < 0.01:  # 10ms threshold
                        burst_activity += 1
                burst_activity = burst_activity / time_window
            
            # Feature 5: Local synchrony (simplified)
            local_sync = 0
            if len(window_spikes) > 0:
                # Count how many other channels have spikes within 5ms
                sync_count = 0
                for other_channel_id in channel_ids:
                    if other_channel_id == channel_id:
                        continue
                    other_train = spike_list.spike_trains[other_channel_id]
                    other_mask = (other_train.spike_times >= start_time) & (other_train.spike_times < end_time)
                    other_window_spikes = other_train.spike_times[other_mask]
                    
                    # Check for synchrony
                    for spike_time in window_spikes:
                        if np.any(np.abs(other_window_spikes - spike_time) < 0.005):
                            sync_count += 1
                            break
                
                local_sync = sync_count / len(channel_ids) if len(channel_ids) > 1 else 0
            
            # Combine features for this channel
            bin_features.extend([firing_rate, isi_mean, isi_std, burst_activity, local_sync])
        
        feature_matrix[bin_idx, :] = bin_features
    
    logger.info(f"Extracted features: {feature_matrix.shape}")
    return feature_matrix, bin_centers


def perform_comprehensive_manifold_analysis(spike_list: SpikeList) -> Dict:
    """
    Perform comprehensive manifold learning analysis using MEA-Flow.
    
    Parameters
    ----------
    spike_list : SpikeList
        MEA spike data
        
    Returns
    -------
    Dict
        Manifold analysis results
    """
    logger.info("Performing comprehensive manifold analysis")
    
    # Use MEA-Flow's ManifoldAnalysis class
    analyzer = ManifoldAnalysis()
    results = {}
    
    # Skip heavy MEA-Flow manifold analysis, do simple PCA directly
    logger.info("Performing lightweight PCA analysis")
    try:
        features, time_points = extract_population_features(spike_list, time_window=2.0)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        if features.shape[0] > 1 and features.shape[1] > 1:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Basic PCA
            n_components = min(2, features_scaled.shape[1], features_scaled.shape[0])
            pca = PCA(n_components=n_components)
            pca_embedding = pca.fit_transform(features_scaled)
            
            results['pca'] = {
                'embedding': pca_embedding,
                'explained_variance': pca.explained_variance_ratio_,
                'time_points': time_points
            }
        else:
            logger.warning("Insufficient data for PCA analysis")
            results = {'error': 'Insufficient data'}
            
    except Exception as e:
        logger.error(f"PCA analysis failed: {e}")
        results = {'error': str(e)}
    
    return results


def visualize_manifold_results(spike_list: SpikeList, manifold_results: Dict):
    """
    Visualize manifold learning results using MEA-Flow's plotting capabilities.
    
    Parameters
    ----------
    spike_list : SpikeList
        Original spike data
    manifold_results : Dict
        Results from manifold analysis
    """
    logger.info("Creating manifold visualizations")
    
    plotter = MEAPlotter()
    
    try:
        # Create basic raster plot
        fig_raster = plotter.plot_raster(spike_list, time_range=(0, 60))
        plotter.save_figure(fig_raster, "examples/output/manifold_raster_plot.png")
        
        # If we have PCA results, plot them
        if 'pca' in manifold_results and 'embedding' in manifold_results['pca']:
            embedding = manifold_results['pca']['embedding']
            time_points = manifold_results['pca'].get('time_points', np.arange(len(embedding)))
            
            # Create a simple scatter plot of the PCA embedding
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if embedding.shape[1] >= 2:
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                   c=time_points, cmap='viridis', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title('PCA Embedding of Neural Population Dynamics')
                plt.colorbar(scatter, label='Time (s)')
                
                # Add trajectory line
                ax.plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.3, linewidth=1)
                
                plt.tight_layout()
                plotter.save_figure(fig, "examples/output/pca_manifold.png")
                plt.close(fig)
        
        # Plot explained variance if available
        if 'pca' in manifold_results and 'explained_variance' in manifold_results['pca']:
            explained_var = manifold_results['pca']['explained_variance']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, len(explained_var) + 1), explained_var)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            plt.tight_layout()
            plotter.save_figure(fig, "examples/output/pca_explained_variance.png")
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def main():
    """
    Main function to run the manifold learning workflow.
    """
    logger.info("Starting MEA-Flow Manifold Learning Workflow")
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Create synthetic MEA data
        logger.info("Creating synthetic MEA data")
        spike_list = create_dynamic_mea_data()
        
        # 2. Perform manifold analysis
        logger.info("Performing manifold analysis")
        manifold_results = perform_comprehensive_manifold_analysis(spike_list)
        
        # 3. Create visualizations
        logger.info("Creating visualizations")
        visualize_manifold_results(spike_list, manifold_results)
        
        # 4. Print summary
        logger.info("Analysis complete!")
        if 'error' not in manifold_results:
            logger.info("Manifold analysis completed successfully")
            if 'pca' in manifold_results:
                logger.info("PCA analysis available")
                if 'explained_variance' in manifold_results['pca']:
                    total_var = np.sum(manifold_results['pca']['explained_variance'][:3])
                    logger.info(f"First 3 PCs explain {total_var:.2%} of variance")
        else:
            logger.error(f"Analysis failed: {manifold_results['error']}")
            
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()