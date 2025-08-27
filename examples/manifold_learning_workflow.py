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
    from mea_flow.data import SpikeData, MEARecording
    from mea_flow.analysis import ActivityAnalyzer, BurstDetector
    from mea_flow.manifold import ManifoldAnalyzer
    from mea_flow.visualization import ActivityPlotter, ManifoldPlotter
except ImportError as e:
    logger.error(f"Failed to import MEA-Flow modules: {e}")
    logger.error("Please ensure MEA-Flow is installed: pip install -e .")
    exit(1)

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


def create_dynamic_mea_data(n_channels: int = 64, duration: float = 600.0, 
                           state_transitions: bool = True, seed: int = 42) -> MEARecording:
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
    logger.info(f"Creating dynamic MEA data: {n_channels} channels, {duration}s duration")
    
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
                # Generate burst
                burst_duration = np.random.uniform(0.1, 0.8)
                burst_rate = current_rate * np.random.uniform(5, 15)
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
            n_sync_events = int(duration / 60)  # ~1 per minute
            sync_times = np.random.uniform(0, duration, n_sync_events)
            for sync_time in sync_times:
                if np.random.random() < 0.7:  # 70% probability to participate
                    # Add some jitter to make it realistic
                    jittered_time = sync_time + np.random.normal(0, 0.01)  # 10ms jitter
                    spike_times.append(max(0, jittered_time))
        
        # Sort spike times and convert to samples
        spike_times = np.array(sorted(spike_times))
        spike_samples = (spike_times * sampling_rate).astype(int)
        
        # Create SpikeData object
        spike_trains[channel_name] = SpikeData(
            spike_times=spike_times,
            spike_samples=spike_samples,
            channel_id=channel_name,
            sampling_rate=sampling_rate
        )
    
    # Create MEA recording with metadata
    metadata = {
        'electrode_layout': f"{grid_size}x{grid_size}",
        'electrode_positions': electrode_positions,
        'state_transitions': state_transitions,
        'created_by': 'manifold_learning_workflow.py'
    }
    
    if state_transitions:
        metadata['state_timeline'] = state_timeline
    
    recording = MEARecording(
        spike_trains=spike_trains,
        duration=duration,
        sampling_rate=sampling_rate,
        metadata=metadata
    )
    
    logger.info(f"Created recording with {len(spike_trains)} channels")
    return recording


def extract_population_features(recording: MEARecording, 
                               time_window: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract population-level features for manifold analysis.
    
    Parameters
    ----------
    recording : MEARecording
        MEA recording data
    time_window : float
        Time window for feature extraction (seconds)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix and time points
    """
    logger.info(f"Extracting population features with {time_window}s windows")
    
    # Create time bins
    n_bins = int(recording.duration / time_window)
    time_points = np.linspace(0, recording.duration, n_bins + 1)
    bin_centers = (time_points[:-1] + time_points[1:]) / 2
    
    # Initialize feature matrix
    n_channels = len(recording.spike_trains)
    n_features_per_channel = 5  # firing rate, ISI mean/std, burst activity, local sync
    feature_matrix = np.zeros((n_bins, n_channels * n_features_per_channel))
    
    channel_names = list(recording.spike_trains.keys())
    
    for bin_idx in range(n_bins):
        start_time = time_points[bin_idx]
        end_time = time_points[bin_idx + 1]
        
        bin_features = []
        
        for channel_name in channel_names:
            spike_data = recording.spike_trains[channel_name]
            
            # Get spikes in this time window
            mask = (spike_data.spike_times >= start_time) & (spike_data.spike_times < end_time)
            window_spikes = spike_data.spike_times[mask]
            
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
                for other_channel in channel_names:
                    if other_channel == channel_name:
                        continue
                    other_spikes = recording.spike_trains[other_channel].spike_times
                    other_mask = (other_spikes >= start_time) & (other_spikes < end_time)
                    other_window_spikes = other_spikes[other_mask]
                    
                    # Check for synchrony
                    for spike_time in window_spikes:
                        if np.any(np.abs(other_window_spikes - spike_time) < 0.005):
                            sync_count += 1
                            break
                
                local_sync = sync_count / len(channel_names) if len(channel_names) > 1 else 0
            
            # Combine features for this channel
            bin_features.extend([firing_rate, isi_mean, isi_std, burst_activity, local_sync])
        
        feature_matrix[bin_idx, :] = bin_features
    
    logger.info(f"Extracted features: {feature_matrix.shape}")
    return feature_matrix, bin_centers


def perform_comprehensive_manifold_analysis(features: np.ndarray, 
                                          time_points: np.ndarray) -> Dict:
    """
    Perform comprehensive manifold learning analysis.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_timepoints x n_features)
    time_points : np.ndarray
        Time points corresponding to features
        
    Returns
    -------
    Dict
        Manifold analysis results
    """
    logger.info("Performing comprehensive manifold analysis")
    
    analyzer = ManifoldAnalyzer()
    results = {}
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 1. Principal Component Analysis
    logger.info("Applying PCA")
    try:
        pca_2d = analyzer.apply_pca(features_scaled, n_components=2)
        pca_3d = analyzer.apply_pca(features_scaled, n_components=3)
        
        # Calculate explained variance
        from sklearn.decomposition import PCA
        pca_full = PCA()
        pca_full.fit(features_scaled)
        explained_variance = pca_full.explained_variance_ratio_
        
        results['pca'] = {
            'embedding_2d': pca_2d,
            'embedding_3d': pca_3d,
            'explained_variance': explained_variance,
            'cumulative_variance': np.cumsum(explained_variance)
        }
    except Exception as e:
        logger.error(f"PCA failed: {e}")
    
    # 2. t-SNE Analysis
    logger.info("Applying t-SNE")
    try:
        # Different perplexity values
        perplexities = [5, 10, 30]
        tsne_results = {}
        
        for perp in perplexities:
            if len(features_scaled) > perp:
                tsne_embedding = analyzer.apply_tsne(features_scaled, n_components=2, 
                                                   perplexity=perp)
                tsne_results[f'perplexity_{perp}'] = tsne_embedding
        
        results['tsne'] = tsne_results
    except Exception as e:
        logger.error(f"t-SNE failed: {e}")
    
    # 3. UMAP Analysis (if available)
    logger.info("Applying UMAP")
    try:
        umap_2d = analyzer.apply_umap(features_scaled, n_components=2)
        umap_3d = analyzer.apply_umap(features_scaled, n_components=3)
        
        results['umap'] = {
            'embedding_2d': umap_2d,
            'embedding_3d': umap_3d
        }
    except ImportError:
        logger.warning("UMAP not available")
    except Exception as e:
        logger.error(f"UMAP failed: {e}")
    
    # 4. Multidimensional Scaling
    logger.info("Applying MDS")
    try:
        mds_2d = analyzer.apply_mds(features_scaled, n_components=2)
        mds_3d = analyzer.apply_mds(features_scaled, n_components=3)
        
        results['mds'] = {
            'embedding_2d': mds_2d,
            'embedding_3d': mds_3d
        }
    except Exception as e:
        logger.error(f"MDS failed: {e}")
    
    # 5. Isomap (if enough samples)
    logger.info("Applying Isomap")
    try:
        if len(features_scaled) > 10:
            isomap_2d = analyzer.apply_isomap(features_scaled, n_components=2)
            isomap_3d = analyzer.apply_isomap(features_scaled, n_components=3)
            
            results['isomap'] = {
                'embedding_2d': isomap_2d,
                'embedding_3d': isomap_3d
            }
    except Exception as e:
        logger.error(f"Isomap failed: {e}")
    
    # Add metadata
    results['metadata'] = {
        'n_timepoints': len(time_points),
        'n_features': features.shape[1],
        'time_points': time_points,
        'feature_names': ['firing_rate', 'isi_mean', 'isi_std', 'burst_activity', 'local_sync']
    }
    
    return results


def analyze_trajectory_dynamics(manifold_results: Dict) -> Dict:
    """
    Analyze trajectory dynamics in the manifold space.
    
    Parameters
    ----------
    manifold_results : Dict
        Results from manifold learning analysis
        
    Returns
    -------
    Dict
        Trajectory analysis results
    """
    logger.info("Analyzing trajectory dynamics")
    
    time_points = manifold_results['metadata']['time_points']
    trajectory_results = {}
    
    for method_name, method_results in manifold_results.items():
        if method_name == 'metadata':
            continue
        
        if isinstance(method_results, dict) and 'embedding_2d' in method_results:
            embedding = method_results['embedding_2d']
        elif method_name == 'tsne' and isinstance(method_results, dict):
            # Use the first t-SNE result
            embedding = list(method_results.values())[0]
        else:
            continue
        
        if embedding is None or len(embedding) < 2:
            continue
        
        # Calculate trajectory properties
        velocities = np.diff(embedding, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Calculate trajectory length
        trajectory_length = np.sum(speeds)
        
        # Calculate curvature (change in direction)
        if len(velocities) > 1:
            velocity_angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_changes = np.diff(velocity_angles)
            # Handle angle wrapping
            angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
            curvatures = np.abs(angle_changes)
            mean_curvature = np.mean(curvatures)
        else:
            curvatures = np.array([])
            mean_curvature = 0
        
        # Find stationary periods (low speed)
        speed_threshold = np.percentile(speeds, 25)  # Bottom 25%
        stationary_periods = speeds < speed_threshold
        
        trajectory_results[method_name] = {
            'speeds': speeds,
            'mean_speed': np.mean(speeds),
            'trajectory_length': trajectory_length,
            'curvatures': curvatures,
            'mean_curvature': mean_curvature,
            'stationary_periods': stationary_periods,
            'stationary_fraction': np.mean(stationary_periods)
        }
    
    return trajectory_results


def create_manifold_visualizations(manifold_results: Dict, trajectory_results: Dict,
                                 recording_metadata: Dict, output_dir: Path) -> None:
    """
    Create comprehensive manifold learning visualizations.
    
    Parameters
    ----------
    manifold_results : Dict
        Manifold analysis results
    trajectory_results : Dict
        Trajectory analysis results
    recording_metadata : Dict
        MEA recording metadata
    output_dir : Path
        Output directory for plots
    """
    logger.info("Creating manifold visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    time_points = manifold_results['metadata']['time_points']
    
    # 1. Overview plot of all manifold methods
    methods_with_2d = []
    embeddings_2d = []
    
    for method_name, method_results in manifold_results.items():
        if method_name == 'metadata':
            continue
        
        if isinstance(method_results, dict) and 'embedding_2d' in method_results:
            methods_with_2d.append(method_name.upper())
            embeddings_2d.append(method_results['embedding_2d'])
        elif method_name == 'tsne' and isinstance(method_results, dict):
            # Use perplexity_30 if available, otherwise first available
            if 'perplexity_30' in method_results:
                methods_with_2d.append('t-SNE (p=30)')
                embeddings_2d.append(method_results['perplexity_30'])
            else:
                key = list(method_results.keys())[0]
                methods_with_2d.append(f't-SNE ({key})')
                embeddings_2d.append(method_results[key])
    
    if embeddings_2d:
        n_methods = len(embeddings_2d)
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(5 * ((n_methods + 1) // 2), 10))
        if n_methods == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create colormap based on time
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
        
        for i, (method_name, embedding) in enumerate(zip(methods_with_2d, embeddings_2d)):
            ax = axes[i]
            
            # Plot trajectory with time coloring
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=time_points, 
                               cmap='viridis', s=30, alpha=0.7)
            
            # Add trajectory line
            ax.plot(embedding[:, 0], embedding[:, 1], 'k-', alpha=0.3, linewidth=1)
            
            # Mark start and end
            ax.scatter(embedding[0, 0], embedding[0, 1], c='red', s=100, 
                      marker='o', label='Start', zorder=5)
            ax.scatter(embedding[-1, 0], embedding[-1, 1], c='blue', s=100, 
                      marker='s', label='End', zorder=5)
            
            ax.set_title(f'{method_name} Embedding')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Time (s)')
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'manifold_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. PCA detailed analysis
    if 'pca' in manifold_results:
        pca_results = manifold_results['pca']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Explained variance
        axes[0, 0].bar(range(1, min(21, len(pca_results['explained_variance']) + 1)), 
                       pca_results['explained_variance'][:20])
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        axes[0, 1].plot(range(1, min(21, len(pca_results['cumulative_variance']) + 1)),
                        pca_results['cumulative_variance'][:20], 'o-')
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D PCA trajectory
        pca_2d = pca_results['embedding_2d']
        scatter = axes[1, 0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=time_points, 
                                   cmap='viridis', s=30, alpha=0.7)
        axes[1, 0].plot(pca_2d[:, 0], pca_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].set_title('PCA 2D Trajectory')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Time (s)')
        
        # 3D PCA projection (2D view)
        if 'embedding_3d' in pca_results:
            pca_3d = pca_results['embedding_3d']
            scatter = axes[1, 1].scatter(pca_3d[:, 0], pca_3d[:, 2], c=time_points, 
                                       cmap='viridis', s=30, alpha=0.7)
            axes[1, 1].plot(pca_3d[:, 0], pca_3d[:, 2], 'k-', alpha=0.3, linewidth=1)
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC3')
            axes[1, 1].set_title('PCA 3D Projection (PC1 vs PC3)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Time (s)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Trajectory dynamics analysis
    if trajectory_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        method_names = list(trajectory_results.keys())
        
        # Speed over time
        for method_name in method_names:
            if 'speeds' in trajectory_results[method_name]:
                speeds = trajectory_results[method_name]['speeds']
                axes[0, 0].plot(time_points[1:], speeds, label=method_name, alpha=0.7)
        
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Speed')
        axes[0, 0].set_title('Trajectory Speed Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean speed comparison
        method_labels = []
        mean_speeds = []
        for method_name in method_names:
            if 'mean_speed' in trajectory_results[method_name]:
                method_labels.append(method_name)
                mean_speeds.append(trajectory_results[method_name]['mean_speed'])
        
        if mean_speeds:
            axes[0, 1].bar(method_labels, mean_speeds)
            axes[0, 1].set_ylabel('Mean Speed')
            axes[0, 1].set_title('Average Trajectory Speed by Method')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Curvature analysis
        for method_name in method_names:
            if 'curvatures' in trajectory_results[method_name]:
                curvatures = trajectory_results[method_name]['curvatures']
                if len(curvatures) > 0:
                    axes[1, 0].plot(time_points[2:], curvatures, label=method_name, alpha=0.7)
        
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Curvature (radians)')
        axes[1, 0].set_title('Trajectory Curvature Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stationary periods
        stationary_fractions = []
        for method_name in method_names:
            if 'stationary_fraction' in trajectory_results[method_name]:
                stationary_fractions.append(trajectory_results[method_name]['stationary_fraction'])
        
        if stationary_fractions:
            axes[1, 1].bar(method_labels, stationary_fractions)
            axes[1, 1].set_ylabel('Stationary Period Fraction')
            axes[1, 1].set_title('Fraction of Time in Stationary States')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trajectory_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. State analysis (if state transitions are available)
    if ('state_transitions' in recording_metadata and 
        recording_metadata['state_transitions'] and
        'state_timeline' in recording_metadata):
        
        state_timeline = recording_metadata['state_timeline']
        
        # Create state-colored manifold plots
        if 'pca' in manifold_results:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            pca_2d = manifold_results['pca']['embedding_2d']
            
            # Assign state labels to time points
            state_labels = []
            for t in time_points:
                current_state = 'unknown'
                for state in state_timeline:
                    if state['start'] <= t < state['end']:
                        current_state = state['type']
                        break
                state_labels.append(current_state)
            
            # Color by state
            unique_states = list(set(state_labels))
            state_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_states)))
            color_map = dict(zip(unique_states, state_colors))
            
            colors = [color_map[state] for state in state_labels]
            
            # PCA with state coloring
            axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, s=30, alpha=0.7)
            axes[0].plot(pca_2d[:, 0], pca_2d[:, 1], 'k-', alpha=0.3, linewidth=1)
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].set_title('PCA Trajectory Colored by Network State')
            
            # Create custom legend
            for state, color in color_map.items():
                axes[0].scatter([], [], c=[color], label=state, s=50)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # State timeline
            y_pos = 0
            for state in state_timeline:
                axes[1].barh(y_pos, state['end'] - state['start'], 
                            left=state['start'], height=0.8,
                            color=color_map[state['type']], alpha=0.7)
                # Add state label
                mid_point = (state['start'] + state['end']) / 2
                axes[1].text(mid_point, y_pos, state['type'], 
                           ha='center', va='center', fontweight='bold')
            
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Network State')
            axes[1].set_title('Network State Timeline')
            axes[1].set_yticks([])
            axes[1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'state_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"Manifold visualizations saved to {output_dir}")


def generate_manifold_report(manifold_results: Dict, trajectory_results: Dict,
                           recording_metadata: Dict, output_dir: Path) -> None:
    """
    Generate comprehensive manifold analysis report.
    
    Parameters
    ----------
    manifold_results : Dict
        Manifold analysis results
    trajectory_results : Dict
        Trajectory analysis results
    recording_metadata : Dict
        MEA recording metadata
    output_dir : Path
        Output directory for the report
    """
    logger.info("Generating manifold analysis report")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'manifold_analysis_report.txt', 'w') as f:
        f.write("MEA-Flow Manifold Learning Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic information
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Recording Duration: {recording_metadata.get('duration', 'N/A')} seconds\n")
        f.write(f"Number of Channels: {recording_metadata.get('n_channels', 'N/A')}\n")
        f.write(f"Electrode Layout: {recording_metadata.get('electrode_layout', 'N/A')}\n\n")
        
        # Manifold methods applied
        f.write("Manifold Learning Methods Applied:\n")
        f.write("-" * 35 + "\n")
        for method_name in manifold_results.keys():
            if method_name != 'metadata':
                f.write(f"  ✓ {method_name.upper()}\n")
        f.write("\n")
        
        # Feature extraction details
        metadata = manifold_results['metadata']
        f.write("Feature Extraction:\n")
        f.write("-" * 20 + "\n")
        f.write(f"  Time windows: {metadata['n_timepoints']}\n")
        f.write(f"  Features per window: {metadata['n_features']}\n")
        f.write(f"  Feature types: {', '.join(metadata['feature_names'])}\n\n")
        
        # PCA results
        if 'pca' in manifold_results:
            pca_results = manifold_results['pca']
            f.write("Principal Component Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"  First 5 PCs explain: {pca_results['cumulative_variance'][4]:.1%} of variance\n")
            f.write(f"  First 10 PCs explain: {pca_results['cumulative_variance'][9]:.1%} of variance\n")
            f.write(f"  PC1 explains: {pca_results['explained_variance'][0]:.1%} of variance\n")
            f.write(f"  PC2 explains: {pca_results['explained_variance'][1]:.1%} of variance\n\n")
        
        # Trajectory analysis
        if trajectory_results:
            f.write("Trajectory Dynamics Analysis:\n")
            f.write("-" * 30 + "\n")
            
            for method_name, results in trajectory_results.items():
                f.write(f"  {method_name.upper()}:\n")
                f.write(f"    Mean speed: {results.get('mean_speed', 0):.3f}\n")
                f.write(f"    Trajectory length: {results.get('trajectory_length', 0):.3f}\n")
                f.write(f"    Mean curvature: {results.get('mean_curvature', 0):.3f} rad\n")
                f.write(f"    Stationary periods: {results.get('stationary_fraction', 0):.1%}\n")
                f.write("\n")
        
        # State analysis (if available)
        if ('state_transitions' in recording_metadata and 
            recording_metadata['state_transitions'] and
            'state_timeline' in recording_metadata):
            
            state_timeline = recording_metadata['state_timeline']
            f.write("Network State Analysis:\n")
            f.write("-" * 25 + "\n")
            
            for state in state_timeline:
                duration = state['end'] - state['start']
                f.write(f"  {state['type'].title()} state: {duration:.0f}s "
                       f"({state['start']:.0f}s - {state['end']:.0f}s)\n")
            f.write("\n")
        
        # Recommendations
        f.write("Analysis Recommendations:\n")
        f.write("-" * 25 + "\n")
        
        if 'pca' in manifold_results:
            pca_var = manifold_results['pca']['explained_variance'][0]
            if pca_var > 0.5:
                f.write("  • High PC1 variance suggests strong dominant activity pattern\n")
            elif pca_var < 0.2:
                f.write("  • Low PC1 variance suggests diverse activity patterns\n")
        
        if trajectory_results:
            # Find method with highest trajectory length
            max_length = 0
            best_method = None
            for method_name, results in trajectory_results.items():
                length = results.get('trajectory_length', 0)
                if length > max_length:
                    max_length = length
                    best_method = method_name
            
            if best_method:
                f.write(f"  • {best_method.upper()} shows most dynamic trajectory\n")
        
        f.write("  • Consider longer recordings for better trajectory analysis\n")
        f.write("  • Experiment with different time window sizes\n")
        f.write("  • Apply clustering to identify distinct network states\n")
    
    logger.info(f"Manifold analysis report saved to {output_dir}")


def main():
    """Main workflow for manifold learning analysis."""
    logger.info("Starting Manifold Learning Workflow Analysis")
    
    # Create output directory
    output_dir = Path("output/manifold_learning_workflow")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate synthetic MEA data with population dynamics
    logger.info("Step 1: Generating synthetic MEA data with population dynamics")
    recording = create_dynamic_mea_data(
        n_channels=64,        # 8x8 MEA
        duration=600.0,       # 10 minutes
        state_transitions=True,  # Include network state transitions
        seed=42
    )
    
    # Step 2: Extract population features
    logger.info("Step 2: Extracting population-level features")
    features, time_points = extract_population_features(recording, time_window=10.0)
    
    # Step 3: Perform comprehensive manifold analysis
    logger.info("Step 3: Performing comprehensive manifold analysis")
    manifold_results = perform_comprehensive_manifold_analysis(features, time_points)
    
    # Step 4: Analyze trajectory dynamics
    logger.info("Step 4: Analyzing trajectory dynamics")
    trajectory_results = analyze_trajectory_dynamics(manifold_results)
    
    # Step 5: Create visualizations
    logger.info("Step 5: Creating comprehensive visualizations")
    create_manifold_visualizations(manifold_results, trajectory_results, 
                                 recording.metadata, output_dir)
    
    # Step 6: Generate detailed report
    logger.info("Step 6: Generating detailed analysis report")
    generate_manifold_report(manifold_results, trajectory_results, 
                           recording.metadata, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("MANIFOLD LEARNING WORKFLOW ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")
    
    print(f"\nRecording details:")
    print(f"  - Duration: {recording.duration:.0f} seconds")
    print(f"  - Channels: {len(recording.spike_trains)}")
    print(f"  - Layout: {recording.metadata.get('electrode_layout', 'N/A')}")
    print(f"  - State transitions: {recording.metadata.get('state_transitions', False)}")
    
    print(f"\nFeature extraction:")
    print(f"  - Time windows: {len(time_points)}")
    print(f"  - Features per window: {features.shape[1]}")
    
    print(f"\nManifold methods applied:")
    for method_name in manifold_results.keys():
        if method_name != 'metadata':
            print(f"  - {method_name.upper()}")
    
    if 'pca' in manifold_results:
        pca_var = manifold_results['pca']['explained_variance'][:2]
        print(f"\nPCA results:")
        print(f"  - PC1: {pca_var[0]:.1%} variance")
        print(f"  - PC2: {pca_var[1]:.1%} variance")
    
    print("\nManifold learning workflow completed successfully!")


if __name__ == "__main__":
    main()