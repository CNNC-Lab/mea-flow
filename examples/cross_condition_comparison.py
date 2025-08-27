#!/usr/bin/env python3
"""
Cross-Condition Comparison Example
=================================

This example demonstrates how to use MEA-Flow for comparing neural activity
across different experimental conditions using statistical analysis and 
manifold learning techniques.

Authors: MEA-Flow Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

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


def create_condition_data(condition_name: str, n_channels: int = 16, 
                         duration: float = 300.0, seed: int = None) -> MEARecording:
    """
    Create synthetic MEA data for a specific experimental condition.
    
    Parameters
    ----------
    condition_name : str
        Name of the experimental condition
    n_channels : int
        Number of electrode channels
    duration : float
        Recording duration in seconds
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    MEARecording
        Synthetic MEA recording for the condition
    """
    if seed is not None:
        np.random.seed(seed)
    
    logger.info(f"Creating synthetic data for condition: {condition_name}")
    
    # Define condition-specific parameters
    condition_params = {
        'control': {'base_rate': 5.0, 'burst_prob': 0.1, 'sync_level': 0.3},
        'drug_low': {'base_rate': 3.0, 'burst_prob': 0.05, 'sync_level': 0.2},
        'drug_high': {'base_rate': 1.5, 'burst_prob': 0.02, 'sync_level': 0.1},
        'stimulation': {'base_rate': 10.0, 'burst_prob': 0.2, 'sync_level': 0.6}
    }
    
    params = condition_params.get(condition_name.lower(), condition_params['control'])
    
    # Generate spike data for each channel
    spike_trains = {}
    sampling_rate = 20000.0  # 20 kHz
    
    for channel_id in range(n_channels):
        # Create channel-specific activity pattern
        channel_rate = params['base_rate'] * np.random.uniform(0.5, 1.5)
        burst_prob = params['burst_prob'] * np.random.uniform(0.5, 2.0)
        
        # Generate spike times
        spike_times = []
        t = 0.0
        
        while t < duration:
            # Random interval between spikes
            if np.random.random() < burst_prob:
                # Generate burst
                burst_duration = np.random.uniform(0.1, 0.5)
                burst_rate = channel_rate * 10
                burst_end = t + burst_duration
                
                while t < burst_end and t < duration:
                    t += np.random.exponential(1.0 / burst_rate)
                    if t < duration:
                        spike_times.append(t)
            else:
                # Single spike
                t += np.random.exponential(1.0 / channel_rate)
                if t < duration:
                    spike_times.append(t)
        
        # Add synchronous spikes based on sync_level
        if np.random.random() < params['sync_level']:
            sync_times = np.random.uniform(0, duration, int(duration * 0.5))
            spike_times.extend(sync_times)
        
        # Convert to sample indices and sort
        spike_times = np.array(sorted(spike_times))
        spike_samples = (spike_times * sampling_rate).astype(int)
        
        # Create SpikeData object
        spike_trains[f"Channel_{channel_id:02d}"] = SpikeData(
            spike_times=spike_times,
            spike_samples=spike_samples,
            channel_id=f"Channel_{channel_id:02d}",
            sampling_rate=sampling_rate
        )
    
    # Create MEA recording
    recording = MEARecording(
        spike_trains=spike_trains,
        duration=duration,
        sampling_rate=sampling_rate,
        metadata={
            'condition': condition_name,
            'n_channels': n_channels,
            'created_by': 'cross_condition_comparison.py'
        }
    )
    
    logger.info(f"Created recording with {len(spike_trains)} channels, "
                f"duration: {duration}s")
    
    return recording


def analyze_condition_activity(recording: MEARecording, condition_name: str) -> dict:
    """
    Perform comprehensive activity analysis for a condition.
    
    Parameters
    ----------
    recording : MEARecording
        MEA recording data
    condition_name : str
        Name of the experimental condition
        
    Returns
    -------
    dict
        Analysis results including firing rates, burst metrics, and synchrony
    """
    logger.info(f"Analyzing activity for condition: {condition_name}")
    
    analyzer = ActivityAnalyzer()
    burst_detector = BurstDetector()
    
    results = {
        'condition': condition_name,
        'firing_rates': {},
        'burst_metrics': {},
        'synchrony_measures': {},
        'network_bursts': None
    }
    
    # Calculate firing rates for each channel
    for channel_id, spike_data in recording.spike_trains.items():
        firing_rate = analyzer.calculate_firing_rate(spike_data)
        results['firing_rates'][channel_id] = firing_rate
    
    # Detect bursts for each channel
    for channel_id, spike_data in recording.spike_trains.items():
        try:
            bursts = burst_detector.detect_bursts(spike_data)
            if bursts:
                burst_rate = len(bursts) / recording.duration * 60  # bursts per minute
                avg_burst_duration = np.mean([b.duration for b in bursts])
                avg_spikes_per_burst = np.mean([len(b.spike_times) for b in bursts])
                
                results['burst_metrics'][channel_id] = {
                    'burst_rate': burst_rate,
                    'avg_duration': avg_burst_duration,
                    'avg_spikes_per_burst': avg_spikes_per_burst,
                    'n_bursts': len(bursts)
                }
        except Exception as e:
            logger.warning(f"Burst detection failed for {channel_id}: {e}")
            results['burst_metrics'][channel_id] = {
                'burst_rate': 0.0,
                'avg_duration': 0.0,
                'avg_spikes_per_burst': 0.0,
                'n_bursts': 0
            }
    
    # Calculate synchrony measures
    try:
        spike_trains_list = list(recording.spike_trains.values())
        if len(spike_trains_list) >= 2:
            synchrony = analyzer.calculate_synchrony(spike_trains_list[:2])
            results['synchrony_measures']['pairwise'] = synchrony
    except Exception as e:
        logger.warning(f"Synchrony calculation failed: {e}")
        results['synchrony_measures']['pairwise'] = 0.0
    
    # Detect network bursts
    try:
        network_bursts = burst_detector.detect_network_bursts(
            list(recording.spike_trains.values())
        )
        results['network_bursts'] = {
            'count': len(network_bursts) if network_bursts else 0,
            'rate': (len(network_bursts) / recording.duration * 60 
                    if network_bursts else 0.0),
            'bursts': network_bursts
        }
    except Exception as e:
        logger.warning(f"Network burst detection failed: {e}")
        results['network_bursts'] = {'count': 0, 'rate': 0.0, 'bursts': []}
    
    return results


def perform_manifold_analysis(recordings: dict) -> dict:
    """
    Perform manifold learning analysis across conditions.
    
    Parameters
    ----------
    recordings : dict
        Dictionary of condition_name -> MEARecording
        
    Returns
    -------
    dict
        Manifold analysis results
    """
    logger.info("Performing manifold learning analysis across conditions")
    
    manifold_analyzer = ManifoldAnalyzer()
    
    # Prepare feature matrix
    features = []
    labels = []
    condition_names = []
    
    for condition_name, recording in recordings.items():
        logger.info(f"Extracting features for condition: {condition_name}")
        
        # Extract activity features for each channel
        for channel_id, spike_data in recording.spike_trains.items():
            # Basic activity features
            firing_rate = len(spike_data.spike_times) / recording.duration
            
            # ISI statistics
            if len(spike_data.spike_times) > 1:
                isis = np.diff(spike_data.spike_times)
                isi_mean = np.mean(isis)
                isi_std = np.std(isis)
                isi_cv = isi_std / isi_mean if isi_mean > 0 else 0
            else:
                isi_mean = isi_std = isi_cv = 0
            
            # Binned activity (1-second bins)
            bins = np.arange(0, recording.duration + 1, 1.0)
            activity_profile, _ = np.histogram(spike_data.spike_times, bins=bins)
            activity_variance = np.var(activity_profile)
            
            # Combine features
            feature_vector = [
                firing_rate,
                isi_mean,
                isi_std,
                isi_cv,
                activity_variance,
                np.max(activity_profile),
                np.mean(activity_profile)
            ]
            
            features.append(feature_vector)
            labels.append(condition_name)
            condition_names.append(f"{condition_name}_{channel_id}")
    
    features_array = np.array(features)
    
    # Perform dimensionality reduction
    methods = ['pca', 'tsne', 'umap']
    manifold_results = {}
    
    for method in methods:
        try:
            logger.info(f"Applying {method.upper()} dimensionality reduction")
            
            if method == 'pca':
                embedding = manifold_analyzer.apply_pca(features_array, n_components=2)
            elif method == 'tsne':
                embedding = manifold_analyzer.apply_tsne(features_array, n_components=2)
            elif method == 'umap':
                try:
                    embedding = manifold_analyzer.apply_umap(features_array, n_components=2)
                except ImportError:
                    logger.warning("UMAP not available, skipping")
                    continue
            
            manifold_results[method] = {
                'embedding': embedding,
                'labels': labels,
                'condition_names': condition_names
            }
            
        except Exception as e:
            logger.error(f"Failed to apply {method}: {e}")
    
    return manifold_results


def create_comparison_plots(analysis_results: dict, manifold_results: dict, 
                           output_dir: Path) -> None:
    """
    Create comprehensive comparison plots.
    
    Parameters
    ----------
    analysis_results : dict
        Analysis results for each condition
    manifold_results : dict
        Manifold learning results
    output_dir : Path
        Output directory for plots
    """
    logger.info("Creating comparison plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Firing rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = list(analysis_results.keys())
    firing_rates_by_condition = {}
    
    for condition in conditions:
        rates = list(analysis_results[condition]['firing_rates'].values())
        firing_rates_by_condition[condition] = rates
    
    # Box plot of firing rates
    data_to_plot = [firing_rates_by_condition[condition] for condition in conditions]
    box_plot = ax.boxplot(data_to_plot, labels=conditions, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(conditions)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate Comparison Across Conditions')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'firing_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Burst metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = ['burst_rate', 'avg_duration', 'avg_spikes_per_burst', 'n_bursts']
    metric_labels = ['Burst Rate (per min)', 'Avg Duration (s)', 
                    'Avg Spikes per Burst', 'Number of Bursts']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        data_to_plot = []
        for condition in conditions:
            values = []
            for channel_metrics in analysis_results[condition]['burst_metrics'].values():
                if isinstance(channel_metrics, dict):
                    values.append(channel_metrics.get(metric, 0))
            data_to_plot.append(values)
        
        if any(len(d) > 0 for d in data_to_plot):
            box_plot = axes[i].boxplot(data_to_plot, labels=conditions, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
        
        axes[i].set_ylabel(label)
        axes[i].set_title(f'{label} Comparison')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'burst_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Network burst comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    network_burst_rates = []
    for condition in conditions:
        rate = analysis_results[condition]['network_bursts']['rate']
        network_burst_rates.append(rate)
    
    bars = ax.bar(conditions, network_burst_rates, color=colors[:len(conditions)])
    ax.set_ylabel('Network Burst Rate (per min)')
    ax.set_title('Network Burst Rate Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, network_burst_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'network_burst_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Manifold learning plots
    if manifold_results:
        n_methods = len(manifold_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        for i, (method, results) in enumerate(manifold_results.items()):
            embedding = results['embedding']
            labels = results['labels']
            
            # Create color map for conditions
            unique_conditions = list(set(labels))
            color_map = dict(zip(unique_conditions, 
                               plt.cm.Set1(np.linspace(0, 1, len(unique_conditions)))))
            
            # Plot each condition
            for condition in unique_conditions:
                mask = np.array(labels) == condition
                axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[color_map[condition]], label=condition, alpha=0.6)
            
            axes[i].set_title(f'{method.upper()} Embedding')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'manifold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Plots saved to {output_dir}")


def generate_statistical_report(analysis_results: dict, output_dir: Path) -> None:
    """
    Generate statistical comparison report.
    
    Parameters
    ----------
    analysis_results : dict
        Analysis results for each condition
    output_dir : Path
        Output directory for the report
    """
    logger.info("Generating statistical comparison report")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for statistical analysis
    data_records = []
    
    for condition, results in analysis_results.items():
        for channel_id, firing_rate in results['firing_rates'].items():
            record = {
                'condition': condition,
                'channel': channel_id,
                'firing_rate': firing_rate,
                'synchrony': results['synchrony_measures'].get('pairwise', 0),
                'network_burst_rate': results['network_bursts']['rate']
            }
            
            # Add burst metrics if available
            if channel_id in results['burst_metrics']:
                burst_metrics = results['burst_metrics'][channel_id]
                if isinstance(burst_metrics, dict):
                    record.update({
                        'burst_rate': burst_metrics.get('burst_rate', 0),
                        'avg_burst_duration': burst_metrics.get('avg_duration', 0),
                        'avg_spikes_per_burst': burst_metrics.get('avg_spikes_per_burst', 0)
                    })
            
            data_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Generate summary statistics
    summary_stats = df.groupby('condition').agg({
        'firing_rate': ['mean', 'std', 'count'],
        'burst_rate': ['mean', 'std'],
        'network_burst_rate': ['mean', 'std'],
        'synchrony': ['mean', 'std']
    }).round(3)
    
    # Save summary statistics
    with open(output_dir / 'statistical_summary.txt', 'w') as f:
        f.write("MEA-Flow Cross-Condition Statistical Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis performed on: {pd.Timestamp.now()}\n\n")
        f.write("Summary Statistics by Condition:\n")
        f.write(str(summary_stats))
        f.write("\n\n")
        
        # Add condition comparison
        conditions = df['condition'].unique()
        if len(conditions) > 1:
            f.write("Pairwise Comparisons (Firing Rate):\n")
            f.write("-" * 35 + "\n")
            
            for i, cond1 in enumerate(conditions):
                for cond2 in conditions[i+1:]:
                    data1 = df[df['condition'] == cond1]['firing_rate']
                    data2 = df[df['condition'] == cond2]['firing_rate']
                    
                    # Simple statistical comparison
                    mean_diff = data1.mean() - data2.mean()
                    f.write(f"{cond1} vs {cond2}: Δ = {mean_diff:.3f} Hz\n")
    
    # Save detailed data
    df.to_csv(output_dir / 'detailed_analysis_results.csv', index=False)
    
    logger.info(f"Statistical report saved to {output_dir}")


def main():
    """Main analysis workflow for cross-condition comparison."""
    logger.info("Starting Cross-Condition Comparison Analysis")
    
    # Create output directory
    output_dir = Path("output/cross_condition_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experimental conditions
    conditions = ['control', 'drug_low', 'drug_high', 'stimulation']
    
    # Step 1: Generate synthetic data for each condition
    logger.info("Step 1: Generating synthetic MEA data for each condition")
    recordings = {}
    
    for i, condition in enumerate(conditions):
        recordings[condition] = create_condition_data(
            condition_name=condition,
            n_channels=16,
            duration=300.0,  # 5 minutes
            seed=42 + i  # Different seed for each condition
        )
    
    # Step 2: Analyze each condition
    logger.info("Step 2: Analyzing neural activity for each condition")
    analysis_results = {}
    
    for condition, recording in recordings.items():
        analysis_results[condition] = analyze_condition_activity(recording, condition)
    
    # Step 3: Perform manifold learning analysis
    logger.info("Step 3: Performing manifold learning analysis")
    manifold_results = perform_manifold_analysis(recordings)
    
    # Step 4: Create comparison visualizations
    logger.info("Step 4: Creating comparison visualizations")
    create_comparison_plots(analysis_results, manifold_results, output_dir)
    
    # Step 5: Generate statistical report
    logger.info("Step 5: Generating statistical comparison report")
    generate_statistical_report(analysis_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-CONDITION COMPARISON ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")
    
    print(f"\nAnalyzed {len(conditions)} conditions:")
    for condition in conditions:
        n_channels = len(analysis_results[condition]['firing_rates'])
        avg_firing_rate = np.mean(list(analysis_results[condition]['firing_rates'].values()))
        network_bursts = analysis_results[condition]['network_bursts']['count']
        print(f"  - {condition}: {n_channels} channels, "
              f"avg firing rate = {avg_firing_rate:.2f} Hz, "
              f"network bursts = {network_bursts}")
    
    print(f"\nManifold methods applied: {list(manifold_results.keys())}")
    print("\nAnalysis workflow completed successfully!")


if __name__ == "__main__":
    main()