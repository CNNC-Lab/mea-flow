#!/usr/bin/env python3
"""
Basic MEA Analysis Example
=========================

This example demonstrates the basic workflow for analyzing MEA data using MEA-Flow.
It covers loading data, computing metrics, and creating visualizations.

Run this script with:
    python examples/basic_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import MEA-Flow components
from mea_flow import SpikeList, MEAMetrics, MEAPlotter


def create_synthetic_data():
    """Create synthetic MEA data for demonstration."""
    print("🔄 Creating synthetic MEA data...")
    
    # Parameters
    n_channels = 16
    recording_length = 30.0  # seconds
    np.random.seed(42)  # For reproducibility
    
    spike_times = []
    channels = []
    
    # Generate spikes for each channel
    for ch in range(n_channels):
        # Different activity levels per channel
        if ch < 4:  # High activity channels
            n_spikes = np.random.poisson(200)
        elif ch < 8:  # Medium activity channels
            n_spikes = np.random.poisson(100)
        elif ch < 12:  # Low activity channels
            n_spikes = np.random.poisson(50)
        else:  # Very low activity channels
            n_spikes = np.random.poisson(10)
        
        # Generate spike times
        ch_spike_times = np.sort(np.random.uniform(0, recording_length, n_spikes))
        
        # Add some burst activity for first few channels
        if ch < 6:
            n_bursts = np.random.randint(3, 8)
            for _ in range(n_bursts):
                burst_start = np.random.uniform(2, recording_length - 2)
                burst_spikes = np.random.poisson(15)
                burst_times = np.random.uniform(burst_start, burst_start + 0.2, burst_spikes)
                ch_spike_times = np.concatenate([ch_spike_times, burst_times])
        
        # Sort and add to lists
        ch_spike_times = np.sort(ch_spike_times)
        spike_times.extend(ch_spike_times)
        channels.extend([ch] * len(ch_spike_times))
    
    # Create SpikeList
    spike_list = SpikeList(
        spike_data={'times': np.array(spike_times), 'channels': np.array(channels)},
        recording_length=recording_length
    )
    
    print(f"✅ Created synthetic data with {len(spike_times)} spikes across {n_channels} channels")
    return spike_list


def analyze_activity(spike_list):
    """Perform basic activity analysis."""
    print("\n📊 Computing activity metrics...")
    
    # Initialize metrics calculator
    metrics = MEAMetrics()
    
    # Compute comprehensive metrics
    results_df = metrics.compute_all_metrics(spike_list, grouping='global')
    
    # Extract nested results for easier access
    if hasattr(results_df, 'attrs') and 'nested_results' in results_df.attrs:
        results = results_df.attrs['nested_results']
    else:
        # Fallback: reconstruct from flattened DataFrame
        results = {
            'activity': {},
            'regularity': {},
            'synchrony': {}
        }
        for col in results_df.columns:
            if col.startswith('activity_'):
                key = col.replace('activity_', '')
                results['activity'][key] = results_df[col].iloc[0]
            elif col.startswith('regularity_'):
                key = col.replace('regularity_', '')
                results['regularity'][key] = results_df[col].iloc[0]
            elif col.startswith('synchrony_'):
                key = col.replace('synchrony_', '')
                results['synchrony'][key] = results_df[col].iloc[0]
    
    # Print key results
    activity = results['activity']
    print(f"📈 Activity Metrics:")
    print(f"   • Total spikes: {activity.get('total_spike_count', 0)}")
    print(f"   • Active channels: {activity.get('active_channels_count', 0)}")
    print(f"   • Mean firing rate: {activity.get('mean_firing_rate', 0):.2f} ± {activity.get('std_firing_rate', 0):.2f} Hz")
    print(f"   • Network firing rate: {activity.get('network_firing_rate', 0):.2f} Hz")
    
    regularity = results['regularity']
    print(f"🎯 Regularity Metrics:")
    print(f"   • Mean CV-ISI: {regularity.get('mean_cv_isi', 0):.3f}")
    print(f"   • Mean LV: {regularity.get('mean_lv', 0):.3f}")
    
    synchrony = results['synchrony']
    print(f"🔗 Synchrony Metrics:")
    print(f"   • Mean correlation: {synchrony.get('pearson_cc_mean', 0):.3f}")
    print(f"   • Synchrony index: {synchrony.get('synchrony_index', 0):.3f}")
    
    return results


def create_visualizations(spike_list, results):
    """Create and save visualizations."""
    print("\n🎨 Creating visualizations...")
    
    # Initialize plotter
    plotter = MEAPlotter(figsize=(12, 8))
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Raster plot
    print("   • Creating raster plot...")
    fig = plotter.plot_raster(
        spike_list,
        time_range=(0, 10),  # First 10 seconds
        figsize=(12, 8)
    )
    fig.suptitle("MEA Recording - First 10 seconds", fontsize=14)
    plt.savefig(output_dir / "raster_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 2. Activity summary
    print("   • Creating activity summary...")
    fig = plotter.plot_activity_summary(spike_list, results['activity'])
    plotter.save_figure(fig, output_dir / "activity_summary.png", dpi=300)
    plt.close(fig)
    
    # 3. Synchrony analysis
    print("   • Creating synchrony analysis...")
    fig = plotter.plot_synchrony_analysis(spike_list, results['synchrony'])
    plotter.save_figure(fig, output_dir / "synchrony_analysis.png", dpi=300)
    plt.close(fig)
    
    # 4. Comprehensive summary report
    print("   • Creating summary report...")
    fig = plotter.create_summary_report([spike_list], results)
    plotter.save_figure(fig, output_dir / "summary_report.png", dpi=300)
    plt.close(fig)
    
    print(f"✅ Saved visualizations to {output_dir}/")


def analyze_by_channel(spike_list):
    """Demonstrate per-channel analysis."""
    print("\n🔍 Per-channel analysis...")
    
    metrics = MEAMetrics()
    channel_results = metrics.compute_all_metrics(spike_list, grouping='channel')
    
    print("Channel-wise firing rates:")
    for ch_id in sorted(channel_results.keys()):
        if isinstance(channel_results[ch_id], dict):
            firing_rate = channel_results[ch_id].get('firing_rate', 0)
            n_spikes = spike_list.spike_trains[ch_id].n_spikes if ch_id in spike_list.spike_trains else 0
            print(f"   Channel {ch_id:2d}: {firing_rate:5.2f} Hz ({n_spikes} spikes)")


def main():
    """Main analysis workflow."""
    print("🧪 MEA-Flow Basic Analysis Example")
    print("=" * 50)
    
    # Step 1: Create or load data
    spike_list = create_synthetic_data()
    
    # Step 2: Analyze activity
    results = analyze_activity(spike_list)
    
    # Step 3: Per-channel analysis
    analyze_by_channel(spike_list)
    
    # Step 4: Create visualizations
    create_visualizations(spike_list, results)
    
    print("\n🎉 Analysis complete!")
    print("\nNext steps:")
    print("• Check the generated plots in examples/output/")
    print("• Try loading your own MEA data")
    print("• Explore advanced tutorials in notebooks/")


if __name__ == "__main__":
    main()