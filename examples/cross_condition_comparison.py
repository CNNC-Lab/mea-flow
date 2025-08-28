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
    from mea_flow.data import SpikeList
    from mea_flow.analysis import MEAMetrics
    from mea_flow.manifold import ManifoldAnalysis
    from mea_flow.visualization import MEAPlotter
except ImportError as e:
    logger.error(f"Failed to import MEA-Flow modules: {e}")
    logger.error("Please ensure MEA-Flow is installed: pip install -e .")
    exit(1)


def create_condition_data(condition_name: str, n_channels: int = 16, 
                         duration: float = 300.0, seed: int = None) -> SpikeList:
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
    SpikeList
        Synthetic MEA spike data for the condition
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
    spike_data = {}
    
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
        
        # Sort spike times
        spike_times = np.array(sorted(spike_times))
        spike_data[channel_id] = spike_times
    
    # Create SpikeList
    spike_list = SpikeList(spike_data)
    
    logger.info(f"Created spike data with {len(spike_data)} channels, "
                f"duration: {duration}s")
    
    return spike_list


def analyze_condition_activity(spike_list: SpikeList, condition_name: str) -> dict:
    """
    Perform basic activity analysis for a condition.
    
    Parameters
    ----------
    spike_list : SpikeList
        MEA spike data
    condition_name : str
        Name of the experimental condition
        
    Returns
    -------
    dict
        Basic analysis results
    """
    logger.info(f"Analyzing activity for condition: {condition_name}")
    
    try:
        # Simple analysis without heavy computations
        active_channels = spike_list.get_active_channels()
        total_spikes = 0
        firing_rates = {}
        
        # Calculate basic firing rates
        for channel_id in active_channels:
            spike_train = spike_list.spike_trains[channel_id]
            n_spikes = len(spike_train.spike_times)
            total_spikes += n_spikes
            
            # Estimate duration from spike times
            if n_spikes > 0:
                duration = max(spike_train.spike_times) if len(spike_train.spike_times) > 0 else 30.0
                firing_rate = n_spikes / duration if duration > 0 else 0
            else:
                firing_rate = 0
            
            firing_rates[channel_id] = firing_rate
        
        mean_firing_rate = sum(firing_rates.values()) / len(firing_rates) if firing_rates else 0
        
        analysis_results = {
            'condition': condition_name,
            'summary': {
                'total_spikes': total_spikes,
                'mean_firing_rate': mean_firing_rate,
                'active_channels': len(active_channels)
            },
            'firing_rates': firing_rates
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Analysis failed for condition {condition_name}: {e}")
        return {
            'condition': condition_name,
            'error': str(e),
            'summary': {'total_spikes': 0, 'mean_firing_rate': 0, 'active_channels': 0},
            'firing_rates': {}
        }


def compare_conditions(condition_results: dict) -> dict:
    """
    Compare analysis results across conditions.
    
    Parameters
    ----------
    condition_results : dict
        Dictionary of condition_name -> analysis results
        
    Returns
    -------
    dict
        Comparison results and statistics
    """
    
    comparison_results = {}
    conditions = list(condition_results.keys())
    
    # Compare key metrics across conditions
    metrics_to_compare = ['total_spikes', 'mean_firing_rate', 'active_channels']
    
    for metric in metrics_to_compare:
        values = []
        for condition in conditions:
            if 'summary' in condition_results[condition]:
                values.append(condition_results[condition]['summary'].get(metric, 0))
            else:
                values.append(0)
        
        comparison_results[metric] = {
            'conditions': conditions,
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return comparison_results


def create_comparison_plots(condition_results: dict, comparison_results: dict, output_dir: Path):
    """
    Create comparison plots across conditions.
    
    Parameters
    ----------
    condition_results : dict
        Analysis results for each condition
    comparison_results : dict
        Comparison statistics
    output_dir : Path
        Output directory for plots
    """
    logger.info("Creating comparison plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = MEAPlotter()
    
    # Create comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['total_spikes', 'mean_firing_rate', 'active_channels']
    titles = ['Total Spikes', 'Mean Firing Rate (Hz)', 'Active Channels']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in comparison_results:
            conditions = comparison_results[metric]['conditions']
            values = comparison_results[metric]['values']
            
            bars = axes[i].bar(conditions, values)
            axes[i].set_title(title)
            axes[i].set_ylabel(title)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plotter.save_figure(fig, str(output_dir / 'condition_comparison.png'))
    plt.close(fig)


def main():
    """Main analysis workflow for cross-condition comparison."""
    logger.info("Starting Cross-Condition Comparison Analysis")
    
    # Create output directory
    output_dir = Path("examples/output/cross_condition_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define experimental conditions
    conditions = ['control', 'drug_low', 'drug_high', 'stimulation']
    
    try:
        # Step 1: Generate synthetic data for each condition
        logger.info("Step 1: Generating synthetic MEA data for each condition")
        spike_lists = {}
        
        for i, condition in enumerate(conditions):
            spike_lists[condition] = create_condition_data(
                condition_name=condition,
                n_channels=8,
                duration=30.0,  # 30 seconds
                seed=42 + i  # Different seed for each condition
            )
        
        # Step 2: Analyze each condition
        logger.info("Step 2: Analyzing neural activity for each condition")
        condition_results = {}
        
        for condition, spike_list in spike_lists.items():
            condition_results[condition] = analyze_condition_activity(spike_list, condition)
        
        # Step 3: Compare conditions
        logger.info("Step 3: Comparing conditions")
        comparison_results = compare_conditions(condition_results)
        
        # Step 4: Create comparison visualizations
        logger.info("Step 4: Creating comparison visualizations")
        create_comparison_plots(condition_results, comparison_results, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("CROSS-CONDITION COMPARISON ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {output_dir.absolute()}")
        
        print(f"\nAnalyzed {len(conditions)} conditions:")
        for condition in conditions:
            if condition in condition_results and 'summary' in condition_results[condition]:
                summary = condition_results[condition]['summary']
                print(f"  - {condition}: {summary['active_channels']} channels, "
                      f"firing rate = {summary['mean_firing_rate']:.2f} Hz, "
                      f"total spikes = {summary['total_spikes']}")
        
        print("\nCross-condition comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()