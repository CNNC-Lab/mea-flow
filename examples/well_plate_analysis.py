#!/usr/bin/env python3
"""
Well Plate Analysis Example
===========================

This example demonstrates how to use MEA-Flow for analyzing multi-well MEA plate data,
including well-based organization, cross-well comparisons, and dose-response analysis.

Authors: MEA-Flow Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mea_flow.data import SpikeData, MEARecording
    from mea_flow.analysis import ActivityAnalyzer, BurstDetector
    from mea_flow.visualization import ActivityPlotter
except ImportError as e:
    logger.error(f"Failed to import MEA-Flow modules: {e}")
    logger.error("Please ensure MEA-Flow is installed: pip install -e .")
    exit(1)

# Set plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class WellPlateAnalyzer:
    """
    Analyzer for multi-well MEA plate data with well-based organization.
    """
    
    def __init__(self, plate_format: str = "48-well"):
        """
        Initialize well plate analyzer.
        
        Parameters
        ----------
        plate_format : str
            Format of the MEA plate (e.g., "48-well", "96-well")
        """
        self.plate_format = plate_format
        self.well_layout = self._setup_well_layout()
        self.electrode_layout = self._setup_electrode_layout()
        
        logger.info(f"Initialized {plate_format} MEA plate analyzer")
    
    def _setup_well_layout(self) -> Dict:
        """Setup well layout configuration."""
        if self.plate_format == "48-well":
            return {
                'rows': 6,          # A-F
                'columns': 8,       # 1-8
                'total_wells': 48,
                'electrodes_per_well': 16  # 4x4 grid
            }
        elif self.plate_format == "96-well":
            return {
                'rows': 8,          # A-H
                'columns': 12,      # 1-12
                'total_wells': 96,
                'electrodes_per_well': 16
            }
        else:
            raise ValueError(f"Unsupported plate format: {self.plate_format}")
    
    def _setup_electrode_layout(self) -> Dict:
        """Setup electrode layout within each well (4x4 grid)."""
        electrode_positions = {}
        for i in range(16):  # 4x4 = 16 electrodes per well
            row = i // 4
            col = i % 4
            electrode_positions[f"E{i+1:02d}"] = (row, col)
        return electrode_positions
    
    def get_well_id(self, row_idx: int, col_idx: int) -> str:
        """Convert row/column indices to well ID (e.g., A01, B02)."""
        row_letter = chr(ord('A') + row_idx)
        return f"{row_letter}{col_idx+1:02d}"
    
    def parse_well_id(self, well_id: str) -> Tuple[int, int]:
        """Parse well ID to get row/column indices."""
        row_idx = ord(well_id[0]) - ord('A')
        col_idx = int(well_id[1:]) - 1
        return row_idx, col_idx


def create_well_plate_data(analyzer: WellPlateAnalyzer, 
                          experimental_design: Dict,
                          duration: float = 300.0,
                          seed: int = 42) -> Dict[str, MEARecording]:
    """
    Create synthetic MEA data for a multi-well plate experiment.
    
    Parameters
    ----------
    analyzer : WellPlateAnalyzer
        Well plate analyzer instance
    experimental_design : Dict
        Experimental design with conditions and concentrations
    duration : float
        Recording duration in seconds
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, MEARecording]
        Dictionary mapping well IDs to MEA recordings
    """
    np.random.seed(seed)
    logger.info("Creating synthetic well plate MEA data")
    
    well_recordings = {}
    sampling_rate = 20000.0
    
    # Process each well according to experimental design
    for well_id, condition_info in experimental_design.items():
        condition = condition_info['condition']
        concentration = condition_info.get('concentration', 0)
        
        logger.info(f"Creating data for well {well_id}: {condition} "
                   f"(concentration: {concentration})")
        
        # Define condition-specific base parameters
        base_params = {
            'control': {'base_rate': 5.0, 'burst_prob': 0.08, 'sync_level': 0.3},
            'drug_A': {'base_rate': 3.0, 'burst_prob': 0.05, 'sync_level': 0.2},
            'drug_B': {'base_rate': 8.0, 'burst_prob': 0.12, 'sync_level': 0.4},
            'stimulant': {'base_rate': 12.0, 'burst_prob': 0.15, 'sync_level': 0.6},
            'inhibitor': {'base_rate': 1.5, 'burst_prob': 0.02, 'sync_level': 0.1}
        }
        
        # Apply dose-response effects
        params = base_params.get(condition, base_params['control']).copy()
        if concentration > 0:
            # Simple dose-response model
            dose_effect = 1.0 + (concentration - 1.0) * 0.3  # 30% change per log unit
            params['base_rate'] *= dose_effect
            params['burst_prob'] *= dose_effect
            params['sync_level'] = min(0.9, params['sync_level'] * dose_effect)
        
        # Add well-specific variability
        well_row, well_col = analyzer.parse_well_id(well_id)
        edge_effect = 0.9 if (well_row == 0 or well_row == analyzer.well_layout['rows']-1 or
                             well_col == 0 or well_col == analyzer.well_layout['columns']-1) else 1.0
        
        # Generate spike data for electrodes in this well
        spike_trains = {}
        
        for electrode_id in range(analyzer.well_layout['electrodes_per_well']):
            electrode_name = f"Well_{well_id}_E{electrode_id+1:02d}"
            
            # Electrode-specific parameters
            electrode_row, electrode_col = analyzer.electrode_layout[f"E{electrode_id+1:02d}"]
            position_factor = 1.0 + 0.2 * np.sin(electrode_row * np.pi / 4) * np.cos(electrode_col * np.pi / 4)
            
            firing_rate = params['base_rate'] * position_factor * edge_effect
            burst_prob = params['burst_prob']
            sync_level = params['sync_level']
            
            # Generate spike times
            spike_times = []
            t = 0.0
            
            while t < duration:
                if np.random.random() < burst_prob:
                    # Generate burst
                    burst_duration = np.random.uniform(0.1, 0.6)
                    burst_rate = firing_rate * np.random.uniform(8, 20)
                    burst_end = min(t + burst_duration, duration)
                    
                    while t < burst_end:
                        t += np.random.exponential(1.0 / burst_rate)
                        if t < duration:
                            spike_times.append(t)
                else:
                    # Regular spike
                    t += np.random.exponential(1.0 / firing_rate)
                    if t < duration:
                        spike_times.append(t)
            
            # Add well-level synchronous events
            if np.random.random() < sync_level:
                n_sync_events = int(duration / 45)  # Every ~45 seconds
                sync_times = np.random.uniform(0, duration, n_sync_events)
                for sync_time in sync_times:
                    if np.random.random() < 0.8:  # 80% participation
                        jittered_time = sync_time + np.random.normal(0, 0.008)  # 8ms jitter
                        spike_times.append(max(0, jittered_time))
            
            # Sort and convert to samples
            spike_times = np.array(sorted(spike_times))
            spike_samples = (spike_times * sampling_rate).astype(int)
            
            # Create SpikeData object
            spike_trains[electrode_name] = SpikeData(
                spike_times=spike_times,
                spike_samples=spike_samples,
                channel_id=electrode_name,
                sampling_rate=sampling_rate
            )
        
        # Create MEA recording for this well
        well_recordings[well_id] = MEARecording(
            spike_trains=spike_trains,
            duration=duration,
            sampling_rate=sampling_rate,
            metadata={
                'well_id': well_id,
                'condition': condition,
                'concentration': concentration,
                'well_position': (well_row, well_col),
                'edge_effect': edge_effect,
                'created_by': 'well_plate_analysis.py'
            }
        )
    
    logger.info(f"Created recordings for {len(well_recordings)} wells")
    return well_recordings


def analyze_well_activity(recording: MEARecording) -> Dict:
    """
    Analyze neural activity for a single well.
    
    Parameters
    ----------
    recording : MEARecording
        MEA recording for one well
        
    Returns
    -------
    Dict
        Analysis results for the well
    """
    well_id = recording.metadata['well_id']
    logger.debug(f"Analyzing activity for well {well_id}")
    
    analyzer = ActivityAnalyzer()
    burst_detector = BurstDetector()
    
    results = {
        'well_id': well_id,
        'condition': recording.metadata.get('condition', 'unknown'),
        'concentration': recording.metadata.get('concentration', 0),
        'electrode_metrics': {},
        'well_summary': {}
    }
    
    # Analyze each electrode in the well
    firing_rates = []
    burst_rates = []
    active_electrodes = 0
    
    for electrode_id, spike_data in recording.spike_trains.items():
        # Basic activity metrics
        firing_rate = analyzer.calculate_firing_rate(spike_data)
        firing_rates.append(firing_rate)
        
        # Burst detection
        try:
            bursts = burst_detector.detect_bursts(spike_data)
            if bursts:
                burst_rate = len(bursts) / recording.duration * 60  # per minute
                avg_burst_duration = np.mean([b.duration for b in bursts])
                avg_spikes_per_burst = np.mean([len(b.spike_times) for b in bursts])
                burst_rates.append(burst_rate)
            else:
                burst_rate = avg_burst_duration = avg_spikes_per_burst = 0
                burst_rates.append(0)
        except Exception as e:
            logger.warning(f"Burst detection failed for {electrode_id}: {e}")
            burst_rate = avg_burst_duration = avg_spikes_per_burst = 0
            burst_rates.append(0)
        
        # Activity threshold (consider active if firing rate > 0.1 Hz)
        is_active = firing_rate > 0.1
        if is_active:
            active_electrodes += 1
        
        # Store electrode-specific results
        results['electrode_metrics'][electrode_id] = {
            'firing_rate': firing_rate,
            'burst_rate': burst_rate,
            'avg_burst_duration': avg_burst_duration,
            'avg_spikes_per_burst': avg_spikes_per_burst,
            'is_active': is_active,
            'total_spikes': len(spike_data.spike_times)
        }
    
    # Calculate well-level summary statistics
    results['well_summary'] = {
        'mean_firing_rate': np.mean(firing_rates),
        'std_firing_rate': np.std(firing_rates),
        'max_firing_rate': np.max(firing_rates),
        'mean_burst_rate': np.mean(burst_rates),
        'std_burst_rate': np.std(burst_rates),
        'active_electrodes': active_electrodes,
        'active_electrode_fraction': active_electrodes / len(recording.spike_trains),
        'total_spikes': sum(len(spike_data.spike_times) 
                          for spike_data in recording.spike_trains.values()),
        'cv_firing_rate': np.std(firing_rates) / np.mean(firing_rates) if np.mean(firing_rates) > 0 else 0
    }
    
    return results


def perform_dose_response_analysis(well_results: Dict[str, Dict]) -> Dict:
    """
    Perform dose-response analysis across wells.
    
    Parameters
    ----------
    well_results : Dict[str, Dict]
        Analysis results for each well
        
    Returns
    -------
    Dict
        Dose-response analysis results
    """
    logger.info("Performing dose-response analysis")
    
    # Organize data by condition and concentration
    condition_data = {}
    
    for well_id, results in well_results.items():
        condition = results['condition']
        concentration = results['concentration']
        
        if condition not in condition_data:
            condition_data[condition] = {'concentrations': [], 'responses': []}
        
        condition_data[condition]['concentrations'].append(concentration)
        condition_data[condition]['responses'].append(results['well_summary'])
    
    # Calculate dose-response curves
    dose_response_results = {}
    
    for condition, data in condition_data.items():
        concentrations = np.array(data['concentrations'])
        responses = data['responses']
        
        # Extract different response metrics
        firing_rates = [r['mean_firing_rate'] for r in responses]
        burst_rates = [r['mean_burst_rate'] for r in responses]
        active_fractions = [r['active_electrode_fraction'] for r in responses]
        
        # Sort by concentration
        sort_idx = np.argsort(concentrations)
        concentrations = concentrations[sort_idx]
        firing_rates = np.array(firing_rates)[sort_idx]
        burst_rates = np.array(burst_rates)[sort_idx]
        active_fractions = np.array(active_fractions)[sort_idx]
        
        dose_response_results[condition] = {
            'concentrations': concentrations,
            'firing_rates': firing_rates,
            'burst_rates': burst_rates,
            'active_fractions': active_fractions,
            'n_wells': len(concentrations)
        }
        
        # Calculate EC50/IC50 if applicable (simplified)
        if len(concentrations) >= 3 and np.max(concentrations) > 0:
            # Normalized response (0-1 scale)
            if condition == 'control':
                continue
                
            baseline = np.min(firing_rates)
            max_response = np.max(firing_rates)
            
            if max_response > baseline:
                normalized_response = (firing_rates - baseline) / (max_response - baseline)
                
                # Find concentration closest to 50% response
                ec50_idx = np.argmin(np.abs(normalized_response - 0.5))
                ec50_estimate = concentrations[ec50_idx]
                
                dose_response_results[condition]['ec50_estimate'] = ec50_estimate
    
    return dose_response_results


def create_well_plate_visualizations(analyzer: WellPlateAnalyzer,
                                   well_results: Dict[str, Dict],
                                   dose_response_results: Dict,
                                   output_dir: Path) -> None:
    """
    Create comprehensive well plate analysis visualizations.
    
    Parameters
    ----------
    analyzer : WellPlateAnalyzer
        Well plate analyzer instance
    well_results : Dict[str, Dict]
        Analysis results for each well
    dose_response_results : Dict
        Dose-response analysis results
    output_dir : Path
        Output directory for plots
    """
    logger.info("Creating well plate visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Well plate heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data for heatmaps
    plate_layout = analyzer.well_layout
    firing_rate_plate = np.full((plate_layout['rows'], plate_layout['columns']), np.nan)
    burst_rate_plate = np.full((plate_layout['rows'], plate_layout['columns']), np.nan)
    active_fraction_plate = np.full((plate_layout['rows'], plate_layout['columns']), np.nan)
    condition_plate = np.full((plate_layout['rows'], plate_layout['columns']), '', dtype=object)
    
    # Fill plate matrices
    for well_id, results in well_results.items():
        row_idx, col_idx = analyzer.parse_well_id(well_id)
        summary = results['well_summary']
        
        firing_rate_plate[row_idx, col_idx] = summary['mean_firing_rate']
        burst_rate_plate[row_idx, col_idx] = summary['mean_burst_rate']
        active_fraction_plate[row_idx, col_idx] = summary['active_electrode_fraction']
        condition_plate[row_idx, col_idx] = results['condition']
    
    # Create heatmaps
    row_labels = [chr(ord('A') + i) for i in range(plate_layout['rows'])]
    col_labels = [str(i+1) for i in range(plate_layout['columns'])]
    
    # Firing rate heatmap
    im1 = axes[0, 0].imshow(firing_rate_plate, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Mean Firing Rate (Hz)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    axes[0, 0].set_xticks(range(plate_layout['columns']))
    axes[0, 0].set_yticks(range(plate_layout['rows']))
    axes[0, 0].set_xticklabels(col_labels)
    axes[0, 0].set_yticklabels(row_labels)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Burst rate heatmap
    im2 = axes[0, 1].imshow(burst_rate_plate, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Mean Burst Rate (per min)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    axes[0, 1].set_xticks(range(plate_layout['columns']))
    axes[0, 1].set_yticks(range(plate_layout['rows']))
    axes[0, 1].set_xticklabels(col_labels)
    axes[0, 1].set_yticklabels(row_labels)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Active electrode fraction heatmap
    im3 = axes[1, 0].imshow(active_fraction_plate, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Active Electrode Fraction')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    axes[1, 0].set_xticks(range(plate_layout['columns']))
    axes[1, 0].set_yticks(range(plate_layout['rows']))
    axes[1, 0].set_xticklabels(col_labels)
    axes[1, 0].set_yticklabels(row_labels)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Condition map (categorical)
    unique_conditions = list(set(well_results[well_id]['condition'] 
                               for well_id in well_results.keys()))
    condition_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_conditions)))
    condition_to_num = {cond: i for i, cond in enumerate(unique_conditions)}
    
    condition_num_plate = np.full((plate_layout['rows'], plate_layout['columns']), -1)
    for well_id, results in well_results.items():
        row_idx, col_idx = analyzer.parse_well_id(well_id)
        condition_num_plate[row_idx, col_idx] = condition_to_num[results['condition']]
    
    im4 = axes[1, 1].imshow(condition_num_plate, cmap='Set3', aspect='auto', 
                          vmin=0, vmax=len(unique_conditions)-1)
    axes[1, 1].set_title('Experimental Conditions')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    axes[1, 1].set_xticks(range(plate_layout['columns']))
    axes[1, 1].set_yticks(range(plate_layout['rows']))
    axes[1, 1].set_xticklabels(col_labels)
    axes[1, 1].set_yticklabels(row_labels)
    
    # Add condition legend
    for i, condition in enumerate(unique_conditions):
        axes[1, 1].text(0.02, 0.98 - i*0.08, condition, 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=condition_colors[i], alpha=0.7),
                       verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'well_plate_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dose-response curves
    if dose_response_results:
        n_conditions = len(dose_response_results)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['firing_rates', 'burst_rates', 'active_fractions']
        titles = ['Firing Rate', 'Burst Rate', 'Active Electrode Fraction']
        ylabels = ['Firing Rate (Hz)', 'Burst Rate (per min)', 'Active Fraction']
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_conditions))
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            for j, (condition, results) in enumerate(dose_response_results.items()):
                if condition == 'control':
                    continue  # Skip control for dose-response
                
                concentrations = results['concentrations']
                values = results[metric]
                
                if len(concentrations) > 1:
                    axes[i].semilogx(concentrations, values, 'o-', 
                                   color=colors[j], label=condition, linewidth=2, markersize=6)
            
            axes[i].set_xlabel('Concentration (log scale)')
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f'Dose-Response: {title}')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dose_response_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Statistical comparison by condition
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Organize data by condition for box plots
    conditions = list(set(results['condition'] for results in well_results.values()))
    
    firing_rate_data = {condition: [] for condition in conditions}
    burst_rate_data = {condition: [] for condition in conditions}
    active_fraction_data = {condition: [] for condition in conditions}
    cv_data = {condition: [] for condition in conditions}
    
    for results in well_results.values():
        condition = results['condition']
        summary = results['well_summary']
        
        firing_rate_data[condition].append(summary['mean_firing_rate'])
        burst_rate_data[condition].append(summary['mean_burst_rate'])
        active_fraction_data[condition].append(summary['active_electrode_fraction'])
        cv_data[condition].append(summary['cv_firing_rate'])
    
    # Box plots
    datasets = [firing_rate_data, burst_rate_data, active_fraction_data, cv_data]
    titles = ['Firing Rate by Condition', 'Burst Rate by Condition', 
              'Active Fraction by Condition', 'CV Firing Rate by Condition']
    ylabels = ['Firing Rate (Hz)', 'Burst Rate (per min)', 'Active Fraction', 'CV']
    
    for i, (data, title, ylabel) in enumerate(zip(datasets, titles, ylabels)):
        ax = axes[i // 2, i % 2]
        
        box_data = [data[condition] for condition in conditions]
        box_plot = ax.boxplot(box_data, labels=conditions, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors[:len(conditions)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'condition_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Well plate visualizations saved to {output_dir}")


def generate_plate_analysis_report(analyzer: WellPlateAnalyzer,
                                 well_results: Dict[str, Dict],
                                 dose_response_results: Dict,
                                 output_dir: Path) -> None:
    """
    Generate comprehensive well plate analysis report.
    
    Parameters
    ----------
    analyzer : WellPlateAnalyzer
        Well plate analyzer instance
    well_results : Dict[str, Dict]
        Analysis results for each well
    dose_response_results : Dict
        Dose-response analysis results
    output_dir : Path
        Output directory for the report
    """
    logger.info("Generating well plate analysis report")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detailed CSV export
    detailed_data = []
    for well_id, results in well_results.items():
        base_record = {
            'well_id': well_id,
            'condition': results['condition'],
            'concentration': results['concentration']
        }
        
        # Add well summary metrics
        base_record.update({f"well_{k}": v for k, v in results['well_summary'].items()})
        
        # Add electrode-level data
        for electrode_id, electrode_metrics in results['electrode_metrics'].items():
            record = base_record.copy()
            record['electrode_id'] = electrode_id
            record.update({f"electrode_{k}": v for k, v in electrode_metrics.items()})
            detailed_data.append(record)
    
    # Save detailed data
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(output_dir / 'detailed_well_analysis.csv', index=False)
    
    # Create well summary data
    well_summary_data = []
    for well_id, results in well_results.items():
        record = {
            'well_id': well_id,
            'condition': results['condition'],
            'concentration': results['concentration']
        }
        record.update(results['well_summary'])
        well_summary_data.append(record)
    
    df_summary = pd.DataFrame(well_summary_data)
    df_summary.to_csv(output_dir / 'well_summary.csv', index=False)
    
    # Generate text report
    with open(output_dir / 'plate_analysis_report.txt', 'w') as f:
        f.write("MEA-Flow Well Plate Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Basic information
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Plate Format: {analyzer.plate_format}\n")
        f.write(f"Total Wells Analyzed: {len(well_results)}\n")
        f.write(f"Electrodes per Well: {analyzer.well_layout['electrodes_per_well']}\n\n")
        
        # Condition summary
        conditions = df_summary['condition'].unique()
        f.write("Experimental Conditions:\n")
        f.write("-" * 25 + "\n")
        
        for condition in conditions:
            condition_wells = df_summary[df_summary['condition'] == condition]
            n_wells = len(condition_wells)
            concentrations = sorted(condition_wells['concentration'].unique())
            
            f.write(f"  {condition}: {n_wells} wells\n")
            if len(concentrations) > 1:
                f.write(f"    Concentrations: {concentrations}\n")
            f.write("\n")
        
        # Overall statistics
        f.write("Overall Plate Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"  Mean firing rate: {df_summary['mean_firing_rate'].mean():.2f} ± "
               f"{df_summary['mean_firing_rate'].std():.2f} Hz\n")
        f.write(f"  Mean burst rate: {df_summary['mean_burst_rate'].mean():.2f} ± "
               f"{df_summary['mean_burst_rate'].std():.2f} per min\n")
        f.write(f"  Mean active electrode fraction: "
               f"{df_summary['active_electrode_fraction'].mean():.2f} ± "
               f"{df_summary['active_electrode_fraction'].std():.2f}\n\n")
        
        # Condition-specific statistics
        f.write("Condition-Specific Statistics:\n")
        f.write("-" * 30 + "\n")
        
        condition_stats = df_summary.groupby('condition').agg({
            'mean_firing_rate': ['mean', 'std', 'count'],
            'mean_burst_rate': ['mean', 'std'],
            'active_electrode_fraction': ['mean', 'std']
        }).round(3)
        
        f.write(str(condition_stats))
        f.write("\n\n")
        
        # Dose-response analysis
        if dose_response_results:
            f.write("Dose-Response Analysis:\n")
            f.write("-" * 25 + "\n")
            
            for condition, results in dose_response_results.items():
                if condition == 'control':
                    continue
                
                f.write(f"  {condition}:\n")
                f.write(f"    Concentration range: {np.min(results['concentrations']):.1f} - "
                       f"{np.max(results['concentrations']):.1f}\n")
                f.write(f"    Number of wells: {results['n_wells']}\n")
                
                if 'ec50_estimate' in results:
                    f.write(f"    EC50 estimate: {results['ec50_estimate']:.2f}\n")
                
                f.write(f"    Firing rate range: {np.min(results['firing_rates']):.2f} - "
                       f"{np.max(results['firing_rates']):.2f} Hz\n")
                f.write("\n")
        
        # Quality metrics
        f.write("Data Quality Assessment:\n")
        f.write("-" * 25 + "\n")
        
        total_wells = len(df_summary)
        active_wells = len(df_summary[df_summary['active_electrode_fraction'] > 0.1])
        high_activity_wells = len(df_summary[df_summary['mean_firing_rate'] > 1.0])
        
        f.write(f"  Wells with activity (>10% active electrodes): {active_wells}/{total_wells} "
               f"({100*active_wells/total_wells:.1f}%)\n")
        f.write(f"  Wells with high activity (>1 Hz): {high_activity_wells}/{total_wells} "
               f"({100*high_activity_wells/total_wells:.1f}%)\n")
        
        # CV analysis
        cv_threshold = 1.0
        low_cv_wells = len(df_summary[df_summary['cv_firing_rate'] < cv_threshold])
        f.write(f"  Wells with consistent activity (CV < {cv_threshold}): "
               f"{low_cv_wells}/{total_wells} ({100*low_cv_wells/total_wells:.1f}%)\n")
    
    logger.info(f"Well plate analysis report saved to {output_dir}")


def main():
    """Main workflow for well plate analysis."""
    logger.info("Starting Well Plate Analysis Workflow")
    
    # Create output directory
    output_dir = Path("output/well_plate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Setup well plate analyzer
    logger.info("Step 1: Setting up well plate analyzer")
    analyzer = WellPlateAnalyzer(plate_format="48-well")
    
    # Step 2: Define experimental design
    logger.info("Step 2: Defining experimental design")
    
    # Create a dose-response experiment design
    experimental_design = {}
    
    # Control wells (A01-A08)
    for col in range(8):
        well_id = analyzer.get_well_id(0, col)
        experimental_design[well_id] = {'condition': 'control', 'concentration': 0}
    
    # Drug A dose response (B01-B08)
    drug_a_concentrations = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    for col, conc in enumerate(drug_a_concentrations):
        well_id = analyzer.get_well_id(1, col)
        experimental_design[well_id] = {'condition': 'drug_A', 'concentration': conc}
    
    # Drug B dose response (C01-C08)
    drug_b_concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    for col, conc in enumerate(drug_b_concentrations):
        well_id = analyzer.get_well_id(2, col)
        experimental_design[well_id] = {'condition': 'drug_B', 'concentration': conc}
    
    # Stimulant conditions (D01-D04)
    stimulant_concentrations = [1.0, 3.0, 10.0, 30.0]
    for col, conc in enumerate(stimulant_concentrations):
        well_id = analyzer.get_well_id(3, col)
        experimental_design[well_id] = {'condition': 'stimulant', 'concentration': conc}
    
    # Inhibitor conditions (D05-D08)
    inhibitor_concentrations = [0.1, 0.3, 1.0, 3.0]
    for col, conc in enumerate(inhibitor_concentrations):
        well_id = analyzer.get_well_id(3, col + 4)
        experimental_design[well_id] = {'condition': 'inhibitor', 'concentration': conc}
    
    logger.info(f"Experimental design includes {len(experimental_design)} wells")
    
    # Step 3: Generate synthetic data
    logger.info("Step 3: Generating synthetic MEA data for all wells")
    well_recordings = create_well_plate_data(
        analyzer=analyzer,
        experimental_design=experimental_design,
        duration=300.0,  # 5 minutes per well
        seed=42
    )
    
    # Step 4: Analyze each well
    logger.info("Step 4: Analyzing neural activity for each well")
    well_results = {}
    
    for well_id, recording in well_recordings.items():
        well_results[well_id] = analyze_well_activity(recording)
    
    # Step 5: Perform dose-response analysis
    logger.info("Step 5: Performing dose-response analysis")
    dose_response_results = perform_dose_response_analysis(well_results)
    
    # Step 6: Create visualizations
    logger.info("Step 6: Creating well plate visualizations")
    create_well_plate_visualizations(analyzer, well_results, dose_response_results, output_dir)
    
    # Step 7: Generate comprehensive report
    logger.info("Step 7: Generating comprehensive analysis report")
    generate_plate_analysis_report(analyzer, well_results, dose_response_results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("WELL PLATE ANALYSIS WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")
    
    print(f"\nPlate configuration:")
    print(f"  - Format: {analyzer.plate_format}")
    print(f"  - Wells analyzed: {len(well_results)}")
    print(f"  - Electrodes per well: {analyzer.well_layout['electrodes_per_well']}")
    
    print(f"\nExperimental conditions:")
    conditions = set(results['condition'] for results in well_results.values())
    for condition in sorted(conditions):
        condition_wells = [w for w, r in well_results.items() if r['condition'] == condition]
        print(f"  - {condition}: {len(condition_wells)} wells")
    
    print(f"\nOverall statistics:")
    all_firing_rates = [r['well_summary']['mean_firing_rate'] for r in well_results.values()]
    all_active_fractions = [r['well_summary']['active_electrode_fraction'] for r in well_results.values()]
    
    print(f"  - Mean firing rate: {np.mean(all_firing_rates):.2f} ± {np.std(all_firing_rates):.2f} Hz")
    print(f"  - Mean active electrode fraction: {np.mean(all_active_fractions):.2f} ± {np.std(all_active_fractions):.2f}")
    
    if dose_response_results:
        print(f"\nDose-response analysis:")
        for condition in dose_response_results:
            if condition != 'control':
                n_wells = dose_response_results[condition]['n_wells']
                print(f"  - {condition}: {n_wells} concentration points")
    
    print("\nWell plate analysis workflow completed successfully!")


if __name__ == "__main__":
    main()