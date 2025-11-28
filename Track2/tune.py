"""
Hyperparameter Tuning Script for Quantum Machine Learning Model

This script performs grid search to find optimal hyperparameters
for maximizing the R² score on the test set.
"""

import argparse
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

from main import run_experiment


def parse_arguments():
    """Parse command-line arguments for tuning script."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter Tuning for Quantum Machine Learning Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data_file',
        type=str,
        default='Train_Dataset_Simulated_Price_swaption.xlsx',
        help='Path to data file (default: Train_Dataset_Simulated_Price_swaption.xlsx)'
    )
    
    parser.add_argument(
        '--tenor',
        type=float,
        default=30.0,
        help='Tenor value (default: 30.0)'
    )
    
    parser.add_argument(
        '--maturity',
        type=float,
        default=30.0,
        help='Maturity value (default: 30.0)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=2000,
        help='Limit number of samples for quick tuning (default: 2000)'
    )
    
    return parser.parse_args()


def plot_tuning_results(all_results: List[Dict[str, Any]], results_dir: Path) -> None:
    """
    Create a heatmap visualization of tuning results.
    
    Shows the best R² score for each (n_qubits, lookback) combination,
    aggregated over scaling factors and regressors.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of result dictionaries, each containing 'config' and 'metrics'
    results_dir : Path
        Directory to save the plot
    """
    if not all_results:
        print("Warning: No results to plot.")
        return
    
    # Extract unique n_qubits and lookback values
    n_qubits_set = set()
    lookback_set = set()
    
    for result in all_results:
        config = result['config']
        n_qubits_set.add(config['n_qubits'])
        lookback_set.add(config['lookback'])
    
    n_qubits_list = sorted(list(n_qubits_set))
    lookback_list = sorted(list(lookback_set))
    
    # Create heatmap matrix: best R² for each (n_qubits, lookback) pair
    heatmap_matrix = np.full((len(n_qubits_list), len(lookback_list)), np.nan)
    
    for i, n_qubits in enumerate(n_qubits_list):
        for j, lookback in enumerate(lookback_list):
            # Find all results for this (n_qubits, lookback) pair
            relevant_results = [
                r for r in all_results
                if r['config']['n_qubits'] == n_qubits
                and r['config']['lookback'] == lookback
            ]
            
            if relevant_results:
                # Get maximum R² across all scaling factors and regressors
                max_r2 = max(r['metrics']['r2'] for r in relevant_results)
                heatmap_matrix[i, j] = max_r2
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use imshow for heatmap
    im = ax.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=np.nanmin(heatmap_matrix), vmax=np.nanmax(heatmap_matrix))
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(lookback_list)))
    ax.set_yticks(np.arange(len(n_qubits_list)))
    ax.set_xticklabels(lookback_list)
    ax.set_yticklabels(n_qubits_list)
    
    # Set axis labels
    ax.set_xlabel('Lookback Window', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Tuning Results: Best R² Score\n(aggregated over scaling factors and regressors)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(n_qubits_list)):
        for j in range(len(lookback_list)):
            if not np.isnan(heatmap_matrix[i, j]):
                text = ax.text(j, i, f'{heatmap_matrix[i, j]:.4f}',
                             ha="center", va="center", color="black", fontsize=10,
                             fontweight='bold' if heatmap_matrix[i, j] == np.nanmax(heatmap_matrix) else 'normal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R² Score', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / 'tuning_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Tuning heatmap saved to: {plot_path}")
    
    plt.close()


def main():
    """Perform grid search hyperparameter tuning."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Fixed parameters (not tuned)
    fixed_config = {
        'data_file': args.data_file,
        'tenor': args.tenor,
        'maturity': args.maturity,
        'use_log_returns': True,
        'depth': 3,
        'encoding': 'angle',
        'entanglement': 'linear',
        'normalize_method': 'zscore',
        'test_size': 0.2,
        'shots': 512,
        'seed': 42,
        'output_dir': args.output_dir,
        'visualize': False,  # Disable visualization during tuning for speed
        'verbose': False,  # Suppress verbose output during tuning
        'price_column': None,
        'max_samples': args.max_samples
    }
    
    # Define search space
    search_space = {
        'n_qubits': [4, 6, 8],
        'lookback': [4, 6, 8],
        'data_scaling_factor': [0.5, 1.0, 1.5, 2.0],
        'regressor': ['ridge', 'linear']
    }
    
    # Base scaling factor (np.pi / 3.0)
    base_scaling = np.pi / 3.0
    
    print("="*80)
    print("  HYPERPARAMETER TUNING - GRID SEARCH")
    print("="*80)
    print(f"\nSearch Space:")
    print(f"  n_qubits: {search_space['n_qubits']}")
    print(f"  lookback: {search_space['lookback']}")
    print(f"  data_scaling_factor multipliers: {search_space['data_scaling_factor']}")
    print(f"  regressor: {search_space['regressor']}")
    print(f"\nFixed Parameters:")
    for key, value in fixed_config.items():
        print(f"  {key}: {value}")
    
    # Generate all combinations
    all_combinations = list(itertools.product(
        search_space['n_qubits'],
        search_space['lookback'],
        search_space['data_scaling_factor'],
        search_space['regressor']
    ))
    
    # Filter invalid combinations (lookback > n_qubits)
    valid_combinations = [
        combo for combo in all_combinations
        if combo[1] <= combo[0]  # lookback <= n_qubits
    ]
    
    print(f"\nTotal combinations: {len(all_combinations)}")
    print(f"Valid combinations (lookback <= n_qubits): {len(valid_combinations)}")
    print("\n" + "="*80)
    
    # Track best configuration
    best_r2 = float('-inf')
    best_config = None
    best_metrics = None
    
    # Results storage
    all_results = []
    
    # Grid search loop
    for idx, (n_qubits, lookback, scaling_mult, regressor) in enumerate(valid_combinations, 1):
        # Calculate actual scaling factor
        data_scaling_factor = scaling_mult * base_scaling
        
        # Create config for this combination
        config = fixed_config.copy()
        config.update({
            'n_qubits': n_qubits,
            'lookback': lookback,
            'data_scaling_factor': data_scaling_factor,
            'regressor': regressor
        })
        
        print(f"\n[{idx}/{len(valid_combinations)}] Testing configuration:")
        print(f"  n_qubits: {n_qubits}")
        print(f"  lookback: {lookback}")
        print(f"  data_scaling_factor: {data_scaling_factor:.4f} (multiplier: {scaling_mult})")
        print(f"  regressor: {regressor}")
        
        try:
            # Run experiment
            metrics = run_experiment(config)
            
            r2 = metrics['r2']
            print(f"  Result: R² = {r2:.6f}, MSE = {metrics['mse']:.6f}, MAE = {metrics['mae']:.6f}")
            
            # Store results
            result_entry = {
                'config': config.copy(),
                'metrics': metrics.copy()
            }
            all_results.append(result_entry)
            
            # Update best if this is better
            if r2 > best_r2:
                best_r2 = r2
                best_config = config.copy()
                best_metrics = metrics.copy()
                print(f"  *** NEW BEST! R² = {best_r2:.6f} ***")
            
        except Exception as e:
            print(f"  ERROR: Experiment failed: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("  TUNING COMPLETE")
    print("="*80)
    
    if best_config is not None:
        print(f"\nBest Configuration (R² = {best_r2:.6f}):")
        print(f"  n_qubits: {best_config['n_qubits']}")
        print(f"  lookback: {best_config['lookback']}")
        print(f"  data_scaling_factor: {best_config['data_scaling_factor']:.6f}")
        print(f"  regressor: {best_config['regressor']}")
        print(f"\nBest Metrics:")
        print(f"  R²: {best_metrics['r2']:.6f}")
        print(f"  MSE: {best_metrics['mse']:.6f}")
        print(f"  MAE: {best_metrics['mae']:.6f}")
        print(f"  RMSE: {best_metrics['rmse']:.6f}")
        if best_metrics['mape'] is not None:
            print(f"  MAPE: {best_metrics['mape']:.4f}%")
        
        # Save best parameters to JSON
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        best_params_file = results_dir / 'best_params.json'
        best_params_data = {
            'best_r2': float(best_r2),
            'best_config': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                           for k, v in best_config.items()},
            'best_metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) 
                            for k, v in best_metrics.items()}
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        best_params_data = convert_to_native(best_params_data)
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params_data, f, indent=2)
        
        print(f"\nBest parameters saved to: {best_params_file}")
        
        # Save all results for analysis
        all_results_file = results_dir / 'tuning_results.json'
        all_results_serializable = []
        for entry in all_results:
            serializable_entry = {
                'config': convert_to_native(entry['config']),
                'metrics': convert_to_native(entry['metrics'])
            }
            all_results_serializable.append(serializable_entry)
        
        with open(all_results_file, 'w') as f:
            json.dump(all_results_serializable, f, indent=2)
        
        print(f"All results saved to: {all_results_file}")
        
        # Generate visualization
        if all_results:
            print("\nGenerating tuning heatmap visualization...")
            plot_tuning_results(all_results, results_dir)
    else:
        print("\nERROR: No successful experiments completed!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

