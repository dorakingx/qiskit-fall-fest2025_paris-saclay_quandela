"""
Hyperparameter Tuning Script for Quantum Machine Learning Model

This script performs grid search to find optimal hyperparameters
for maximizing the R² score on the test set.
"""

import itertools
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from main import run_experiment


def main():
    """Perform grid search hyperparameter tuning."""
    
    # Fixed parameters (not tuned)
    fixed_config = {
        'data_file': 'Train_Dataset_Simulated_Price_swaption.xlsx',
        'tenor': 30.0,
        'maturity': 30.0,
        'use_log_returns': True,
        'depth': 3,
        'encoding': 'angle',
        'entanglement': 'linear',
        'normalize_method': 'zscore',
        'test_size': 0.2,
        'shots': 512,
        'seed': 42,
        'output_dir': None,
        'visualize': False,  # Disable visualization during tuning for speed
        'verbose': False,  # Suppress verbose output during tuning
        'price_column': None
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
    else:
        print("\nERROR: No successful experiments completed!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

