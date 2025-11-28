"""
Script to run main.py with the best parameters from best_params.json
"""

import json
from pathlib import Path
from main import run_experiment

def main():
    # Load best parameters
    results_dir = Path(__file__).parent / 'results'
    best_params_path = results_dir / 'best_params.json'
    
    if not best_params_path.exists():
        print(f"Error: {best_params_path} not found!")
        return
    
    with open(best_params_path, 'r') as f:
        best_data = json.load(f)
    
    config = best_data['best_config'].copy()
    
    # Override visualize and verbose to True for full output
    config['visualize'] = True
    config['verbose'] = True
    
    print("="*80)
    print("  RUNNING WITH BEST PARAMETERS FROM TUNING")
    print("="*80)
    print(f"\nBest R² Score: {best_data['best_r2']:.6f}")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "="*80 + "\n")
    
    # Run experiment
    metrics = run_experiment(config)
    
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)
    print(f"  Test R²: {metrics['r2']:.6f}")
    print(f"  Test MSE: {metrics['mse']:.6f}")
    print(f"  Test MAE: {metrics['mae']:.6f}")
    print(f"  Test RMSE: {metrics['rmse']:.6f}")
    if metrics.get('mape') is not None:
        print(f"  Test MAPE: {metrics['mape']:.4f}%")
    print("="*80)

if __name__ == "__main__":
    main()

