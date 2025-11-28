"""
Main Script for Quantum Machine Learning Option Price Prediction

This script implements a complete QML pipeline using Quantum Reservoir Computing
to predict option prices from historical data.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import warnings

from src.data_loader import DataLoader
from src.quantum_reservoir import QuantumReservoir
from src.model import HybridQMLModel


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Quantum Machine Learning for Option Price Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data_file Train_Dataset_Simulated_Price_swaption.xlsx
  python main.py --data_file data.csv --n_qubits 6 --depth 4 --lookback 15
  python main.py --data_file data.xlsx --regressor mlp --test_size 0.3
        """
    )
    
    # Data parameters
    parser.add_argument(
        '--data_file',
        type=str,
        default='Train_Dataset_Simulated_Price_swaption.xlsx',
        help='Path to data file (Excel or CSV)'
    )
    
    parser.add_argument(
        '--price_column',
        type=str,
        default=None,
        help='Name of the price column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--tenor',
        type=float,
        default=None,
        help='Tenor value to filter by (for Swaption data)'
    )
    
    parser.add_argument(
        '--maturity',
        type=float,
        default=None,
        help='Maturity value to filter by (for Swaption data)'
    )
    
    parser.add_argument(
        '--use_log_returns',
        action='store_true',
        help='Use log returns instead of raw prices (default: True, recommended for stationarity)'
    )
    
    parser.add_argument(
        '--no_log_returns',
        dest='use_log_returns',
        action='store_false',
        help='Disable log returns (use raw prices)'
    )
    
    # Set default to True for use_log_returns
    parser.set_defaults(use_log_returns=True)
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=4,
        help='Lookback window size for time-series (default: 4, aligned with n_qubits)'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    
    # Quantum reservoir parameters
    parser.add_argument(
        '--n_qubits',
        type=int,
        default=4,
        help='Number of qubits in reservoir (default: 4)'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='Depth of reservoir circuit (default: 3)'
    )
    
    parser.add_argument(
        '--encoding',
        type=str,
        default='angle',
        choices=['angle', 'amplitude'],
        help='Encoding type: angle or amplitude (default: angle)'
    )
    
    parser.add_argument(
        '--entanglement',
        type=str,
        default='linear',
        choices=['linear', 'circular', 'full'],
        help='Entanglement pattern: linear, circular, or full (default: linear)'
    )
    
    parser.add_argument(
        '--shots',
        type=int,
        default=1024,
        help='Number of measurement shots (default: 1024)'
    )
    
    # Classical regressor parameters
    parser.add_argument(
        '--regressor',
        type=str,
        default='ridge',
        choices=['linear', 'ridge', 'mlp'],
        help='Type of classical regressor: linear, ridge, or mlp (default: ridge)'
    )
    
    # Normalization
    parser.add_argument(
        '--normalize_method',
        type=str,
        default='zscore',
        choices=['minmax', 'zscore'],
        help='Normalization method (default: zscore, better for log returns)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save the trained model'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def create_output_dir(output_dir: Optional[str]) -> Path:
    """Create output directory if it doesn't exist."""
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / 'results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[Path] = None
) -> None:
    """Plot predictions against actual values."""
    plt.figure(figsize=(12, 6))
    
    # Plot predictions and actual values
    n_samples = len(y_true)
    x = np.arange(n_samples)
    
    plt.plot(x, y_true, 'b-', label='Actual', alpha=0.7, linewidth=2)
    plt.plot(x, y_pred, 'r--', label='Predicted', alpha=0.7, linewidth=2)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Price (Normalized)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None
) -> None:
    """Plot residual errors."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals over time
    axes[0].plot(residuals, 'o', alpha=0.6, markersize=4)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Sample Index', fontsize=11)
    axes[0].set_ylabel('Residual Error', fontsize=11)
    axes[0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual Error', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MAPE value (percentage)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero
    mask = np.abs(y_true) > 1e-8
    if np.sum(mask) == 0:
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def print_results_summary(
    train_metrics: dict,
    test_metrics: dict,
    model_info: dict
) -> None:
    """Print a formatted summary of results."""
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    
    print("\nModel Configuration:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        if metric == 'mape':
            print(f"  {metric.upper()}: {value:.4f}%")
        else:
            print(f"  {metric.upper()}: {value:.6f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        if metric == 'mape':
            print(f"  {metric.upper()}: {value:.4f}%")
        else:
            print(f"  {metric.upper()}: {value:.6f}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    print("\n" + "="*80)
    print("  QUANTUM MACHINE LEARNING FOR OPTION PRICE PREDICTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data_file}")
    if args.tenor is not None:
        print(f"  Tenor: {args.tenor}")
    if args.maturity is not None:
        print(f"  Maturity: {args.maturity}")
    print(f"  Use log returns: {args.use_log_returns}")
    print(f"  Qubits: {args.n_qubits}")
    print(f"  Circuit depth: {args.depth}")
    print(f"  Encoding: {args.encoding}")
    print(f"  Entanglement: {args.entanglement}")
    print(f"  Lookback window: {args.lookback}")
    print(f"  Regressor: {args.regressor}")
    print(f"  Test size: {args.test_size}")
    
    # ========================================================================
    # DATA LOADING AND PREPROCESSING
    # ========================================================================
    print("\n" + "-"*80)
    print("  STEP 1: Loading and Preprocessing Data")
    print("-"*80)
    
    data_loader = DataLoader(
        normalize_method=args.normalize_method,
        lookback_window=args.lookback,
        test_size=args.test_size,
        random_seed=args.seed
    )
    
    # Get data file path
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = Path(__file__).parent / data_file
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from: {data_file}")
    
    # Check for available Tenor/Maturity pairs (Swaption format)
    try:
        available_pairs = data_loader.get_available_pairs(data_file)
        if len(available_pairs) > 0:
            print(f"\nAvailable (Tenor, Maturity) pairs in dataset: {len(available_pairs)}")
            for i, (t, m) in enumerate(available_pairs[:10]):  # Show first 10
                print(f"  {i+1}. Tenor={t}, Maturity={m}")
            if len(available_pairs) > 10:
                print(f"  ... and {len(available_pairs) - 10} more pairs")
            
            # Validate requested pair
            if args.tenor is not None and args.maturity is not None:
                matching_pair = None
                for (t, m) in available_pairs:
                    if (abs(t - args.tenor) < 1e-6 and abs(m - args.maturity) < 1e-6):
                        matching_pair = (t, m)
                        break
                
                if matching_pair is None:
                    print(f"\nWARNING: Requested pair (Tenor={args.tenor}, Maturity={args.maturity}) not found!")
                    print("Available pairs:")
                    for (t, m) in available_pairs:
                        print(f"  - Tenor={t}, Maturity={m}")
                    raise ValueError(
                        f"Tenor={args.tenor}, Maturity={args.maturity} not found in dataset"
                    )
                else:
                    print(f"\nUsing pair: Tenor={matching_pair[0]}, Maturity={matching_pair[1]}")
            elif args.tenor is not None or args.maturity is not None:
                print("\nWARNING: Both --tenor and --maturity must be specified for Swaption data")
                print("Available pairs:")
                for (t, m) in available_pairs:
                    print(f"  - Tenor={t}, Maturity={m}")
    except Exception as e:
        # Not Swaption format or error parsing, continue with standard format
        if "Swaption" in str(e) or "parse" in str(e).lower():
            warnings.warn(f"Could not parse Swaption format: {e}", UserWarning)
        pass
    
    if args.tenor is not None or args.maturity is not None:
        print(f"Filtering for Tenor={args.tenor}, Maturity={args.maturity}")
    if args.use_log_returns:
        print("Using log returns instead of raw prices")
    
    result = data_loader.prepare_data(
        data_file,
        price_column=args.price_column,
        tenor=args.tenor,
        maturity=args.maturity,
        use_log_returns=args.use_log_returns
    )
    X_train, X_test, y_train, y_test, test_initial_prices = result
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape}")
    
    # ========================================================================
    # QUANTUM RESERVOIR SETUP
    # ========================================================================
    print("\n" + "-"*80)
    print("  STEP 2: Building Quantum Reservoir")
    print("-"*80)
    
    quantum_reservoir = QuantumReservoir(
        n_qubits=args.n_qubits,
        depth=args.depth,
        encoding_type=args.encoding,
        entanglement_pattern=args.entanglement,
        random_seed=args.seed,
        shots=args.shots
    )
    
    print(f"Reservoir circuit depth: {quantum_reservoir.get_circuit_depth()}")
    
    # Warn if lookback > n_qubits with angle encoding (older history will be truncated)
    if args.encoding == 'angle' and args.lookback > args.n_qubits:
        warnings.warn(
            f"Lookback window ({args.lookback}) > number of qubits ({args.n_qubits}). "
            f"With angle encoding, only the last {args.n_qubits} values will be used. "
            f"Older history will be truncated. Consider setting lookback={args.n_qubits} "
            f"for 1-to-1 mapping without information loss.",
            UserWarning
        )
    
    # Visualize circuit if requested
    if args.visualize:
        circuit_path = output_dir / 'quantum_reservoir_circuit.png'
        try:
            quantum_reservoir.visualize_circuit(str(circuit_path))
        except Exception as e:
            print(f"Warning: Could not visualize circuit: {e}")
    
    # ========================================================================
    # MODEL TRAINING
    # ========================================================================
    print("\n" + "-"*80)
    print("  STEP 3: Training Hybrid QML Model")
    print("-"*80)
    
    model = HybridQMLModel(
        quantum_reservoir=quantum_reservoir,
        regressor_type=args.regressor
    )
    
    model.fit(X_train, y_train, verbose=True)
    
    # ========================================================================
    # GENERATE PREDICTIONS
    # ========================================================================
    # Generate predictions first (needed for price reconstruction)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # ========================================================================
    # PRICE RECONSTRUCTION (if using log returns)
    # ========================================================================
    y_train_pred_prices = None
    y_train_actual_prices = None
    y_test_pred_prices = None
    y_test_actual_prices = None
    
    if args.use_log_returns and test_initial_prices is not None:
        print("\n" + "-"*80)
        print("  STEP 4.5: Reconstructing Prices from Log Returns")
        print("-"*80)
        
        # Denormalize log returns
        y_test_pred_denorm = data_loader.denormalize(y_test_pred)
        y_test_denorm = data_loader.denormalize(y_test)
        y_train_pred_denorm = data_loader.denormalize(y_train_pred)
        y_train_denorm = data_loader.denormalize(y_train)
        
        # Reconstruct prices for test set
        # Formula: Price_t = Price_{t-1} * exp(log_return_t)
        y_test_pred_prices = np.zeros(len(y_test_pred))
        y_test_actual_prices = np.zeros(len(y_test))
        
        # First prediction uses test_initial_prices[0] (price before first test prediction)
        if test_initial_prices is not None and len(test_initial_prices) > 0:
            initial_price = float(test_initial_prices[0])
            y_test_pred_prices[0] = initial_price * np.exp(y_test_pred_denorm[0])
            y_test_actual_prices[0] = initial_price * np.exp(y_test_denorm[0])
            
            # Subsequent predictions use previous reconstructed price
            for i in range(1, len(y_test_pred)):
                y_test_pred_prices[i] = y_test_pred_prices[i-1] * np.exp(y_test_pred_denorm[i])
                y_test_actual_prices[i] = y_test_actual_prices[i-1] * np.exp(y_test_denorm[i])
        
        # For training set, we need to reconstruct from the beginning
        # We'll use a similar approach but need the initial price for training set
        # For now, we'll skip training price reconstruction or use a placeholder
        # (Training metrics on log-returns are still meaningful)
        print(f"Reconstructed {len(y_test_pred_prices)} test prices")
        print(f"  First predicted price: {y_test_pred_prices[0]:.4f}")
        print(f"  First actual price: {y_test_actual_prices[0]:.4f}")
    
    # ========================================================================
    # EVALUATION
    # ========================================================================
    print("\n" + "-"*80)
    print("  STEP 4: Evaluating Model")
    print("-"*80)
    
    # Training metrics (on log-returns)
    train_metrics = model.evaluate(X_train, y_train, metrics=['mse', 'mae', 'r2', 'rmse'])
    
    # Test metrics (on log-returns)
    test_metrics = model.evaluate(X_test, y_test, metrics=['mse', 'mae', 'r2', 'rmse'])
    
    # Calculate MAPE and MSE on reconstructed prices if available
    if args.use_log_returns and y_test_pred_prices is not None and y_test_actual_prices is not None:
        # Calculate MAPE on reconstructed prices (meaningful metric)
        test_mape_prices = calculate_mape(y_test_actual_prices, y_test_pred_prices)
        test_metrics['mape'] = test_mape_prices
        
        # Calculate MSE on reconstructed prices
        test_mse_prices = np.mean((y_test_actual_prices - y_test_pred_prices) ** 2)
        test_metrics['mse_prices'] = test_mse_prices
        test_metrics['rmse_prices'] = np.sqrt(test_mse_prices)
        
        # Keep log-return metrics for reference
        test_metrics['mse_log_returns'] = test_metrics['mse']
        test_metrics['r2_log_returns'] = test_metrics['r2']
        
        print("\nTest Metrics (on Reconstructed Prices):")
        print(f"  MSE: {test_mse_prices:.6f}")
        print(f"  RMSE: {np.sqrt(test_mse_prices):.6f}")
        print(f"  MAPE: {test_mape_prices:.4f}%")
        print("\nTest Metrics (on Log Returns - for reference):")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  R²: {test_metrics['r2']:.6f}")
    else:
        # Fallback: calculate MAPE on log-returns (less meaningful but better than nothing)
        train_mape = calculate_mape(y_train, model.predict(X_train))
        train_metrics['mape'] = train_mape
        test_mape = calculate_mape(y_test, model.predict(X_test))
        test_metrics['mape'] = test_mape
        
        print("\nTraining Metrics (on Log Returns):")
        for metric, value in train_metrics.items():
            if metric == 'mape':
                print(f"  {metric.upper()}: {value:.4f}%")
            else:
                print(f"  {metric.upper()}: {value:.6f}")
        
        print("\nTest Metrics (on Log Returns):")
        for metric, value in test_metrics.items():
            if metric == 'mape':
                print(f"  {metric.upper()}: {value:.4f}%")
            else:
                print(f"  {metric.upper()}: {value:.6f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    if args.visualize:
        print("\n" + "-"*80)
        print("  STEP 5: Generating Visualizations")
        print("-"*80)
        
        # Plot reconstructed prices if available, otherwise plot log-returns
        if args.use_log_returns and y_test_pred_prices is not None and y_test_actual_prices is not None:
            # Plot reconstructed prices (real currency values)
            plot_predictions(
                y_test_actual_prices, y_test_pred_prices,
                title="Test Set: Predicted vs Actual Prices (Reconstructed)",
                save_path=output_dir / 'test_predictions.png'
            )
            
            # Main prediction plot (test set) - saved as prediction_plot.png
            plot_predictions(
                y_test_actual_prices, y_test_pred_prices,
                title="Predicted vs Actual Prices",
                save_path=output_dir / 'prediction_plot.png'
            )
            
            # Residuals on reconstructed prices
            plot_residuals(
                y_test_actual_prices, y_test_pred_prices,
                save_path=output_dir / 'residuals.png'
            )
            
            # Also plot log-returns for reference
            plot_predictions(
                y_test, y_test_pred,
                title="Training Set: Predictions vs Actual (Log Returns)",
                save_path=output_dir / 'train_predictions.png'
            )
        else:
            # Fallback: plot log-returns
            plot_predictions(
                y_train, y_train_pred,
                title="Training Set: Predictions vs Actual",
                save_path=output_dir / 'train_predictions.png'
            )
            
            plot_predictions(
                y_test, y_test_pred,
                title="Test Set: Predictions vs Actual",
                save_path=output_dir / 'test_predictions.png'
            )
            
            plot_predictions(
                y_test, y_test_pred,
                title="Predicted vs Actual Prices",
                save_path=output_dir / 'prediction_plot.png'
            )
            
            plot_residuals(
                y_test, y_test_pred,
                save_path=output_dir / 'residuals.png'
            )
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    model_info = {
        'n_qubits': args.n_qubits,
        'circuit_depth': args.depth,
        'encoding': args.encoding,
        'entanglement': args.entanglement,
        'lookback_window': args.lookback,
        'regressor': args.regressor,
        'use_log_returns': args.use_log_returns,
        'tenor': args.tenor,
        'maturity': args.maturity,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    print_results_summary(train_metrics, test_metrics, model_info)
    
    # Save results to file
    results_file = output_dir / 'results_summary.txt'
    with open(results_file, 'w') as f:
        f.write("QUANTUM MACHINE LEARNING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("Model Configuration:\n")
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTraining Metrics:\n")
        for metric, value in train_metrics.items():
            if metric == 'mape':
                f.write(f"  {metric.upper()}: {value:.4f}%\n")
            else:
                f.write(f"  {metric.upper()}: {value:.6f}\n")
        f.write("\nTest Metrics:\n")
        for metric, value in test_metrics.items():
            if metric == 'mape':
                f.write(f"  {metric.upper()}: {value:.4f}%\n")
            else:
                f.write(f"  {metric.upper()}: {value:.6f}\n")
    
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

