"""
Main Script for Quantum Option Pricing Comparison

This script runs classical benchmarks and quantum simulations to price
European Call and Put options, then compares the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple
from classical_benchmark import (
    black_scholes_call_analytic,
    black_scholes_put_analytic,
    binomial_tree_call,
    binomial_tree_put
)
from quantum_random_walk import QuantumRandomWalk


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_results_header(title: str):
    """Print a formatted header for results section."""
    print_separator()
    print(f"  {title}")
    print_separator()


def format_price(price: float) -> str:
    """Format price with appropriate precision."""
    return f"${price:.6f}"


def calculate_error(quantum_price: float, classical_price: float) -> Tuple[float, float]:
    """
    Calculate absolute and relative errors.
    
    Parameters:
    -----------
    quantum_price : float
        Price from quantum simulation
    classical_price : float
        Price from classical method
    
    Returns:
    --------
    Tuple[float, float]
        (absolute_error, relative_error_percent)
    """
    abs_error = abs(quantum_price - classical_price)
    rel_error = (abs_error / classical_price * 100) if classical_price != 0 else 0.0
    return abs_error, rel_error


def main():
    """
    Main function to run option pricing comparisons.
    """
    # Default parameters
    S0 = 100.0  # Spot price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    T = 1.0     # Time to maturity (1 year)
    N = 5       # Number of time steps (and qubits)
    shots = 10000  # Number of measurement shots for quantum simulation
    
    print("\n" + "="*80)
    print("  QUANTUM OPTION PRICING USING QUANTUM RANDOM WALK")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Spot Price (S0):     ${S0:.2f}")
    print(f"  Strike Price (K):    ${K:.2f}")
    print(f"  Risk-free Rate (r):  {r*100:.2f}%")
    print(f"  Volatility (σ):       {sigma*100:.2f}%")
    print(f"  Time to Maturity (T): {T:.2f} years")
    print(f"  Number of Steps (N):  {N}")
    print(f"  Quantum Shots:       {shots:,}")
    
    # ========================================================================
    # CLASSICAL BENCHMARKS
    # ========================================================================
    print_results_header("CLASSICAL BENCHMARKS")
    
    # Analytical Black-Scholes
    print("\n1. Analytical Black-Scholes Formula:")
    bs_call = black_scholes_call_analytic(S0, K, r, sigma, T)
    bs_put = black_scholes_put_analytic(S0, K, r, sigma, T)
    print(f"   Call Option Price: {format_price(bs_call)}")
    print(f"   Put Option Price:  {format_price(bs_put)}")
    
    # Binomial Tree
    print("\n2. Binomial Tree Method:")
    bt_call_result = binomial_tree_call(S0, K, r, sigma, T, N)
    bt_put_result = binomial_tree_put(S0, K, r, sigma, T, N)
    bt_call = bt_call_result['price']
    bt_put = bt_put_result['price']
    print(f"   Call Option Price: {format_price(bt_call)}")
    print(f"   Put Option Price:  {format_price(bt_put)}")
    print(f"   Risk-neutral prob (p): {bt_call_result['parameters']['p']:.6f}")
    print(f"   Up factor (u):         {bt_call_result['parameters']['u']:.6f}")
    print(f"   Down factor (d):       {bt_call_result['parameters']['d']:.6f}")
    
    # ========================================================================
    # QUANTUM SIMULATION
    # ========================================================================
    print_results_header("QUANTUM RANDOM WALK SIMULATION")
    
    # Initialize quantum random walk
    print("\nBuilding quantum circuit...")
    qrw = QuantumRandomWalk(S0, K, r, sigma, T, N)
    circuit = qrw.get_circuit()
    
    # Display circuit
    print("\nQuantum Circuit Structure:")
    print(circuit.draw(output='text'))
    
    # Get circuit metrics
    metrics = qrw._get_circuit_metrics()
    print(f"\nCircuit Metrics:")
    print(f"  Number of Qubits:  {metrics['num_qubits']}")
    print(f"  Circuit Depth:     {metrics['circuit_depth']}")
    print(f"  Number of Gates:   {metrics['num_gates']}")
    
    # Price Call option using quantum simulation
    print(f"\nRunning quantum simulation for Call option ({shots:,} shots)...")
    qrw_call_result = qrw.price_option('call', shots=shots, use_statevector=False)
    qrw_call = qrw_call_result['price']
    
    print(f"   Quantum Call Price: {format_price(qrw_call)}")
    print(f"   Expected Payoff:    ${qrw_call_result['expected_payoff']:.6f}")
    
    # Price Put option using quantum simulation
    print(f"\nRunning quantum simulation for Put option ({shots:,} shots)...")
    qrw_put_result = qrw.price_option('put', shots=shots, use_statevector=False)
    qrw_put = qrw_put_result['price']
    
    print(f"   Quantum Put Price:  {format_price(qrw_put)}")
    print(f"   Expected Payoff:    ${qrw_put_result['expected_payoff']:.6f}")
    
    # Show sample measurement results
    print(f"\nSample Measurement Results (Call):")
    sample_counts = dict(list(qrw_call_result['measurement_counts'].items())[:5])
    for measurement, count in sample_counts.items():
        stock_price = qrw._state_to_stock_price(measurement)
        payoff = qrw.calculate_payoff_call(stock_price)
        print(f"   {measurement}: Stock=${stock_price:.2f}, Payoff=${payoff:.2f}, Count={count}")
    
    # ========================================================================
    # COMPARISON AND ERROR ANALYSIS
    # ========================================================================
    print_results_header("COMPARISON AND ERROR ANALYSIS")
    
    # Compare Call options
    print("\nCALL OPTION COMPARISON:")
    print(f"  Analytical Black-Scholes:  {format_price(bs_call)}")
    print(f"  Classical Binomial Tree:    {format_price(bt_call)}")
    print(f"  Quantum Random Walk:        {format_price(qrw_call)}")
    
    # Errors for Call
    abs_error_bs_call, rel_error_bs_call = calculate_error(qrw_call, bs_call)
    abs_error_bt_call, rel_error_bt_call = calculate_error(qrw_call, bt_call)
    
    print(f"\n  Quantum vs Analytical BS:")
    print(f"    Absolute Error: ${abs_error_bs_call:.6f}")
    print(f"    Relative Error: {rel_error_bs_call:.4f}%")
    
    print(f"\n  Quantum vs Classical Binomial Tree:")
    print(f"    Absolute Error: ${abs_error_bt_call:.6f}")
    print(f"    Relative Error: {rel_error_bt_call:.4f}%")
    
    # Compare Put options
    print("\nPUT OPTION COMPARISON:")
    print(f"  Analytical Black-Scholes:  {format_price(bs_put)}")
    print(f"  Classical Binomial Tree:    {format_price(bt_put)}")
    print(f"  Quantum Random Walk:         {format_price(qrw_put)}")
    
    # Errors for Put
    abs_error_bs_put, rel_error_bs_put = calculate_error(qrw_put, bs_put)
    abs_error_bt_put, rel_error_bt_put = calculate_error(qrw_put, bt_put)
    
    print(f"\n  Quantum vs Analytical BS:")
    print(f"    Absolute Error: ${abs_error_bs_put:.6f}")
    print(f"    Relative Error: {rel_error_bs_put:.4f}%")
    
    print(f"\n  Quantum vs Classical Binomial Tree:")
    print(f"    Absolute Error: ${abs_error_bt_put:.6f}")
    print(f"    Relative Error: {rel_error_bt_put:.4f}%")
    
    # ========================================================================
    # DEEP DIVE ANALYSIS
    # ========================================================================
    print_results_header("DEEP DIVE ANALYSIS")
    
    # 1. Distribution Plot: Quantum Measurement Histogram vs Theoretical Binomial Distribution
    print("\n1. Distribution Comparison: Quantum vs Theoretical Binomial")
    print("   Generating distribution comparison plot...")
    
    # Use existing qrw instance (N=5)
    # Get measurement counts from the call option result
    measurement_counts = qrw_call_result['measurement_counts']
    
    # Group measurements by number of up-moves (k)
    quantum_probs = np.zeros(N + 1)
    total_shots = sum(measurement_counts.values())
    
    for measurement, count in measurement_counts.items():
        k = measurement.count('1')  # Count number of up-moves
        quantum_probs[k] += count / total_shots
    
    # Calculate theoretical binomial probabilities
    theoretical_probs = np.zeros(N + 1)
    p = qrw.p  # Risk-neutral probability
    for k in range(N + 1):
        theoretical_probs[k] = math.comb(N, k) * (p ** k) * ((1 - p) ** (N - k))
    
    # Create bar chart
    x = np.arange(N + 1)
    width = 0.35
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width/2, quantum_probs, width, label='Quantum Measurement Histogram', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, theoretical_probs, width, label='Theoretical Binomial Distribution', alpha=0.8, color='coral')
    ax1.set_xlabel('Number of Up-Moves (k)', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Distribution Comparison: Quantum Random Walk vs Theoretical Binomial', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in range(N + 1)])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: distribution_comparison.png")
    plt.close()
    
    # 2. Scaling Analysis: Qubit count vs Accuracy
    print("\n2. Scaling Analysis: Number of Steps (Qubits) vs Option Price Error")
    print("   Running scaling analysis...")
    
    N_values = [2, 3, 4, 5, 6, 7, 8]
    errors = []
    quantum_prices = []
    
    for n_val in N_values:
        print(f"   Computing for N={n_val}...", end=' ')
        qrw_scale = QuantumRandomWalk(S0, K, r, sigma, T, n_val)
        result_scale = qrw_scale.price_option('call', use_statevector=True)
        quantum_price = result_scale['price']
        bs_price = black_scholes_call_analytic(S0, K, r, sigma, T)
        error = abs(quantum_price - bs_price)
        errors.append(error)
        quantum_prices.append(quantum_price)
        print(f"Error: ${error:.6f}")
    
    # Create scaling plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(N_values, errors, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax2.set_xlabel('Number of Steps (Qubits)', fontsize=12)
    ax2.set_ylabel('Option Price Error ($)', fontsize=12)
    ax2.set_title('Scaling Analysis: Convergence to Black-Scholes Price', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(N_values)
    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved: scaling_analysis.png")
    plt.close()
    
    # 3. Noise Analysis: Noise Rate vs Calculated Price
    print("\n3. Noise Analysis: Impact of Quantum Noise on Option Pricing")
    print("   Running noise analysis...")
    
    noise_rates = [0.00, 0.01, 0.02, 0.05, 0.10]
    noisy_prices = []
    
    # Use N=5 for noise analysis
    qrw_noise = QuantumRandomWalk(S0, K, r, sigma, T, N)
    bs_price_ref = black_scholes_call_analytic(S0, K, r, sigma, T)
    
    for noise_rate in noise_rates:
        print(f"   Testing noise rate: {noise_rate*100:.1f}%...", end=' ')
        noise_model = QuantumRandomWalk.get_noise_model(noise_rate)
        result_noise = qrw_noise.price_option('call', shots=10000, noise_model=noise_model)
        noisy_price = result_noise['price']
        noisy_prices.append(noisy_price)
        print(f"Price: ${noisy_price:.6f}")
    
    # Create noise analysis plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(noise_rates, noisy_prices, marker='o', linewidth=2, markersize=8, color='darkred', label='Quantum Price with Noise')
    ax3.axhline(y=bs_price_ref, color='green', linestyle='--', linewidth=2, label=f'True Black-Scholes Price (${bs_price_ref:.6f})')
    ax3.set_xlabel('Noise Rate', fontsize=12)
    ax3.set_ylabel('Calculated Price ($)', fontsize=12)
    ax3.set_title('Noise Analysis: Impact of Quantum Hardware Errors', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('noise_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved: noise_analysis.png")
    plt.close()
    
    print("\nDeep Dive Analysis complete! All plots saved.")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_results_header("SUMMARY")
    
    print("\nThe quantum random walk successfully simulates the binomial tree")
    print("model for option pricing. Key observations:")
    print(f"  • Quantum circuit uses {metrics['num_qubits']} qubits for {N} time steps")
    print(f"  • Circuit depth: {metrics['circuit_depth']} gates")
    print(f"  • Quantum prices closely match classical binomial tree prices")
    print(f"  • Differences are primarily due to:")
    print(f"    - Sampling noise (reduced with more shots)")
    print(f"    - Finite number of time steps (discretization error)")
    
    print("\n" + "="*80)
    print("  SIMULATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

