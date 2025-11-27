# Quantum Option Pricing using Quantum Random Walk

This project implements a quantum random walk approach to price European options, simulating the binomial tree model on quantum hardware. It was developed for the Qiskit Fall Fest 2025 competition.

## Overview

The implementation uses a quantum circuit to encode the probability distribution of stock prices at maturity, allowing us to calculate option payoffs in superposition. The circuit achieves **O(1) constant depth** using independent R_y rotations, making it well-suited for Noisy Intermediate-Scale Quantum (NISQ) devices.

### Key Features

- **Quantum Random Walk**: Simulates binomial tree model using quantum circuits
- **Noise Simulation**: Includes depolarizing and readout errors for realistic hardware simulation
- **Comprehensive Analysis**: 
  - Distribution comparison (Quantum vs Theoretical Binomial)
  - Scaling analysis (Qubit count vs Accuracy)
  - Noise analysis (Impact of quantum hardware errors)
- **Visualization**: Generates circuit diagrams and analysis plots
- **Classical Benchmarks**: Compares against Black-Scholes and Binomial Tree methods

## Project Structure

```
Track1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main.py                            # Main execution script
├── quantum_random_walk.py            # Quantum circuit implementation
├── classical_benchmark.py            # Classical pricing methods
├── Example_Black_Scholes_Model.py    # Example implementation
├── Example_Black_Scholes_Model.xlsx  # Example data
├── Black-Scholes-Model.pdf           # Documentation
├── Black-Scholes-Model.docx          # Documentation
├── images/                            # Generated visualization files
│   ├── circuit_diagram.png           # Circuit visualization
│   ├── distribution_comparison.png    # Distribution analysis plot
│   ├── scaling_analysis.png          # Scaling analysis plot
│   └── noise_analysis.png            # Noise analysis plot
└── results_summary.md                 # Detailed results summary
```

## Installation

### Prerequisites

- Python 3.11 or 3.12 (Python 3.14 may have compatibility issues with qiskit-aer)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dorakingx/qiskit-fall-fest2025_paris-saclay_quandela.git
cd qiskit-fall-fest2025_paris-saclay_quandela/Quandela/Track1
```

2. Create a virtual environment (recommended):
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Running the Main Simulation

Execute the main script to run the complete analysis:

```bash
python main.py
```

This will:
1. Run classical benchmarks (Black-Scholes and Binomial Tree)
2. Build and execute the quantum circuit
3. Generate comparison analysis
4. Create visualization plots
5. Generate results summary

### Output Files

The script generates the following files in the `images/` directory:

- **images/circuit_diagram.png**: Visual representation of the quantum circuit
- **images/distribution_comparison.png**: Comparison between quantum measurements and theoretical binomial distribution
- **images/scaling_analysis.png**: Error convergence as number of qubits increases
- **images/noise_analysis.png**: Impact of quantum noise on pricing accuracy
- **results_summary.md**: Detailed markdown summary of results (in root directory)

### Customizing Parameters

Edit the parameters in `main.py`:

```python
S0 = 100.0      # Spot price
K = 100.0       # Strike price
r = 0.05        # Risk-free rate (5%)
sigma = 0.2     # Volatility (20%)
T = 1.0         # Time to maturity (1 year)
N = 5           # Number of time steps (qubits)
shots = 10000   # Number of measurement shots
```

## How It Works

### Quantum Circuit Design

The quantum circuit uses N qubits to represent N time steps in the binomial tree:
- Each qubit |0⟩ represents a down-move, |1⟩ represents an up-move
- The state |b_{N-1} b_{N-2} ... b_0⟩ represents a path through the tree
- If k qubits are |1⟩, this represents k up-moves and (N-k) down-moves
- Final stock price: S_T = S_0 * u^k * d^(N-k)
- Probability: P(k) = C(N,k) * p^k * (1-p)^(N-k)

### State Preparation

The circuit uses independent R_y rotations on each qubit:
- **Constant Depth**: O(1) regardless of number of qubits
- **Noise Resilience**: Independent rotations minimize noise accumulation
- **NISQ Compatible**: Shallow circuit depth suitable for near-term quantum hardware

### Noise Model

The implementation includes:
- **Depolarizing Errors**: Simulates gate errors on single-qubit gates
- **Readout Errors**: Simulates measurement errors (default 1% error rate)

## Results

### Performance Metrics

- **Circuit Depth**: 1 gate (constant, independent of qubit count)
- **Accuracy**: 
  - Call options: ~2.5% error vs Black-Scholes
  - Put options: ~4.4% error vs Black-Scholes
- **Noise Robustness**: Maintains accuracy with error rates up to 10%

### Key Findings

1. The quantum random walk successfully simulates the binomial tree model
2. Quantum prices closely match classical methods
3. The O(1) depth circuit is well-suited for NISQ devices
4. Implementation remains robust under realistic noise conditions

## Technical Details

### Dependencies

- `qiskit>=1.0.0`: Quantum computing framework
- `qiskit-aer>=0.13.0`: Quantum simulator with noise models
- `numpy>=1.21.0`: Numerical computations
- `scipy>=1.7.0`: Scientific computing (for Black-Scholes)
- `matplotlib>=3.5.0`: Plotting and visualization
- `pylatexenc>=2.0`: Circuit diagram rendering
- `pillow>=8.0`: Image processing

### Circuit Metrics

For N=5 qubits:
- Number of Qubits: 5
- Circuit Depth: 1
- Number of Gates: 5 (all R_y rotations)

## References

- Black-Scholes Model: [Black-Scholes-Model.pdf](Black-Scholes-Model.pdf)
- Qiskit Documentation: https://qiskit.org/
- Binomial Tree Model: Cox-Ross-Rubinstein model

## License

This project was developed for the Qiskit Fall Fest 2025 competition.

## Authors

Developed for Qiskit Fall Fest 2025 - Paris-Saclay Track 1

## Acknowledgments

- Qiskit team for the quantum computing framework
- IBM Quantum for quantum simulation capabilities

