# Qiskit Fall Fest 2025 - Paris-Saclay, Quandela

This repository contains two complementary quantum computing projects for option pricing, developed for the **Qiskit Fall Fest 2025** competition (Paris-Saclay, Quandela track).

## Overview

This project explores quantum computing applications in quantitative finance through two distinct approaches:

- **Track 1**: Quantum Random Walk for Option Pricing - Simulates binomial tree models using quantum circuits
- **Track 2**: Quantum Machine Learning for Option Price Prediction - Uses Quantum Reservoir Computing to predict option prices from historical data

Both tracks demonstrate practical applications of quantum computing in financial modeling and showcase different quantum algorithms suitable for NISQ (Noisy Intermediate-Scale Quantum) devices.

---

## Track 1: Quantum Option Pricing using Quantum Random Walk

### Description

Track 1 implements a quantum random walk approach to price European options by simulating the binomial tree model on quantum hardware. The implementation uses a quantum circuit to encode the probability distribution of stock prices at maturity, allowing option payoffs to be calculated in superposition.

### Key Features

- **O(1) Constant Depth Circuit**: Uses independent R_y rotations, making it well-suited for NISQ devices
- **Quantum Random Walk**: Simulates binomial tree model using quantum circuits
- **Noise Simulation**: Includes depolarizing and readout errors for realistic hardware simulation
- **Comprehensive Analysis**:
  - Distribution comparison (Quantum vs Theoretical Binomial)
  - Scaling analysis (Qubit count vs Accuracy)
  - Noise analysis (Impact of quantum hardware errors)
- **Visualization**: Generates circuit diagrams and analysis plots
- **Classical Benchmarks**: Compares against Black-Scholes and Binomial Tree methods

### Project Structure

```
Track1/
├── README.md                          # Detailed Track 1 documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Main execution script
├── quantum_random_walk.py            # Quantum circuit implementation
├── classical_benchmark.py            # Classical pricing methods
├── Example_Black_Scholes_Model.py    # Example implementation
├── images/                            # Generated visualization files
│   ├── circuit_diagram.png           # Circuit visualization
│   ├── distribution_comparison.png   # Distribution analysis plot
│   ├── scaling_analysis.png         # Scaling analysis plot
│   └── noise_analysis.png           # Noise analysis plot
└── results_summary.md                 # Detailed results summary
```

### Installation

1. Navigate to Track1 directory:
```bash
cd Track1
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

**Note**: Python 3.11 or 3.12 recommended (Python 3.14 may have compatibility issues with qiskit-aer)

### Usage

Run the main simulation:
```bash
python main.py
```

Customize parameters:
```bash
python main.py --spot_price 110.0 --strike_price 100.0 --steps 8 --shots 20000
```

### How It Works

The quantum circuit uses N qubits to represent N time steps in the binomial tree:
- Each qubit |0⟩ represents a down-move, |1⟩ represents an up-move
- The state |b_{N-1} b_{N-2} ... b_0⟩ represents a path through the tree
- Final stock price: S_T = S_0 * u^k * d^(N-k) where k is the number of up-moves
- Probability: P(k) = C(N,k) * p^k * (1-p)^(N-k)

The circuit uses independent R_y rotations on each qubit, achieving **O(1) constant depth** regardless of the number of qubits.

### Results

- **Circuit Depth**: 1 gate (constant, independent of qubit count)
- **Accuracy**: 
  - Call options: ~2.5% error vs Black-Scholes
  - Put options: ~4.4% error vs Black-Scholes
- **Noise Robustness**: Maintains accuracy with error rates up to 10%

For detailed results and analysis, see `Track1/results_summary.md`.

---

## Track 2: Quantum Machine Learning for Option Price Prediction

### Description

Track 2 implements a **Quantum Reservoir Computing (QRC)** approach for predicting option prices using historical financial data. The implementation combines a fixed, non-trainable quantum circuit (quantum reservoir) with a trainable classical machine learning model, leveraging quantum computing's ability to capture non-linear patterns while maintaining the efficiency of classical optimization.

### Key Features

- **Quantum Reservoir Computing**: Fixed quantum circuit that encodes classical data and extracts quantum features
- **Hybrid Architecture**: Combines quantum reservoir with classical regressor (Linear Regression, Ridge Regression, or MLP)
- **Time-Series Processing**: Handles historical price data with configurable lookback windows
- **Data Preprocessing**: Supports log returns, normalization, and Swaption data format
- **Comprehensive Evaluation**: Multiple metrics including MSE, MAE, R², RMSE, and MAPE
- **Visualization**: Generates prediction plots, residual analysis, and circuit diagrams

### Project Structure

```
Track2/
├── README.md                          # Detailed Track 2 documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Main training and evaluation script
├── tune.py                            # Hyperparameter tuning script
├── run_best_params.py                 # Run with optimized parameters
├── src/
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── quantum_reservoir.py          # QRC circuit implementation
│   └── model.py                      # Hybrid QML model
├── results/                           # Generated results and visualizations
│   ├── prediction_plot.png           # Main prediction visualization
│   ├── train_predictions.png         # Training set predictions
│   ├── test_predictions.png          # Test set predictions
│   ├── residuals.png                 # Residual error analysis
│   ├── tuning_heatmap.png            # Hyperparameter tuning results
│   └── results_summary.txt           # Results summary
└── *.xlsx                             # Dataset files
```

### Installation

1. Navigate to Track2 directory:
```bash
cd Track2
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Python 3.10 or higher required

### Usage

#### Basic Usage

Run with default parameters:
```bash
python main.py --data_file Train_Dataset_Simulated_Price_swaption.xlsx
```

#### Advanced Usage

Customize quantum reservoir and model parameters:
```bash
python main.py \
    --data_file Train_Dataset_Simulated_Price_swaption.xlsx \
    --n_qubits 6 \
    --depth 4 \
    --lookback 15 \
    --regressor mlp \
    --test_size 0.3 \
    --use_log_returns \
    --output_dir results
```

#### Hyperparameter Tuning

Run hyperparameter tuning:
```bash
python tune.py --data_file Train_Dataset_Simulated_Price_swaption.xlsx
```

Run with best parameters:
```bash
python run_best_params.py
```

### How It Works

1. **Data Encoding**: Classical time-series data is encoded into quantum states using rotation gates (angle encoding) or amplitude encoding
2. **Reservoir Processing**: A fixed quantum circuit processes the encoded data:
   - Multiple layers of rotation gates (RX, RY, RZ) with random angles
   - Entangling gates (CNOTs) create quantum correlations
   - The circuit is **non-trainable** (fixed parameters)
3. **Feature Extraction**: Expectation values of Pauli Z operators are measured on each qubit, producing quantum features
4. **Classical Learning**: A classical regressor learns to map quantum features to target prices

### Why Quantum Reservoir Computing?

- **Expressivity**: Quantum circuits can capture complex non-linear patterns
- **Efficiency**: Only the classical regressor is trained (smaller parameter space)
- **NISQ Compatibility**: Works well on current quantum hardware (shallow circuits)
- **Interpretability**: Quantum features provide insights into data structure

### Results

Results vary based on hyperparameters and dataset. The model supports:
- Multiple regressor types (Linear, Ridge, MLP)
- Configurable quantum reservoir depth and qubit count
- Log returns for stationarity
- Comprehensive evaluation metrics

For detailed results, see `Track2/results/results_summary.txt`.

---

## Dependencies

### Track 1 Dependencies
- `qiskit>=1.0.0`: Quantum computing framework
- `qiskit-aer>=0.13.0`: Quantum simulator with noise models
- `numpy>=1.21.0`: Numerical computations
- `scipy>=1.7.0`: Scientific computing (for Black-Scholes)
- `matplotlib>=3.5.0`: Plotting and visualization
- `pylatexenc>=2.0`: Circuit diagram rendering
- `pillow>=8.0`: Image processing

### Track 2 Dependencies
- `qiskit>=1.0.0`: Quantum computing framework
- `qiskit-aer>=0.13.0`: Quantum simulator with noise models
- `qiskit-machine-learning>=0.7.0`: Quantum machine learning tools
- `numpy>=1.21.0`: Numerical computations
- `pandas>=1.3.0`: Data manipulation
- `scipy>=1.7.0`: Scientific computing
- `scikit-learn>=1.0.0`: Classical machine learning
- `matplotlib>=3.5.0`: Plotting and visualization
- `openpyxl>=3.0.0`: Excel file support

---

## Project Comparison

| Aspect | Track 1 | Track 2 |
|--------|---------|---------|
| **Approach** | Quantum Random Walk | Quantum Reservoir Computing |
| **Goal** | Price options using quantum simulation | Predict option prices from historical data |
| **Input** | Option parameters (S₀, K, r, σ, T) | Historical price time-series |
| **Output** | Option price (Call/Put) | Predicted future prices |
| **Circuit Depth** | O(1) constant | Configurable (typically 3-5 layers) |
| **Training** | No training required | Classical regressor training |
| **Comparison** | vs Black-Scholes & Binomial Tree | vs Actual future prices |
| **Use Case** | Real-time option pricing | Price forecasting |

---

## Key Findings

### Track 1
1. The quantum random walk successfully simulates the binomial tree model
2. Quantum prices closely match classical methods (~2.5-4.4% error)
3. The O(1) depth circuit is well-suited for NISQ devices
4. Implementation remains robust under realistic noise conditions (up to 10% error rates)

### Track 2
1. Quantum Reservoir Computing can effectively capture non-linear patterns in financial time-series
2. Hybrid quantum-classical approach balances expressivity and efficiency
3. Log returns improve model performance by ensuring stationarity
4. Hyperparameter tuning significantly impacts model performance

---

## Repository Structure

```
.
├── README.md                    # This file (overview of both tracks)
├── Track1/                      # Quantum Random Walk for Option Pricing
│   ├── README.md               # Detailed Track 1 documentation
│   ├── main.py                 # Main execution script
│   ├── quantum_random_walk.py # Quantum circuit implementation
│   ├── classical_benchmark.py # Classical pricing methods
│   └── ...
└── Track2/                      # Quantum Machine Learning for Price Prediction
    ├── README.md               # Detailed Track 2 documentation
    ├── main.py                 # Main training script
    ├── tune.py                 # Hyperparameter tuning
    ├── src/                    # Source code modules
    └── ...
```

---

## Getting Started

### Quick Start - Track 1

```bash
cd Track1
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Quick Start - Track 2

```bash
cd Track2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --data_file Train_Dataset_Simulated_Price_swaption.xlsx
```

---

## References

- **Qiskit Documentation**: https://qiskit.org/
- **Black-Scholes Model**: See `Track1/Black-Scholes-Model.pdf`
- **Quantum Reservoir Computing**: https://arxiv.org/abs/2003.05924
- **Qiskit Machine Learning**: https://qiskit.org/ecosystem/machine-learning/
- **Binomial Tree Model**: Cox-Ross-Rubinstein model

---

## License

This project was developed for the **Qiskit Fall Fest 2025** competition.

## Sponsors

A Mila, Quandela, AMF

## Authors

Developed for Qiskit Fall Fest 2025 - Paris-Saclay, Quandela Track 1 & Track 2

## Acknowledgments

- Qiskit team for the quantum computing framework
- IBM Quantum for quantum simulation capabilities
- Competition organizers and sponsors

---

## Additional Resources

- **Track 1 Detailed Documentation**: See `Track1/README.md`
- **Track 2 Detailed Documentation**: See `Track2/README.md`
- **Track 1 Results**: See `Track1/results_summary.md`
- **Track 2 Results**: See `Track2/results/results_summary.txt`

For questions or issues, please refer to the individual track README files for detailed documentation and troubleshooting guides.

