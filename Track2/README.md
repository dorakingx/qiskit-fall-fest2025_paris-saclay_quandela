# Quantum Machine Learning for Option Price Prediction

This project implements a **Quantum Reservoir Computing (QRC)** approach for predicting option prices using historical financial data. It was developed for the **Qiskit Fall Fest Quandela 2** competition (Track 2: Quantum Machine Learning).

## Overview

The implementation combines:
- **Quantum Reservoir**: A fixed, non-trainable quantum circuit that encodes classical data and extracts quantum features
- **Classical Regressor**: A trainable classical machine learning model (Linear Regression or MLP) that learns to predict prices from quantum features

This hybrid approach leverages quantum computing's ability to capture non-linear patterns while maintaining the efficiency of classical optimization.

## Architecture

### 1. Data Preprocessing (`src/data_loader.py`)
- Loads data from Excel (.xlsx) or CSV files
- Normalizes data using min-max or z-score normalization
- Creates time-series windows with configurable lookback periods
- Splits data into training and testing sets

### 2. Quantum Reservoir (`src/quantum_reservoir.py`)
- **Encoding**: Maps classical data to quantum states using angle or amplitude encoding
- **Reservoir Circuit**: Fixed circuit with:
  - Rotation gates (RX, RY, RZ) with random angles
  - Entangling gates (CNOTs) in linear or circular patterns
- **Feature Extraction**: Measures expectation values of Pauli Z operators

### 3. Hybrid Model (`src/model.py`)
- Combines quantum reservoir with classical regressor
- Supports Linear Regression and Multi-Layer Perceptron (MLP)
- Provides training, prediction, and evaluation methods

### 4. Main Script (`main.py`)
- Complete pipeline: data loading → preprocessing → training → evaluation
- Generates visualizations and saves results
- Configurable via command-line arguments

## Project Structure

```
Track2/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── quantum_reservoir.py    # QRC circuit implementation
│   └── model.py                # Hybrid QML model
├── main.py                     # Main training and evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Navigate to the Track2 directory:
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

## Usage

### Basic Usage

Run with default parameters:
```bash
python main.py --data_file Train_Dataset_Simulated_Price_swaption.xlsx
```

### Advanced Usage

Customize quantum reservoir and model parameters:
```bash
python main.py \
    --data_file Train_Dataset_Simulated_Price_swaption.xlsx \
    --n_qubits 6 \
    --depth 4 \
    --lookback 15 \
    --regressor mlp \
    --test_size 0.3 \
    --output_dir results
```

### Command-Line Arguments

#### Data Parameters
- `--data_file`: Path to data file (Excel or CSV) [default: Train_Dataset_Simulated_Price_swaption.xlsx]
- `--price_column`: Name of the price column (auto-detected if not specified)
- `--lookback`: Lookback window size for time-series [default: 10]
- `--test_size`: Proportion of data for testing [default: 0.2]

#### Quantum Reservoir Parameters
- `--n_qubits`: Number of qubits in reservoir [default: 4]
- `--depth`: Depth of reservoir circuit [default: 3]
- `--encoding`: Encoding type: `angle` or `amplitude` [default: angle]
- `--entanglement`: Entanglement pattern: `linear` or `circular` [default: linear]
- `--shots`: Number of measurement shots [default: 1024]

#### Classical Regressor Parameters
- `--regressor`: Type of regressor: `linear` or `mlp` [default: linear]

#### Other Parameters
- `--normalize_method`: Normalization method: `minmax` or `zscore` [default: minmax]
- `--output_dir`: Output directory for results [default: results/]
- `--seed`: Random seed for reproducibility [default: 42]
- `--visualize`: Generate visualization plots [default: True]
- `--save_model`: Save the trained model

## Data Format

The data loader supports both Excel (.xlsx, .xls) and CSV files. Expected format:

- **Required**: A column containing price data
- **Optional**: A date column for time-series sorting

The loader will auto-detect common column names like:
- Price columns: `Price`, `price`, `SwaptionPrice`, `swaption_price`, `Value`, `value`
- Date columns: `Date`, `date`

### Example Data Structure

| Date       | Price  |
|------------|--------|
| 2024-01-01 | 100.5  |
| 2024-01-02 | 101.2  |
| ...        | ...    |

## Output

The script generates:

1. **Visualizations** (in `results/` directory):
   - `train_predictions.png`: Training set predictions vs actual
   - `test_predictions.png`: Test set predictions vs actual
   - `residuals.png`: Residual error analysis
   - `quantum_reservoir_circuit.png`: Circuit diagram

2. **Results Summary** (`results_summary.txt`):
   - Model configuration
   - Training and test metrics (MSE, MAE, R², RMSE)

## Example Output

```
================================================================================
  QUANTUM MACHINE LEARNING FOR OPTION PRICE PREDICTION
================================================================================

Configuration:
  Data file: Train_Dataset_Simulated_Price_swaption.xlsx
  Qubits: 4
  Circuit depth: 3
  Encoding: angle
  Lookback window: 10
  Regressor: linear
  Test size: 0.2

--------------------------------------------------------------------------------
  STEP 1: Loading and Preprocessing Data
--------------------------------------------------------------------------------
Loading data from: Train_Dataset_Simulated_Price_swaption.xlsx
Training samples: 800
Test samples: 200
Input shape: (800, 10, 1)

--------------------------------------------------------------------------------
  STEP 2: Building Quantum Reservoir
--------------------------------------------------------------------------------
Reservoir circuit depth: 12

--------------------------------------------------------------------------------
  STEP 3: Training Hybrid QML Model
--------------------------------------------------------------------------------
Extracting quantum reservoir states...
Quantum features shape: (800, 4)
Training classical regressor...
Training completed!

--------------------------------------------------------------------------------
  STEP 4: Evaluating Model
--------------------------------------------------------------------------------

Training Metrics:
  MSE: 0.001234
  MAE: 0.028765
  R2: 0.987654

Test Metrics:
  MSE: 0.001456
  MAE: 0.031234
  R2: 0.985432
```

## How It Works

### Quantum Reservoir Computing

1. **Data Encoding**: Classical time-series data is encoded into quantum states using rotation gates (angle encoding) or amplitude encoding.

2. **Reservoir Processing**: A fixed quantum circuit processes the encoded data:
   - Multiple layers of rotation gates (RX, RY, RZ) with random angles
   - Entangling gates (CNOTs) create quantum correlations
   - The circuit is **non-trainable** (fixed parameters)

3. **Feature Extraction**: Expectation values of Pauli Z operators are measured on each qubit, producing quantum features.

4. **Classical Learning**: A classical regressor learns to map quantum features to target prices.

### Why Quantum Reservoir Computing?

- **Expressivity**: Quantum circuits can capture complex non-linear patterns
- **Efficiency**: Only the classical regressor is trained (smaller parameter space)
- **NISQ Compatibility**: Works well on current quantum hardware (shallow circuits)
- **Interpretability**: Quantum features provide insights into data structure

## Customization

### Using Custom Data

```python
from src.data_loader import DataLoader

loader = DataLoader(lookback_window=15, normalize_method='zscore')
X_train, X_test, y_train, y_test = loader.prepare_data('your_data.csv')
```

### Custom Quantum Reservoir

```python
from src.quantum_reservoir import QuantumReservoir

reservoir = QuantumReservoir(
    n_qubits=8,
    depth=5,
    encoding_type='amplitude',
    entanglement_pattern='circular'
)
```

### Custom Model

```python
from src.model import HybridQMLModel
from sklearn.neural_network import MLPRegressor

model = HybridQMLModel(
    quantum_reservoir=reservoir,
    regressor_type='mlp',
    regressor_params={'hidden_layer_sizes': (100, 50), 'max_iter': 1000}
)
```

## Performance Tips

1. **Qubit Count**: Start with 4-6 qubits. More qubits increase expressivity but also computation time.

2. **Circuit Depth**: Depth of 3-5 layers usually provides good balance between expressivity and noise.

3. **Lookback Window**: Experiment with different window sizes (5-20) based on your data characteristics.

4. **Encoding**: Angle encoding is faster and more NISQ-friendly. Amplitude encoding requires more qubits but can capture more information.

5. **Regressor**: Linear regression is faster and often sufficient. MLP can capture more complex patterns but requires more data.

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.10+)

### Data Loading Issues
- Verify file path is correct
- Check that the file format is supported (.xlsx, .csv)
- Ensure the price column exists in the data

### Memory Issues
- Reduce number of qubits or circuit depth
- Reduce lookback window size
- Process data in batches

### Poor Performance
- Try different normalization methods
- Experiment with different lookback windows
- Increase circuit depth or number of qubits
- Try MLP regressor instead of linear

## References

- Qiskit Documentation: https://qiskit.org/
- Quantum Reservoir Computing: https://arxiv.org/abs/2003.05924
- Qiskit Machine Learning: https://qiskit.org/ecosystem/machine-learning/

## License

Developed for Qiskit Fall Fest 2025 - Paris-Saclay, Quandera Track 2

## Sponsors: A Mila, Quandela, AMF

## Authors

Team Name: Quantum Pioneers
