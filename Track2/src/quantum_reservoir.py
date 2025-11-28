"""
Quantum Reservoir Computing Module

This module implements a fixed, non-trainable quantum circuit (reservoir)
that encodes classical data and extracts quantum features for machine learning.
"""

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import warnings

# Try to import AerSimulator, fallback to Statevector simulation if not available
try:
    from qiskit_aer import AerSimulator
    HAS_AER = True
except ImportError:
    HAS_AER = False
    warnings.warn(
        "qiskit-aer not available. Using Statevector simulation instead.",
        UserWarning
    )


class QuantumReservoir:
    """
    Quantum Reservoir Computing circuit for feature extraction.
    
    The reservoir is a fixed, non-trainable quantum circuit that encodes
    classical data and produces quantum features through measurement.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 3,
        encoding_type: str = "angle",
        entanglement_pattern: str = "linear",
        random_seed: Optional[int] = None,
        shots: int = 1024,
        noise_model: Optional[object] = None
    ):
        """
        Initialize the Quantum Reservoir.
        
        Parameters
        ----------
        n_qubits : int, default=4
            Number of qubits in the reservoir circuit
        depth : int, default=3
            Depth of the reservoir circuit (number of layers)
        encoding_type : str, default="angle"
            Encoding method: "angle" (angle encoding) or "amplitude" (amplitude encoding)
        entanglement_pattern : str, default="linear"
            Entanglement pattern: "linear", "circular", or "full"
        random_seed : int, optional
            Random seed for reproducible circuit generation
        shots : int, default=1024
            Number of measurement shots for expectation value estimation
        noise_model : object, optional
            Qiskit noise model for simulation. If None, uses clean simulation.
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.encoding_type = encoding_type
        self.entanglement_pattern = entanglement_pattern
        self.shots = shots
        self.noise_model = noise_model
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize circuit
        self.circuit: Optional[QuantumCircuit] = None
        
        # Initialize simulator with noise model if provided
        if HAS_AER:
            if noise_model is not None:
                self.simulator = AerSimulator(noise_model=noise_model)
            else:
                self.simulator = AerSimulator()
        else:
            self.simulator = None  # Will use Statevector simulation
        
        # Build the reservoir circuit
        self.build_reservoir()
    
    def build_reservoir(self) -> None:
        """
        Build the fixed reservoir circuit with entanglement and rotations.
        
        The circuit consists of:
        1. Data encoding layer
        2. Multiple layers of entangling gates (CNOTs) and rotations (RX, RY, RZ)
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        
        # Generate fixed random angles for rotation gates
        # These are fixed (non-trainable) parameters
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for layer in range(self.depth):
            # Rotation gates with fixed random angles
            for qubit in range(self.n_qubits):
                # Random rotations around X, Y, Z axes
                angle_x = np.random.uniform(0, 2 * np.pi)
                angle_y = np.random.uniform(0, 2 * np.pi)
                angle_z = np.random.uniform(0, 2 * np.pi)
                
                self.circuit.rx(angle_x, qubit)
                self.circuit.ry(angle_y, qubit)
                self.circuit.rz(angle_z, qubit)
            
            # Entangling gates
            if self.entanglement_pattern == "linear":
                # Linear chain: CNOT(i, i+1)
                for qubit in range(self.n_qubits - 1):
                    self.circuit.cx(qubit, qubit + 1)
            
            elif self.entanglement_pattern == "circular":
                # Circular: CNOT(i, i+1) and CNOT(n-1, 0)
                for qubit in range(self.n_qubits - 1):
                    self.circuit.cx(qubit, qubit + 1)
                if self.n_qubits > 1:
                    self.circuit.cx(self.n_qubits - 1, 0)
            
            elif self.entanglement_pattern == "full":
                # Full entanglement: CNOT gates between all pairs (i, j) where i < j
                # This creates maximum entanglement but is more expensive
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        self.circuit.cx(i, j)
            
            else:
                raise ValueError(
                    f"Unknown entanglement pattern: {self.entanglement_pattern}. "
                    "Supported patterns: 'linear', 'circular', 'full'"
                )
    
    def encode_data(
        self,
        data: np.ndarray,
        circuit: Optional[QuantumCircuit] = None
    ) -> QuantumCircuit:
        """
        Encode classical data sequence into quantum circuit.
        
        This method encodes the full sequence of data points to preserve temporal
        information. For angle encoding, elements are mapped 1-to-1 to qubits.
        
        Parameters
        ----------
        data : np.ndarray
            Input data sequence to encode. Can be of any length:
            - Angle encoding: 1D array (will be truncated/padded to n_qubits)
            - Amplitude encoding: 1D array of length 2^n_qubits
        circuit : QuantumCircuit, optional
            Circuit to encode into. If None, creates a new circuit.
        
        Returns
        -------
        QuantumCircuit
            Circuit with encoded data sequence
        """
        if circuit is None:
            circuit = QuantumCircuit(self.n_qubits)
        else:
            # Create a copy to avoid modifying the original
            circuit = circuit.copy()
        
        data = np.asarray(data).flatten()
        
        if self.encoding_type == "angle":
            # Angle encoding: map sequence of data points 1-to-1 to rotation angles
            # This preserves temporal information by encoding the full sequence
            # IMPORTANT: Data is assumed to be globally normalized (Z-score) from DataLoader
            # We preserve magnitude information by NOT normalizing per-window
            if len(data) != self.n_qubits:
                # Handle length mismatches
                if len(data) < self.n_qubits:
                    # Pad with zeros (older history is zero-padded)
                    data = np.pad(data, (0, self.n_qubits - len(data)), 'constant')
                else:
                    # Take last n_qubits values (most recent history)
                    # This ensures we capture the most recent temporal patterns
                    data = data[-self.n_qubits:]
            
            # Map Z-score normalized data directly to rotation angles
            # This preserves magnitude information (small vs large returns are distinguishable)
            # Option 1: Linear mapping - maps Z-score of ±3 to ±π
            # Option 2: Bounded mapping using arctan (commented out)
            # Using linear mapping to preserve magnitude relationships
            angles = data * (np.pi / 3.0)
            # Alternative bounded mapping: angles = np.arctan(data) * 2
            
            # Apply rotation gates: map each sequence element to a qubit
            # This creates a 1-to-1 mapping preserving temporal order AND magnitude
            for qubit in range(self.n_qubits):
                circuit.ry(angles[qubit], qubit)
        
        elif self.encoding_type == "amplitude":
            # Amplitude encoding: map data to quantum state amplitudes
            required_length = 2 ** self.n_qubits
            if len(data) != required_length:
                if len(data) < required_length:
                    # Pad with zeros
                    data = np.pad(
                        data, (0, required_length - len(data)), 'constant'
                    )
                else:
                    # Truncate
                    data = data[:required_length]
            
            # Normalize to ensure valid quantum state
            norm = np.linalg.norm(data)
            if norm > 0:
                data_normalized = data / norm
            else:
                data_normalized = np.zeros(required_length)
                data_normalized[0] = 1.0
            
            # Initialize statevector
            state = Statevector(data_normalized)
            circuit.initialize(state, range(self.n_qubits))
        
        else:
            raise ValueError(
                f"Unknown encoding type: {self.encoding_type}. "
                "Supported types: 'angle', 'amplitude'"
            )
        
        return circuit
    
    def get_reservoir_states(
        self,
        data: np.ndarray,
        use_expectation: bool = True
    ) -> np.ndarray:
        """
        Extract reservoir states from input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data. For time-series windows, shape should be
            [n_samples, lookback_window] or [n_samples, lookback_window, n_features]
        use_expectation : bool, default=True
            If True, use expectation values. If False, use measurement counts.
        
        Returns
        -------
        np.ndarray
            Reservoir states. Shape: [n_samples, n_features]
            where n_features = n_qubits (for expectation) or 2^n_qubits (for counts)
        """
        data = np.asarray(data)
        
        # Handle different input shapes
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 3:
            # Flatten time dimension: [n_samples, lookback_window, n_features]
            # -> [n_samples, lookback_window * n_features]
            n_samples, lookback, n_features = data.shape
            data = data.reshape(n_samples, lookback * n_features)
        
        n_samples = data.shape[0]
        reservoir_states = []
        
        for i in range(n_samples):
            # Extract the full sequence from the lookback window
            # This preserves temporal information instead of aggregating to a single scalar
            window_data = data[i]
            
            # Pass the full sequence to encoding (preserves temporal dynamics)
            # The encode_data method will handle length mismatches appropriately
            encoding_circuit = self.encode_data(window_data)
            
            # Combine with reservoir circuit
            full_circuit = QuantumCircuit(self.n_qubits)
            full_circuit.compose(encoding_circuit, inplace=True)
            full_circuit.compose(self.circuit, inplace=True)
            
            if use_expectation:
                # Measure expectation values of Pauli Z on each qubit
                features = []
                
                if self.simulator is not None:
                    # Use AerSimulator
                    for qubit in range(self.n_qubits):
                        measure_circuit = full_circuit.copy()
                        measure_circuit.measure_all()
                        
                        job = self.simulator.run(measure_circuit, shots=self.shots)
                        result = job.result()
                        counts = result.get_counts(measure_circuit)
                        
                        # Calculate expectation value <Z>
                        expectation = 0.0
                        total_shots = sum(counts.values())
                        for bitstring, count in counts.items():
                            bit = int(bitstring[::-1][qubit])
                            z_value = 1 if bit == 0 else -1
                            expectation += z_value * (count / total_shots)
                        
                        features.append(expectation)
                else:
                    # Use Statevector for exact simulation
                    state = Statevector(full_circuit)
                    for qubit in range(self.n_qubits):
                        # Calculate expectation value <Z> = <0|Z|0> - <1|Z|1>
                        # Z|0> = |0>, Z|1> = -|1>
                        prob_0 = np.abs(state.probabilities()[0]) if qubit == 0 else 0
                        # For multi-qubit, need to trace out other qubits
                        from qiskit.quantum_info import partial_trace
                        from qiskit.quantum_info.operators import Operator
                        from qiskit.quantum_info import SparsePauliOp
                        
                        # Simpler approach: use probabilities
                        probs = state.probabilities_dict()
                        expectation = 0.0
                        for bitstring, prob in probs.items():
                            bit = int(bitstring[::-1][qubit])
                            z_value = 1 if bit == 0 else -1
                            expectation += z_value * prob
                        
                        features.append(expectation)
                
                reservoir_states.append(features)
            
            else:
                # Use measurement counts as features
                measure_circuit = full_circuit.copy()
                measure_circuit.measure_all()
                
                if self.simulator is not None:
                    job = self.simulator.run(measure_circuit, shots=self.shots)
                    result = job.result()
                    counts = result.get_counts(measure_circuit)
                else:
                    # Use Statevector
                    state = Statevector(full_circuit)
                    probs = state.probabilities_dict()
                    # Sample according to probabilities
                    counts = {}
                    for bitstring, prob in probs.items():
                        counts[bitstring] = int(prob * self.shots)
                
                # Convert counts to feature vector
                n_bits = 2 ** self.n_qubits
                feature_vector = np.zeros(n_bits)
                total_shots = sum(counts.values())
                
                for bitstring, count in counts.items():
                    idx = int(bitstring[::-1], 2)  # Reverse for little-endian
                    feature_vector[idx] = count / total_shots if total_shots > 0 else 0
                
                reservoir_states.append(feature_vector)
        
        return np.array(reservoir_states)
    
    def get_circuit_depth(self) -> int:
        """
        Get the depth of the reservoir circuit.
        
        Returns
        -------
        int
            Circuit depth
        """
        if self.circuit is None:
            return 0
        return self.circuit.depth()
    
    def visualize_circuit(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the reservoir circuit.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save the circuit diagram. If None, just prints.
        """
        if self.circuit is None:
            warnings.warn("Circuit not built yet.")
            return
        
        try:
            from qiskit.visualization import circuit_drawer
            from matplotlib import pyplot as plt
            
            fig = self.circuit.draw(output='mpl', style='clifford')
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Circuit diagram saved to {output_file}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            # Fallback to text representation
            print(self.circuit.draw(output='text'))

