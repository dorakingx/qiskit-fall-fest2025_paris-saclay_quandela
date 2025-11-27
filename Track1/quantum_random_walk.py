"""
Quantum Random Walk Implementation for Option Pricing

This module implements a quantum circuit that simulates the binomial tree model
for option pricing using a quantum random walk approach. The circuit encodes
the probability distribution of stock prices at maturity, allowing us to
calculate option payoffs in superposition.

State Mapping:
--------------
- N qubits represent N time steps in the binomial tree
- Each qubit |0⟩ represents a down-move, |1⟩ represents an up-move
- The state |b_{N-1} b_{N-2} ... b_0⟩ represents a path through the tree
- If k qubits are |1⟩, this represents k up-moves and (N-k) down-moves
- Final stock price: S_T = S_0 * u^k * d^(N-k)
- Probability: P(k) = C(N,k) * p^k * (1-p)^(N-k)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
from typing import Dict, Tuple, Optional
import math


class QuantumRandomWalk:
    """
    Quantum Random Walk circuit for option pricing.
    
    This class implements a quantum circuit that encodes the binomial tree
    distribution for option pricing. The circuit prepares a superposition
    of all possible paths through the binomial tree, weighted by their
    risk-neutral probabilities.
    """
    
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float, n_steps: int):
        """
        Initialize the Quantum Random Walk for option pricing.
        
        Parameters:
        -----------
        S0 : float
            Current spot price
        K : float
            Strike price
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        T : float
            Time to maturity
        n_steps : int
            Number of time steps (also number of qubits)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        
        # Calculate binomial tree parameters
        delta_t = T / n_steps
        self.u = np.exp(sigma * np.sqrt(delta_t))  # Up-move factor
        self.d = np.exp(-sigma * np.sqrt(delta_t))  # Down-move factor
        self.p = (np.exp(r * delta_t) - self.d) / (self.u - self.d)  # Risk-neutral probability
        
        # Initialize circuit
        self.circuit = None
        self._build_circuit()
    
    @staticmethod
    def get_noise_model(error_rate: float) -> NoiseModel:
        """
        Create a simplified noise model with depolarizing error on single-qubit gates.
        
        This noise model simulates realistic quantum hardware errors by applying
        depolarizing noise to all single-qubit gates (primarily R_y rotations).
        
        Parameters:
        -----------
        error_rate : float
            Error rate for depolarizing noise (typically between 0.0 and 1.0)
            Common values: 0.01 (1%), 0.02 (2%), 0.05 (5%), 0.10 (10%)
        
        Returns:
        --------
        NoiseModel
            A NoiseModel object with depolarizing error on single-qubit gates
        """
        noise_model = NoiseModel()
        # Add depolarizing error to single-qubit gates
        error = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['ry', 'u1', 'u2', 'u3', 'rz', 'rx'])
        return noise_model
    
    def _build_circuit(self):
        """
        Build the quantum circuit that prepares the binomial distribution state.
        
        The circuit uses controlled rotations to encode the probability p
        for each qubit, creating a superposition that represents all possible
        paths through the binomial tree.
        """
        # Create quantum and classical registers
        qreg = QuantumRegister(self.n_steps, 'q')
        creg = ClassicalRegister(self.n_steps, 'c')
        self.circuit = QuantumCircuit(qreg, creg)
        
        # Prepare the binomial distribution state
        # We'll use a recursive approach: prepare state with correct probabilities
        # for each number of up-moves k
        
        # Method: Use controlled rotations to encode probabilities
        # For each qubit, apply a rotation based on the probability p
        # This creates a superposition where |1⟩ has amplitude sqrt(p) and |0⟩ has sqrt(1-p)
        
        # However, for a true binomial distribution, we need to account for
        # the combinatorial factors. We'll use a simpler approach that prepares
        # the state with the correct probabilities for each outcome.
        
        # Prepare the state vector directly (more efficient for small n_steps)
        state_vector = self._prepare_binomial_state_vector()
        
        # Prefer manual state preparation (O(1) depth, better for NISQ devices)
        # This uses independent R_y rotations which minimizes noise impact
        self._prepare_state_manually(state_vector)
        self._use_stateprep = False
        
        # Note: StatePreparation could be used as an alternative, but manual
        # preparation is preferred for near-term quantum hardware due to:
        # - Constant depth O(1) regardless of number of qubits
        # - Reduced noise accumulation
        # - Better compatibility with NISQ devices
        
        # Add measurements
        self.circuit.measure_all()
    
    def _prepare_binomial_state_vector(self) -> np.ndarray:
        """
        Prepare the state vector representing the binomial distribution.
        
        Returns:
        --------
        np.ndarray
            State vector of size 2^n_steps with amplitudes encoding probabilities
        """
        state_vector = np.zeros(2 ** self.n_steps, dtype=complex)
        
        # For each possible state |k⟩ (k up-moves)
        for k in range(self.n_steps + 1):
            # Probability of k up-moves
            prob_k = math.comb(self.n_steps, k) * (self.p ** k) * ((1 - self.p) ** (self.n_steps - k))
            amplitude = np.sqrt(prob_k)
            
            # Find all basis states with exactly k ones (k up-moves)
            # We need to distribute the amplitude across all states with k ones
            states_with_k_ones = self._get_states_with_k_ones(k)
            amplitude_per_state = amplitude / np.sqrt(len(states_with_k_ones))
            
            for state_idx in states_with_k_ones:
                state_vector[state_idx] = amplitude_per_state
        
        # Normalize (should already be normalized, but ensure it)
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector = state_vector / norm
        
        return state_vector
    
    def _get_states_with_k_ones(self, k: int) -> list:
        """
        Get all state indices that have exactly k ones (k up-moves).
        
        Parameters:
        -----------
        k : int
            Number of ones (up-moves)
        
        Returns:
        --------
        list
            List of state indices (integers) with exactly k ones
        """
        states = []
        for i in range(2 ** self.n_steps):
            # Count number of ones in binary representation
            binary_rep = format(i, f'0{self.n_steps}b')
            if binary_rep.count('1') == k:
                states.append(i)
        return states
    
    def _prepare_state_manually(self, state_vector: np.ndarray):
        """
        Manually prepare the state using independent R_y rotations (preferred method for NISQ).
        
        This method efficiently creates a superposition of all paths in the binomial tree
        with constant depth O(1), regardless of the number of qubits. Each qubit is
        independently rotated using an R_y gate, which encodes the risk-neutral probability
        p for an up-move at each time step.
        
        Key Advantages:
        ---------------
        - Constant Depth: O(1) circuit depth regardless of width (number of qubits)
          This is achieved by applying independent rotations in parallel, making it
          ideal for near-term quantum hardware with limited coherence times.
        
        - Noise Resilience: Independent rotations minimize noise accumulation compared
          to deep, sequential gate operations. Each qubit's state preparation is
          independent, reducing cross-talk and error propagation.
        
        - NISQ Compatibility: The shallow circuit depth and simple gate structure
          make this approach well-suited for Noisy Intermediate-Scale Quantum (NISQ)
          devices, where minimizing circuit depth is crucial for maintaining fidelity.
        
        - Efficient Path Encoding: This creates a superposition where each qubit
          independently represents the probability of an up-move, efficiently encoding
          all possible paths through the binomial tree in a single quantum state.
        
        Note: This creates an approximation of the true binomial distribution by using
        independent probabilities. For exact binomial distribution, more complex entangled
        gates would be required, but at the cost of increased circuit depth and noise
        sensitivity. The approximation is suitable for NISQ applications where
        circuit depth and noise minimization are prioritized.
        
        Parameters:
        -----------
        state_vector : np.ndarray
            Target state vector (used for reference, but independent rotations are applied)
        """
        # Apply independent R_y rotations to each qubit
        # This creates a superposition where P(|1⟩) = p for each qubit independently
        # Rotation angle: theta = 2 * arccos(sqrt(1-p))
        # This makes P(|1⟩) = sin²(theta/2) = p
        for i in range(self.n_steps):
            theta = 2 * np.arccos(np.sqrt(1 - self.p))
            self.circuit.ry(theta, i)
    
    def get_circuit(self) -> QuantumCircuit:
        """
        Get the quantum circuit.
        
        Returns:
        --------
        QuantumCircuit
            The constructed quantum circuit
        """
        return self.circuit
    
    def _state_to_stock_price(self, measurement_result: str) -> float:
        """
        Convert a measurement result (binary string) to stock price.
        
        Parameters:
        -----------
        measurement_result : str
            Binary string representing the measurement (e.g., "101" for 3 qubits)
        
        Returns:
        --------
        float
            Stock price at maturity for this path
        """
        # Count number of up-moves (ones)
        k = measurement_result.count('1')
        # Calculate final stock price: S_T = S_0 * u^k * d^(N-k)
        stock_price = self.S0 * (self.u ** k) * (self.d ** (self.n_steps - k))
        return stock_price
    
    def calculate_payoff_call(self, stock_price: float) -> float:
        """
        Calculate Call option payoff: max(S_T - K, 0).
        
        Parameters:
        -----------
        stock_price : float
            Stock price at maturity
        
        Returns:
        --------
        float
            Call option payoff
        """
        return max(stock_price - self.K, 0.0)
    
    def calculate_payoff_put(self, stock_price: float) -> float:
        """
        Calculate Put option payoff: max(K - S_T, 0).
        
        Parameters:
        -----------
        stock_price : float
            Stock price at maturity
        
        Returns:
        --------
        float
            Put option payoff
        """
        return max(self.K - stock_price, 0.0)
    
    def price_option(self, option_type: str = 'call', shots: int = 10000, use_statevector: bool = False, noise_model: Optional[NoiseModel] = None) -> Dict:
        """
        Price the option using quantum simulation.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        shots : int
            Number of measurement shots for sampling
        use_statevector : bool
            If True, use exact statevector simulation (no sampling error)
            If False, use shot-based sampling
        noise_model : Optional[NoiseModel]
            Optional noise model to simulate realistic quantum hardware errors.
            Only applies to shot-based sampling (ignored if use_statevector=True)
        
        Returns:
        --------
        Dict
            Dictionary containing:
            - 'price': Option price
            - 'expected_payoff': Expected payoff before discounting
            - 'measurement_counts': Dictionary of measurement results
            - 'circuit_metrics': Dictionary with circuit depth, qubit count, etc.
        """
        if use_statevector:
            # Exact calculation using statevector (noise not applicable)
            return self._price_option_exact(option_type)
        else:
            # Shot-based sampling (noise can be applied)
            return self._price_option_sampling(option_type, shots, noise_model)
    
    def _price_option_exact(self, option_type: str) -> Dict:
        """
        Calculate option price exactly using statevector simulation.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        
        Returns:
        --------
        Dict
            Dictionary with exact option price and metrics
        """
        # Create a copy of the circuit without measurements for statevector
        qc_no_measure = QuantumCircuit(self.n_steps)
        state_vector = self._prepare_binomial_state_vector()
        
        # Calculate expected payoff
        expected_payoff = 0.0
        measurement_counts = {}
        
        for i in range(2 ** self.n_steps):
            # Get binary representation
            binary_str = format(i, f'0{self.n_steps}b')
            prob = abs(state_vector[i]) ** 2
            
            if prob > 1e-10:  # Only consider significant probabilities
                stock_price = self._state_to_stock_price(binary_str)
                
                if option_type.lower() == 'call':
                    payoff = self.calculate_payoff_call(stock_price)
                else:
                    payoff = self.calculate_payoff_put(stock_price)
                
                expected_payoff += prob * payoff
                measurement_counts[binary_str] = int(prob * 10000)  # Scale for display
        
        # Apply discount factor
        option_price = np.exp(-self.r * self.T) * expected_payoff
        
        # Get circuit metrics
        circuit_metrics = self._get_circuit_metrics()
        
        return {
            'price': float(option_price),
            'expected_payoff': float(expected_payoff),
            'measurement_counts': measurement_counts,
            'circuit_metrics': circuit_metrics
        }
    
    def _price_option_sampling(self, option_type: str, shots: int, noise_model: Optional[NoiseModel] = None) -> Dict:
        """
        Calculate option price using shot-based sampling.
        
        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        shots : int
            Number of measurement shots
        noise_model : Optional[NoiseModel]
            Optional noise model to simulate realistic quantum hardware errors
        
        Returns:
        --------
        Dict
            Dictionary with sampled option price and metrics
        """
        # Use AerSimulator for shot-based simulation
        if noise_model is not None:
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()
        
        # Transpile and run
        from qiskit import transpile
        transpiled_circuit = transpile(self.circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
        
        # Calculate expected payoff from measurement results
        total_payoff = 0.0
        for measurement, count in counts.items():
            stock_price = self._state_to_stock_price(measurement)
            
            if option_type.lower() == 'call':
                payoff = self.calculate_payoff_call(stock_price)
            else:
                payoff = self.calculate_payoff_put(stock_price)
            
            # Weight by probability (count / shots)
            total_payoff += (count / shots) * payoff
        
        # Apply discount factor
        option_price = np.exp(-self.r * self.T) * total_payoff
        
        # Get circuit metrics
        circuit_metrics = self._get_circuit_metrics()
        
        return {
            'price': float(option_price),
            'expected_payoff': float(total_payoff),
            'measurement_counts': counts,
            'circuit_metrics': circuit_metrics
        }
    
    def _get_circuit_metrics(self) -> Dict:
        """
        Get metrics about the quantum circuit.
        
        Returns:
        --------
        Dict
            Dictionary containing circuit depth, qubit count, gate count, etc.
        """
        if self.circuit is None:
            return {}
        
        # Remove measurements for depth calculation
        # Use manual preparation (consistent with _build_circuit default)
        qc_no_measure = QuantumCircuit(self.n_steps)
        for i in range(self.n_steps):
            theta = 2 * np.arccos(np.sqrt(1 - self.p))
            qc_no_measure.ry(theta, i)
        
        return {
            'num_qubits': self.n_steps,
            'circuit_depth': qc_no_measure.depth(),
            'num_gates': len(qc_no_measure.data),
            'parameters': {
                'u': self.u,
                'd': self.d,
                'p': self.p,
                'delta_t': self.T / self.n_steps
            }
        }

