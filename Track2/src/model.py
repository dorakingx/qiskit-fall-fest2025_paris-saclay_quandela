"""
Hybrid Quantum-Classical Machine Learning Model

This module combines a Quantum Reservoir (non-trainable) with a
Classical Regressor (trainable) for option price prediction.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .quantum_reservoir import QuantumReservoir


class HybridQMLModel:
    """
    Hybrid Quantum-Classical Machine Learning Model.
    
    Combines a fixed quantum reservoir circuit with a trainable
    classical regressor for prediction tasks.
    """
    
    def __init__(
        self,
        quantum_reservoir: QuantumReservoir,
        regressor_type: str = "linear",
        regressor_params: Optional[Dict] = None
    ):
        """
        Initialize the Hybrid QML Model.
        
        Parameters
        ----------
        quantum_reservoir : QuantumReservoir
            The quantum reservoir instance (non-trainable)
        regressor_type : str, default="linear"
            Type of classical regressor: "linear" or "mlp"
        regressor_params : dict, optional
            Parameters to pass to the regressor
        """
        self.quantum_reservoir = quantum_reservoir
        self.regressor_type = regressor_type
        self.regressor_params = regressor_params or {}
        
        # Initialize regressor
        if regressor_type == "linear":
            self.regressor = LinearRegression(**self.regressor_params)
        elif regressor_type == "mlp":
            # Default MLP parameters
            default_mlp_params = {
                "hidden_layer_sizes": (50, 25),
                "max_iter": 500,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1
            }
            default_mlp_params.update(self.regressor_params)
            self.regressor = MLPRegressor(**default_mlp_params)
        else:
            raise ValueError(
                f"Unknown regressor type: {regressor_type}. "
                "Supported types: 'linear', 'mlp'"
            )
        
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = False
    ) -> None:
        """
        Train the model on data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features. Shape: [n_samples, lookback_window, n_features]
            or [n_samples, lookback_window]
        y : np.ndarray
            Target values. Shape: [n_samples]
        verbose : bool, default=False
            Whether to print training progress
        """
        if verbose:
            print("Extracting quantum reservoir states...")
        
        # Extract quantum features from input data
        X_quantum = self.quantum_reservoir.get_reservoir_states(X)
        
        if verbose:
            print(f"Quantum features shape: {X_quantum.shape}")
            print("Training classical regressor...")
        
        # Train the classical regressor
        self.regressor.fit(X_quantum, y)
        self.is_fitted = True
        
        if verbose:
            print("Training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features. Shape: [n_samples, lookback_window, n_features]
            or [n_samples, lookback_window]
        
        Returns
        -------
        np.ndarray
            Predicted values. Shape: [n_samples]
        """
        if not self.is_fitted:
            raise ValueError(
                "Model has not been fitted. Call fit() first."
            )
        
        # Extract quantum features
        X_quantum = self.quantum_reservoir.get_reservoir_states(X)
        
        # Make predictions
        predictions = self.regressor.predict(X_quantum)
        
        return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True target values
        metrics : list, optional
            List of metric names to compute. Default: ['mse', 'mae', 'r2']
        
        Returns
        -------
        dict
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']
        
        predictions = self.predict(X)
        
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = mean_squared_error(y, predictions)
        
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y, predictions)
        
        if 'r2' in metrics:
            results['r2'] = r2_score(y, predictions)
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y, predictions))
        
        return results
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (coefficients) from the regressor.
        
        Returns
        -------
        np.ndarray or None
            Feature coefficients if available (for linear regression)
        """
        if hasattr(self.regressor, 'coef_'):
            return self.regressor.coef_
        return None
    
    def get_intercept(self) -> Optional[float]:
        """
        Get intercept from the regressor.
        
        Returns
        -------
        float or None
            Intercept value if available
        """
        if hasattr(self.regressor, 'intercept_'):
            return self.regressor.intercept_
        return None

