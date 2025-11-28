"""
Data Loading and Preprocessing Module

This module handles loading financial data (Excel/CSV), normalization,
and creating time-series windows for quantum machine learning.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import warnings


class DataLoader:
    """
    Data loader and preprocessor for option price prediction.
    
    Supports loading Excel and CSV files, normalization, and creating
    time-series windows for sequence-based prediction.
    """
    
    def __init__(
        self,
        normalize_method: str = "minmax",
        lookback_window: int = 10,
        test_size: float = 0.2,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        normalize_method : str, default="minmax"
            Normalization method: "minmax" or "zscore"
        lookback_window : int, default=10
            Number of time steps to look back for prediction
        test_size : float, default=0.2
            Proportion of data to use for testing (0.0 to 1.0)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.normalize_method = normalize_method
        self.lookback_window = lookback_window
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Store normalization parameters
        self.feature_min_ = None
        self.feature_max_ = None
        self.feature_mean_ = None
        self.feature_std_ = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def load_data(
        self,
        file_path: Union[str, Path],
        price_column: Optional[str] = None,
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from Excel or CSV file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file (.xlsx, .xls, or .csv)
        price_column : str, optional
            Name of the price column. If None, will try to auto-detect.
        date_column : str, optional
            Name of the date column. If None, will try to auto-detect.
        
        Returns
        -------
        pd.DataFrame
            Loaded dataframe with data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load based on file extension
        if file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .xlsx, .xls, .csv"
            )
        
        # Auto-detect columns if not specified
        if price_column is None:
            # Try common price column names
            price_candidates = ['Price', 'price', 'PRICE', 'SwaptionPrice', 
                              'swaption_price', 'Value', 'value']
            price_column = next(
                (col for col in price_candidates if col in df.columns),
                df.columns[0]  # Default to first column
            )
            warnings.warn(
                f"Price column not specified. Using: {price_column}",
                UserWarning
            )
        
        if date_column is None and 'Date' in df.columns:
            date_column = 'Date'
        
        # Ensure price column exists
        if price_column not in df.columns:
            raise ValueError(
                f"Price column '{price_column}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Select relevant columns
        columns_to_keep = [price_column]
        if date_column and date_column in df.columns:
            columns_to_keep.insert(0, date_column)
        
        df = df[columns_to_keep].copy()
        
        # Sort by date if available
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.sort_values(by=date_column).reset_index(drop=True)
        
        # Remove any rows with NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def normalize(
        self,
        data: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize the data using the specified method.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to normalize (1D or 2D array)
        fit : bool, default=True
            If True, fit normalization parameters. If False, use existing parameters.
        
        Returns
        -------
        np.ndarray
            Normalized data
        """
        data = np.asarray(data)
        original_shape = data.shape
        data = data.flatten().reshape(-1, 1) if data.ndim == 1 else data
        
        if self.normalize_method == "minmax":
            if fit:
                self.feature_min_ = np.min(data, axis=0)
                self.feature_max_ = np.max(data, axis=0)
                # Avoid division by zero
                self.feature_max_[self.feature_max_ == self.feature_min_] = 1.0
            
            if self.feature_min_ is None or self.feature_max_ is None:
                raise ValueError(
                    "Normalization parameters not fitted. "
                    "Call normalize with fit=True first."
                )
            
            normalized = (data - self.feature_min_) / (
                self.feature_max_ - self.feature_min_ + 1e-8
            )
        
        elif self.normalize_method == "zscore":
            if fit:
                self.feature_mean_ = np.mean(data, axis=0)
                self.feature_std_ = np.std(data, axis=0)
                # Avoid division by zero
                self.feature_std_[self.feature_std_ == 0] = 1.0
            
            if self.feature_mean_ is None or self.feature_std_ is None:
                raise ValueError(
                    "Normalization parameters not fitted. "
                    "Call normalize with fit=True first."
                )
            
            normalized = (data - self.feature_mean_) / (
                self.feature_std_ + 1e-8
            )
        
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalize_method}. "
                "Supported methods: 'minmax', 'zscore'"
            )
        
        # Reshape to original shape
        if len(original_shape) == 1:
            normalized = normalized.flatten()
        
        return normalized
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data using stored normalization parameters.
        
        Parameters
        ----------
        data : np.ndarray
            Normalized data to denormalize
        
        Returns
        -------
        np.ndarray
            Denormalized data
        """
        data = np.asarray(data)
        original_shape = data.shape
        data = data.flatten().reshape(-1, 1) if data.ndim == 1 else data
        
        if self.normalize_method == "minmax":
            if self.feature_min_ is None or self.feature_max_ is None:
                raise ValueError("Normalization parameters not available.")
            
            denormalized = data * (
                self.feature_max_ - self.feature_min_ + 1e-8
            ) + self.feature_min_
        
        elif self.normalize_method == "zscore":
            if self.feature_mean_ is None or self.feature_std_ is None:
                raise ValueError("Normalization parameters not available.")
            
            denormalized = data * (self.feature_std_ + 1e-8) + self.feature_mean_
        
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        # Reshape to original shape
        if len(original_shape) == 1:
            denormalized = denormalized.flatten()
        
        return denormalized
    
    def create_windows(
        self,
        data: np.ndarray,
        target_column_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series windows for sequence prediction.
        
        Parameters
        ----------
        data : np.ndarray
            Input data (2D array: [samples, features])
        target_column_idx : int, default=0
            Index of the column to predict
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X (features): shape [n_samples, lookback_window, n_features]
            y (targets): shape [n_samples]
        """
        data = np.asarray(data)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        if n_samples < self.lookback_window + 1:
            raise ValueError(
                f"Data length ({n_samples}) must be at least "
                f"lookback_window + 1 ({self.lookback_window + 1})"
            )
        
        X = []
        y = []
        
        for i in range(self.lookback_window, n_samples):
            # Input window: [i-lookback_window : i]
            X.append(data[i - self.lookback_window:i])
            # Target: next value
            y.append(data[i, target_column_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def split_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        X : np.ndarray
            Feature array
        y : np.ndarray
            Target array
        shuffle : bool, default=False
            Whether to shuffle before splitting (not recommended for time-series)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]
        
        # For time-series, use last n_test samples as test set
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(
        self,
        file_path: Union[str, Path],
        price_column: Optional[str] = None,
        date_column: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline: load, normalize, window, split.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        price_column : str, optional
            Name of the price column
        date_column : str, optional
            Name of the date column
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test
        """
        # Load data
        df = self.load_data(file_path, price_column, date_column)
        
        # Extract price column (last column if not specified)
        if price_column is None:
            price_column = df.columns[-1]
        
        prices = df[price_column].values.astype(float)
        
        # Normalize
        prices_normalized = self.normalize(prices, fit=True)
        
        # Create windows
        X, y = self.create_windows(
            prices_normalized.reshape(-1, 1),
            target_column_idx=0
        )
        
        # Split train/test
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        
        return X_train, X_test, y_train, y_test

