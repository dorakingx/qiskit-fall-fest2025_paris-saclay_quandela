"""
Data Loading and Preprocessing Module

This module handles loading financial data (Excel/CSV), normalization,
and creating time-series windows for quantum machine learning.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
import warnings
import re


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
    
    def parse_swaption_data(
        self,
        file_path: Union[str, Path]
    ) -> Dict[Tuple[float, float], List[float]]:
        """
        Parse Swaption data file with text-based format.
        
        Expected format:
        - Lines like "Tenor : 1; Maturity : 0.5" followed by price sequences
        - Prices continue until next header or EOF
        
        Parameters
        ----------
        file_path : str or Path
            Path to the Swaption data file
        
        Returns
        -------
        Dict[Tuple[float, float], List[float]]
            Dictionary mapping (tenor, maturity) tuples to price lists
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read Excel file without header to preserve text structure
        if file_path.suffix in ['.xlsx', '.xls']:
            # Read as raw data, preserving all cells
            df_raw = pd.read_excel(file_path, header=None)
        else:
            raise ValueError(
                f"Swaption format parsing only supports Excel files. "
                f"Got: {file_path.suffix}"
            )
        
        # Convert to string representation for regex matching
        # Flatten the dataframe to a list of strings
        data_dict = {}
        current_tenor = None
        current_maturity = None
        current_prices = []
        
        # Regex pattern to match "Tenor : X; Maturity : Y"
        pattern = re.compile(
            r'Tenor\s*:\s*(\d+(?:\.\d+)?)\s*;\s*Maturity\s*:\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE
        )
        
        # Iterate through all cells in the dataframe
        for row_idx in range(len(df_raw)):
            for col_idx in range(len(df_raw.columns)):
                cell_value = df_raw.iloc[row_idx, col_idx]
                
                # Skip NaN/empty cells
                if pd.isna(cell_value):
                    continue
                
                cell_str = str(cell_value).strip()
                
                # Check if this cell matches the Tenor/Maturity pattern
                match = pattern.search(cell_str)
                if match:
                    # Save previous pair if exists
                    if current_tenor is not None and current_maturity is not None:
                        if len(current_prices) > 0:
                            data_dict[(current_tenor, current_maturity)] = current_prices
                    
                    # Start new pair
                    current_tenor = float(match.group(1))
                    current_maturity = float(match.group(2))
                    current_prices = []
                
                # Try to parse as float (price value)
                elif current_tenor is not None and current_maturity is not None:
                    try:
                        price = float(cell_str)
                        if not np.isnan(price) and np.isfinite(price):
                            current_prices.append(price)
                    except (ValueError, TypeError):
                        # Not a valid price, skip
                        pass
        
        # Save last pair
        if current_tenor is not None and current_maturity is not None:
            if len(current_prices) > 0:
                data_dict[(current_tenor, current_maturity)] = current_prices
        
        return data_dict
    
    def get_available_pairs(
        self,
        file_path: Union[str, Path]
    ) -> List[Tuple[float, float]]:
        """
        Get all available (Tenor, Maturity) pairs from Swaption data file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the Swaption data file
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (tenor, maturity) tuples
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try to parse as Swaption format
        if file_path.suffix in ['.xlsx', '.xls']:
            try:
                swaption_data = self.parse_swaption_data(file_path)
                return list(swaption_data.keys())
            except Exception:
                # Not Swaption format, return empty list
                return []
        
        return []
    
    def load_data(
        self,
        file_path: Union[str, Path],
        price_column: Optional[str] = None,
        date_column: Optional[str] = None,
        target_tenor: Optional[float] = None,
        target_maturity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load data from Excel or CSV file.
        
        Supports both Swaption text-based format and standard table format.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file (.xlsx, .xls, or .csv)
        price_column : str, optional
            Name of the price column. If None, will try to auto-detect.
        date_column : str, optional
            Name of the date column. If None, will try to auto-detect.
        target_tenor : float, optional
            Tenor value to extract (for Swaption format)
        target_maturity : float, optional
            Maturity value to extract (for Swaption format)
        
        Returns
        -------
        pd.DataFrame
            Loaded dataframe with data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try to detect Swaption format first
        if file_path.suffix in ['.xlsx', '.xls']:
            # Read a sample to check format
            df_sample = pd.read_excel(file_path, header=None, nrows=50)
            sample_text = df_sample.to_string()
            
            # Check if it contains Swaption format pattern
            swaption_pattern = re.compile(
                r'Tenor\s*:\s*\d+(?:\.\d+)?\s*;\s*Maturity\s*:\s*\d+(?:\.\d+)?',
                re.IGNORECASE
            )
            
            is_swaption_format = swaption_pattern.search(sample_text) is not None
            
            if is_swaption_format:
                # Parse Swaption format
                swaption_data = self.parse_swaption_data(file_path)
                
                # If target pair specified, extract it
                if target_tenor is not None and target_maturity is not None:
                    # Find matching pair (with tolerance for floating point)
                    matching_pair = None
                    for (t, m), prices in swaption_data.items():
                        if (abs(t - target_tenor) < 1e-6 and 
                            abs(m - target_maturity) < 1e-6):
                            matching_pair = (t, m)
                            break
                    
                    if matching_pair is None:
                        available_pairs = list(swaption_data.keys())
                        raise ValueError(
                            f"No data found for Tenor={target_tenor}, "
                            f"Maturity={target_maturity}. "
                            f"Available pairs: {available_pairs}"
                        )
                    
                    prices = swaption_data[matching_pair]
                    # Create DataFrame with sequential index (as time steps)
                    df = pd.DataFrame({
                        'Price': prices
                    })
                    df.index.name = 'TimeStep'
                    return df
                else:
                    # Return all pairs info or first pair
                    if len(swaption_data) == 0:
                        raise ValueError("No Swaption data found in file")
                    
                    # Use first pair if not specified
                    first_pair = list(swaption_data.keys())[0]
                    prices = swaption_data[first_pair]
                    df = pd.DataFrame({
                        'Price': prices
                    })
                    df.index.name = 'TimeStep'
                    warnings.warn(
                        f"No target Tenor/Maturity specified. Using first pair: {first_pair}",
                        UserWarning
                    )
                    return df
        
        # Standard format loading
        if file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: .xlsx, .xls, .csv"
            )
        
        # Auto-detect date column
        if date_column is None:
            date_candidates = ['Date', 'date', 'DATE', 'Time', 'time', 'TIME']
            date_column = next(
                (col for col in date_candidates if col in df.columns),
                None
            )
        
        # Parse date column if available
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            # Set date as index for time-series operations
            df = df.set_index(date_column)
            df = df.sort_index()
        
        # Auto-detect columns if not specified
        if price_column is None:
            # Try common price column names
            price_candidates = ['Price', 'price', 'PRICE', 'SwaptionPrice', 
                              'swaption_price', 'Value', 'value']
            price_column = next(
                (col for col in price_candidates if col in df.columns),
                None
            )
            if price_column is None:
                # If no price column found, use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    price_column = numeric_cols[0]
                    warnings.warn(
                        f"Price column not specified. Using: {price_column}",
                        UserWarning
                    )
                else:
                    raise ValueError(
                        "Could not auto-detect price column. "
                        f"Available columns: {list(df.columns)}"
                    )
        
        # Ensure price column exists
        if price_column not in df.columns:
            raise ValueError(
                f"Price column '{price_column}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Select relevant columns (keep date index if it exists)
        columns_to_keep = [price_column]
        df = df[columns_to_keep].copy()
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        return df
    
    def select_tenor_maturity(
        self,
        df: pd.DataFrame,
        tenor: Optional[float] = None,
        maturity: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Select data for a specific (Tenor, Maturity) pair.
        
        Handles both grid format (where columns/rows represent Tenor/Maturity)
        and list format (where Tenor and Maturity are separate columns).
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        tenor : float, optional
            Tenor value to filter by
        maturity : float, optional
            Maturity value to filter by
        
        Returns
        -------
        pd.DataFrame
            Filtered dataframe for the specified (Tenor, Maturity) pair
        """
        if tenor is None and maturity is None:
            return df
        
        # Check if Tenor and Maturity are columns (list format)
        if 'Tenor' in df.columns and 'Maturity' in df.columns:
            filtered_df = df.copy()
            if tenor is not None:
                filtered_df = filtered_df[filtered_df['Tenor'] == tenor]
            if maturity is not None:
                filtered_df = filtered_df[filtered_df['Maturity'] == maturity]
            return filtered_df
        
        # Check if Tenor and Maturity are in index/columns (grid format)
        # This handles cases where the data is structured as a matrix
        # with Tenor as rows and Maturity as columns, or vice versa
        if tenor is not None or maturity is not None:
            # Try to find columns/rows matching the values
            # This is a simplified approach - may need adjustment based on actual data format
            warnings.warn(
                "Tenor/Maturity filtering for grid format may need manual adjustment. "
                "Please verify the data structure.",
                UserWarning
            )
        
        return df
    
    def compute_log_returns(
        self, 
        prices: np.ndarray,
        clip_extreme: bool = True,
        clip_threshold: float = 5.0
    ) -> np.ndarray:
        """
        Compute log returns from price series: ln(P_t / P_{t-1}).
        
        Log returns are often more stationary and better for financial predictions.
        
        Parameters
        ----------
        prices : np.ndarray
            Price series (1D array)
        clip_extreme : bool, default=True
            If True, clip extreme log returns to prevent outliers
        clip_threshold : float, default=5.0
            Threshold for clipping (in standard deviations)
        
        Returns
        -------
        np.ndarray
            Log returns series (1D array, length = len(prices) - 1)
        """
        prices = np.asarray(prices).flatten()
        
        # Handle NaN and infinite values
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            # Forward fill NaN values
            mask_valid = ~(np.isnan(prices) | np.isinf(prices))
            if np.sum(mask_valid) == 0:
                raise ValueError("No valid prices found (all NaN or Inf)")
            
            # Use forward fill for NaN (handle deprecated method parameter)
            prices_series = pd.Series(prices)
            prices_filled = prices_series.ffill().bfill()
            prices = prices_filled.values
        
        # Ensure all prices are positive
        if np.any(prices <= 0):
            # Replace zero/negative with small positive value
            min_positive = np.min(prices[prices > 0])
            if min_positive > 0:
                prices = np.where(prices <= 0, min_positive, prices)
                warnings.warn(
                    f"Found {np.sum(prices <= 0)} non-positive prices. "
                    "Replaced with minimum positive value.",
                    UserWarning
                )
            else:
                raise ValueError(
                    "Cannot compute log returns: all prices are non-positive."
                )
        
        # Compute log returns: ln(P_t / P_{t-1})
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Handle NaN/Inf in log returns
        if np.any(np.isnan(log_returns)) or np.any(np.isinf(log_returns)):
            # Replace NaN/Inf with 0 or forward fill
            log_returns = np.where(
                np.isnan(log_returns) | np.isinf(log_returns),
                0.0,
                log_returns
            )
            warnings.warn(
                "Found NaN/Inf in log returns. Replaced with 0.",
                UserWarning
            )
        
        # Clip extreme values if requested
        if clip_extreme:
            mean_lr = np.mean(log_returns)
            std_lr = np.std(log_returns)
            if std_lr > 0:
                lower_bound = mean_lr - clip_threshold * std_lr
                upper_bound = mean_lr + clip_threshold * std_lr
                log_returns = np.clip(log_returns, lower_bound, upper_bound)
        
        return log_returns
    
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
        date_column: Optional[str] = None,
        tenor: Optional[float] = None,
        maturity: Optional[float] = None,
        use_log_returns: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Complete data preparation pipeline: load, filter, normalize, window, split.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        price_column : str, optional
            Name of the price column
        date_column : str, optional
            Name of the date column
        tenor : float, optional
            Tenor value to filter by (for Swaption data)
        maturity : float, optional
            Maturity value to filter by (for Swaption data)
        use_log_returns : bool, default=False
            If True, use log returns instead of raw prices
        max_samples : int, optional
            Maximum number of samples to use. If provided and less than dataframe length,
            truncates the dataframe to first max_samples rows for faster processing.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            X_train, X_test, y_train, y_test, test_initial_prices
            test_initial_prices: Raw prices needed for reconstruction (None if not using log returns)
        """
        # Load data (with Tenor/Maturity filtering if specified)
        df = self.load_data(
            file_path, 
            price_column, 
            date_column,
            target_tenor=tenor,
            target_maturity=maturity
        )
        
        # Additional filtering if needed (for standard format)
        if tenor is not None or maturity is not None:
            df = self.select_tenor_maturity(df, tenor, maturity)
            if len(df) == 0:
                raise ValueError(
                    f"No data found for Tenor={tenor}, Maturity={maturity}"
                )
        
        # Truncate dataset if max_samples is specified (for quick tuning)
        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples]
            print(f"Truncating dataset to first {max_samples} samples for speed.")
        
        # Extract price column
        if price_column is None:
            # Find first numeric column (excluding index if it's a date)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_column = numeric_cols[0]
            else:
                raise ValueError("No numeric column found for prices")
        
        raw_prices = df[price_column].values.astype(float)
        prices = raw_prices.copy()
        
        # Store original prices for price reconstruction if using log returns
        test_initial_prices = None
        
        # Compute log returns if requested
        if use_log_returns:
            try:
                prices = self.compute_log_returns(prices)
                # Note: log returns has length len(original_prices) - 1
                # create_windows() handles this correctly by starting from lookback_window index
                # This means we'll have one fewer window, which is expected and correct
                
                # Store raw prices for reconstruction
                # We need prices at indices corresponding to test set predictions
                # After windowing, test set starts at train_size index in the windowed data
                # The corresponding raw price index is: lookback_window + train_size
                # But we need the price BEFORE each prediction, so we store prices at the right indices
                test_initial_prices = raw_prices.copy()
            except ValueError as e:
                warnings.warn(
                    f"Could not compute log returns: {e}. Using raw prices instead.",
                    UserWarning
                )
                use_log_returns = False
        
        # Normalize
        prices_normalized = self.normalize(prices, fit=True)
        
        # Create windows
        X, y = self.create_windows(
            prices_normalized.reshape(-1, 1),
            target_column_idx=0
        )
        
        # Split train/test
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)
        
        # Extract test_initial_prices for price reconstruction
        if use_log_returns and test_initial_prices is not None:
            # Calculate the starting index in the original price series for test set
            # 
            # Indexing explanation:
            # - Original prices: [P0, P1, P2, ..., Pn] (length n+1)
            # - Log returns: [LR1, LR2, ..., LRn] where LRi = ln(Pi/Pi-1) (length n)
            # - Windows: Window at index i uses [i-lookback, ..., i-1] to predict value at index i
            # - So window at index i predicts log return LRi = ln(Pi/Pi-1)
            # - To reconstruct Pi, we need: Pi = Pi-1 * exp(LRi)
            # - We need Pi-1 as the initial price
            #
            # - First window starts at index 'lookback_window' in log returns array
            # - Test set starts at train_size in the windowed data
            # - So first test window is at log return index: lookback_window + train_size
            # - This window predicts log return at index (lookback_window + train_size)
            # - This log return is: LR_{lookback_window + train_size} = ln(P_{lookback_window + train_size + 1} / P_{lookback_window + train_size})
            # - We need P_{lookback_window + train_size} as the initial price
            
            n_train = len(X_train)
            # Index in log returns array where test set starts
            test_start_log_return_idx = self.lookback_window + n_train
            # Corresponding index in original prices (price before first test prediction)
            test_start_price_idx = test_start_log_return_idx
            
            # Return just the first price (subsequent prices reconstructed recursively)
            if test_start_price_idx < len(test_initial_prices):
                test_initial_prices = np.array([test_initial_prices[test_start_price_idx]])
            else:
                # Edge case: not enough prices
                test_initial_prices = None
        else:
            test_initial_prices = None
        
        return X_train, X_test, y_train, y_test, test_initial_prices

