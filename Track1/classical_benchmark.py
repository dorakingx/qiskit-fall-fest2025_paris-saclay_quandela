"""
Classical Benchmark Methods for Option Pricing

This module provides classical methods for pricing European options:
1. Analytical Black-Scholes formulas (continuous-time)
2. Binomial Tree method (discrete-time approximation)

These serve as benchmarks for comparing against quantum random walk implementations.
"""

import numpy as np
from scipy.stats import norm
import math
from typing import Dict, Tuple


def black_scholes_call_analytic(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate European Call option price using the analytical Black-Scholes formula.
    
    Parameters:
    -----------
    S0 : float
        Current spot price of the underlying asset
    K : float
        Strike price of the option
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    T : float
        Time to maturity (in years)
    
    Returns:
    --------
    float
        The theoretical Call option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return float(call_price)


def black_scholes_put_analytic(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate European Put option price using the analytical Black-Scholes formula.
    
    Parameters:
    -----------
    S0 : float
        Current spot price of the underlying asset
    K : float
        Strike price of the option
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    T : float
        Time to maturity (in years)
    
    Returns:
    --------
    float
        The theoretical Put option price
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return float(put_price)


def binomial_tree_parameters(S0: float, r: float, sigma: float, T: float, n_steps: int) -> Tuple[float, float, float, float]:
    """
    Calculate the parameters for the binomial tree model (Cox-Ross-Rubinstein).
    
    Parameters:
    -----------
    S0 : float
        Current spot price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    T : float
        Time to maturity
    n_steps : int
        Number of time steps in the binomial tree
    
    Returns:
    --------
    Tuple[float, float, float, float]
        (delta_t, u, d, p) where:
        - delta_t: time step size
        - u: up-move factor
        - d: down-move factor
        - p: risk-neutral probability of up-move
    """
    delta_t = T / n_steps
    u = np.exp(sigma * np.sqrt(delta_t))  # Up-move factor
    d = np.exp(-sigma * np.sqrt(delta_t))  # Down-move factor
    p = (np.exp(r * delta_t) - d) / (u - d)  # Risk-neutral probability
    
    return delta_t, u, d, p


def binomial_tree_call(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int) -> Dict:
    """
    Price a European Call option using the Binomial Tree method.
    
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
        Number of time steps
    
    Returns:
    --------
    Dict
        Dictionary containing:
        - 'price': Call option price
        - 'payoffs': Array of payoffs at maturity for each possible outcome
        - 'probabilities': Array of probabilities for each outcome
        - 'tree': Matrix representing the binomial tree
        - 'parameters': Dictionary with u, d, p, delta_t
    """
    delta_t, u, d, p = binomial_tree_parameters(S0, r, sigma, T, n_steps)
    
    # Build the binomial tree
    n_rows = 2 * n_steps + 1
    n_rows_mid = (n_rows + 1) // 2
    tree = np.zeros((n_rows, n_steps + 1))
    tree[n_rows_mid - 1, 0] = S0
    
    # Fill the tree
    for i in range(1, n_steps + 1):
        old_idx_non_zero = np.where(tree[:, i - 1] != 0)[0]
        new_idx_non_zero = np.unique(np.concatenate((old_idx_non_zero + 1, old_idx_non_zero - 1)))
        new_idx_non_zero = new_idx_non_zero[(new_idx_non_zero >= 0) & (new_idx_non_zero < n_rows)]
        new_S_val = np.unique(S0 * (u ** (new_idx_non_zero - (n_rows_mid - 1))))
        tree[new_idx_non_zero, i] = new_S_val
    
    # Calculate probabilities and payoffs
    x = np.arange(0, n_steps + 1)
    vec_prob_payoff = np.array([math.comb(n_steps, k) * (1 - p) ** k * p ** (n_steps - k) for k in x])
    bool_payoff = np.array([True if i % 2 == 0 else False for i in range(2 * n_steps)])
    bool_payoff = np.append(bool_payoff, True)
    
    vec_payoff = np.maximum(tree[bool_payoff, n_steps] - K, 0)
    call_price = np.exp(-r * T) * np.sum(vec_payoff * vec_prob_payoff)
    
    return {
        'price': float(call_price),
        'payoffs': vec_payoff,
        'probabilities': vec_prob_payoff,
        'tree': tree,
        'parameters': {
            'u': u,
            'd': d,
            'p': p,
            'delta_t': delta_t
        }
    }


def binomial_tree_put(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int) -> Dict:
    """
    Price a European Put option using the Binomial Tree method.
    
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
        Number of time steps
    
    Returns:
    --------
    Dict
        Dictionary containing:
        - 'price': Put option price
        - 'payoffs': Array of payoffs at maturity for each possible outcome
        - 'probabilities': Array of probabilities for each outcome
        - 'tree': Matrix representing the binomial tree
        - 'parameters': Dictionary with u, d, p, delta_t
    """
    delta_t, u, d, p = binomial_tree_parameters(S0, r, sigma, T, n_steps)
    
    # Build the binomial tree
    n_rows = 2 * n_steps + 1
    n_rows_mid = (n_rows + 1) // 2
    tree = np.zeros((n_rows, n_steps + 1))
    tree[n_rows_mid - 1, 0] = S0
    
    # Fill the tree
    for i in range(1, n_steps + 1):
        old_idx_non_zero = np.where(tree[:, i - 1] != 0)[0]
        new_idx_non_zero = np.unique(np.concatenate((old_idx_non_zero + 1, old_idx_non_zero - 1)))
        new_idx_non_zero = new_idx_non_zero[(new_idx_non_zero >= 0) & (new_idx_non_zero < n_rows)]
        new_S_val = np.unique(S0 * (u ** (new_idx_non_zero - (n_rows_mid - 1))))
        tree[new_idx_non_zero, i] = new_S_val
    
    # Calculate probabilities and payoffs
    x = np.arange(0, n_steps + 1)
    vec_prob_payoff = np.array([math.comb(n_steps, k) * (1 - p) ** k * p ** (n_steps - k) for k in x])
    bool_payoff = np.array([True if i % 2 == 0 else False for i in range(2 * n_steps)])
    bool_payoff = np.append(bool_payoff, True)
    
    vec_payoff = np.maximum(K - tree[bool_payoff, n_steps], 0)
    put_price = np.exp(-r * T) * np.sum(vec_payoff * vec_prob_payoff)
    
    return {
        'price': float(put_price),
        'payoffs': vec_payoff,
        'probabilities': vec_prob_payoff,
        'tree': tree,
        'parameters': {
            'u': u,
            'd': d,
            'p': p,
            'delta_t': delta_t
        }
    }

