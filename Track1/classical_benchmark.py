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
    Price a European Call option using the Binomial Tree method with backward induction.
    
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
        - 'tree': None (not stored in backward induction approach)
        - 'parameters': Dictionary with u, d, p, delta_t
    """
    delta_t, u, d, p = binomial_tree_parameters(S0, r, sigma, T, n_steps)
    
    # Initialize asset prices at maturity
    # j=0: lowest price (all down moves), j=n_steps: highest price (all up moves)
    ST = np.array([S0 * (u ** j) * (d ** (n_steps - j)) for j in range(n_steps + 1)])
    
    # Initialize option values at maturity (payoffs)
    C = np.maximum(ST - K, 0)
    
    # Store payoffs and probabilities for return value
    vec_payoff = C.copy()
    vec_prob = np.array([math.comb(n_steps, j) * (p ** j) * ((1 - p) ** (n_steps - j)) for j in range(n_steps + 1)])
    
    # Backward induction: iterate from time step n_steps-1 down to 0
    discount_factor = np.exp(-r * delta_t)
    for i in range(n_steps - 1, -1, -1):
        # At each time step i, we have i+1 nodes (j = 0 to i)
        # Node j represents j up-moves and (i-j) down-moves
        # Update option values: C[j] = discount * (p * C[j+1] + (1-p) * C[j])
        # Note: C[j+1] is the value after an up-move, C[j] is after a down-move
        for j in range(i + 1):
            C[j] = discount_factor * (p * C[j + 1] + (1 - p) * C[j])
    
    # The option price is C[0] (current node)
    call_price = C[0]
    
    return {
        'price': float(call_price),
        'payoffs': vec_payoff,
        'probabilities': vec_prob,
        'tree': None,  # Not stored in backward induction approach
        'parameters': {
            'u': u,
            'd': d,
            'p': p,
            'delta_t': delta_t
        }
    }


def binomial_tree_put(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int) -> Dict:
    """
    Price a European Put option using the Binomial Tree method with backward induction.
    
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
        - 'tree': None (not stored in backward induction approach)
        - 'parameters': Dictionary with u, d, p, delta_t
    """
    delta_t, u, d, p = binomial_tree_parameters(S0, r, sigma, T, n_steps)
    
    # Initialize asset prices at maturity
    # j=0: lowest price (all down moves), j=n_steps: highest price (all up moves)
    ST = np.array([S0 * (u ** j) * (d ** (n_steps - j)) for j in range(n_steps + 1)])
    
    # Initialize option values at maturity (payoffs for put)
    P = np.maximum(K - ST, 0)
    
    # Store payoffs and probabilities for return value
    vec_payoff = P.copy()
    vec_prob = np.array([math.comb(n_steps, j) * (p ** j) * ((1 - p) ** (n_steps - j)) for j in range(n_steps + 1)])
    
    # Backward induction: iterate from time step n_steps-1 down to 0
    discount_factor = np.exp(-r * delta_t)
    for i in range(n_steps - 1, -1, -1):
        # At each time step i, we have i+1 nodes (j = 0 to i)
        # Node j represents j up-moves and (i-j) down-moves
        # Update option values: P[j] = discount * (p * P[j+1] + (1-p) * P[j])
        # Note: P[j+1] is the value after an up-move, P[j] is after a down-move
        for j in range(i + 1):
            P[j] = discount_factor * (p * P[j + 1] + (1 - p) * P[j])
    
    # The option price is P[0] (current node)
    put_price = P[0]
    
    return {
        'price': float(put_price),
        'payoffs': vec_payoff,
        'probabilities': vec_prob,
        'tree': None,  # Not stored in backward induction approach
        'parameters': {
            'u': u,
            'd': d,
            'p': p,
            'delta_t': delta_t
        }
    }

