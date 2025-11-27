import numpy as np
from scipy.stats import norm
import math
from math import exp, sqrt

#####################
#### Simulations ####
#####################

def black_Scholes_Put_Simulation(S0, K, r,  sigma, T, n_sim):
    mean = (r - 0.5 * sigma ** 2) * T
    std = sigma * np.sqrt(T)
    rng = np.random
    z = rng.normal(loc = mean, scale = std, size = n_sim)
    ST = S0 * np.exp(z)
    payoff = np.maximum(K - ST, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    return float(price)

def black_Scholes_Call_Simulation(S0, K, r, sigma, T, n_sim):
    mean = (r - 0.5 * sigma ** 2) * T
    std = sigma * np.sqrt(T)
    rng = np.random
    z = rng.normal(loc = mean, scale = std, size = n_sim)
    ST = S0 * np.exp(z)
    payoff = np.maximum(ST - K, 0.0)
    price = np.exp(-r * T) * payoff.mean()
    return float(price)

S0 = 1
K = 1
r = 0.02
sigma = 0.2
T = 5
n_steps = 5
n_sim = 10000

black_Scholes_Put_Simulation(S0, K, r,  sigma, T, n_sim)
black_Scholes_Call_Simulation(S0, K, r,  sigma, T, n_sim)

#######################
#### Binomial tree ####
#######################

def black_scholes_call_binomial_tree(S0, K, r, sigma, T, n_steps):
    delta_t = T / n_steps
    u = np.exp(np.sqrt(delta_t) * sigma)
    d = np.exp(-np.sqrt(delta_t) * sigma)
    p = (np.exp(delta_t * r) - d) / (u - d)

    n_rows = 2 * n_steps + 1
    n_rows_mid = (n_rows + 1) // 2
    tree = np.zeros((n_rows, n_steps + 1))
    tree[n_rows_mid - 1, 0] = S0 

    for i in range(1, n_steps + 1):
        old_idx_non_zero = np.where(tree[:, i - 1] != 0)[0]
        new_idx_non_zero = np.unique(np.concatenate((old_idx_non_zero + 1, old_idx_non_zero - 1)))
        new_idx_non_zero = new_idx_non_zero[(new_idx_non_zero >= 0) & (new_idx_non_zero < n_rows)]
        new_S_val = np.unique(S0 * (u ** (new_idx_non_zero - (n_rows_mid - 1))))
        tree[new_idx_non_zero, i] = new_S_val

    x = np.arange(0, n_steps + 1)
    vec_prob_payoff = np.array([math.comb(n_steps, k) * (1 - p) ** k * p ** (n_steps - k) for k in x])
    bool_payoff = np.array([True if i % 2 == 0 else False for i in range(2 * n_steps)])
    bool_payoff = np.append(bool_payoff, True)

    vec_payoff = np.maximum(tree[bool_payoff, n_steps] - K, 0)
    call_price_binomial_tree = np.exp(-T * r) * np.sum(vec_payoff * vec_prob_payoff)

    return {
        "mat_Binomial_Tree": tree,
        "vec_Payoff": vec_payoff,
        "vec_Prob_Payoff": vec_prob_payoff,
        "call_Price_Binomial_Tree": call_price_binomial_tree,
        "p": p,
        "d": d,
        "delta_t": delta_t
    }

def black_scholes_put_binomial_tree(S0, K, r, sigma, T, n_steps):
    delta_t = T / n_steps
    u = np.exp(np.sqrt(delta_t) * sigma)
    d = np.exp(-np.sqrt(delta_t) * sigma)
    p = (np.exp(delta_t * r) - d) / (u - d)

    n_rows = 2 * n_steps + 1
    n_rows_mid = (n_rows + 1) // 2
    tree = np.zeros((n_rows, n_steps + 1))
    tree[n_rows_mid - 1, 0] = S0 

    for i in range(1, n_steps + 1):
        old_idx_non_zero = np.where(tree[:, i - 1] != 0)[0]
        new_idx_non_zero = np.unique(np.concatenate((old_idx_non_zero + 1, old_idx_non_zero - 1)))
        new_idx_non_zero = new_idx_non_zero[(new_idx_non_zero >= 0) & (new_idx_non_zero < n_rows)]
        new_S_val = np.unique(S0 * (u ** (new_idx_non_zero - (n_rows_mid - 1))))
        tree[new_idx_non_zero, i] = new_S_val

    x = np.arange(0, n_steps + 1)
    vec_prob_payoff = np.array([math.comb(n_steps, k) * (1 - p) ** k * p ** (n_steps - k) for k in x])
    bool_payoff = np.array([True if i % 2 == 0 else False for i in range(2 * n_steps)])
    bool_payoff = np.append(bool_payoff, True)

    vec_payoff = np.maximum(K - tree[bool_payoff, n_steps], 0)
    put_price_binomial_tree = np.exp(-T * r) * np.sum(vec_payoff * vec_prob_payoff)

    return {
        "mat_Binomial_Tree": tree,
        "vec_Payoff": vec_payoff,
        "vec_Prob_Payoff": vec_prob_payoff,
        "put_Price_Binomial_Tree": put_price_binomial_tree,
        "p": p,
        "d": d,
        "delta_t": delta_t
    }

S0 = 1
K = 1
r = 0.02
sigma = 0.2
T = 5
n_steps = 5

result = black_scholes_put_binomial_tree(S0, K, r, sigma, T, n_steps)

print("Put price:", result["put_Price_Binomial_Tree"])
print("Probability:", result["vec_Prob_Payoff"])
print("Payoffs:", result["vec_Payoff"])
print("Matrix bonimial tree:\n", result["mat_Binomial_Tree"])

result = black_scholes_call_binomial_tree(S0, K, r, sigma, T, n_steps)

print("Call price:", result["call_Price_Binomial_Tree"])
print("Probability:", result["vec_Prob_Payoff"])
print("Payoffs:", result["vec_Payoff"])
print("Matrix bonimial tree:\n", result["mat_Binomial_Tree"])

##################
#### Analytic ####
##################

def black_Scholes_Call_Analytic(S_Zero, K, r, sigma, T):
    d1 = (np.log(S_Zero / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S_Zero * norm.cdf(d1) - K * np.exp(-r * (T)) * norm.cdf(d2)

def black_Scholes_Put_Analytic(S_Zero, K, r, sigma, T):
    d1 = (np.log(S_Zero / K) + (r + 0.5 * sigma ** 2) * (T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * (T)) * norm.cdf(-d2) - S_Zero * norm.cdf(-d1)

S_Zero = 1
K = 1
r = 0.02
sigma = 0.2
T = 5

black_Scholes_Call_Analytic(S_Zero, K, r, sigma, T)
black_Scholes_Put_Analytic(S_Zero, K, r, sigma, T)
