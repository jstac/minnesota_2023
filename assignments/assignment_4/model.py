"""
The inventory management problem: model (Python version)

The discount factor takes the form β_t = Z_t, where (Z_t) is a 
Tauchen discretization of the Gaussian AR(1) process 

    Z_t = ρ Z_{t-1} + b + ν W_t.

Provides primitives and default values.

John Stachurski
Sun 13 Aug 2023 04:25:41

"""

import quantecon as qe
import numpy as np
from collections import namedtuple
from numba import njit, prange, int32


def f(y, a, d):
    " Inventory update rule. "
    return np.maximum(y - d, 0) + a 

# NamedTuple to hold model parameters
Params = namedtuple(
         "Params", ("K", "c", "κ", "p"))

@njit
def build_R(params, y_vals, d_vals, ϕ_vals):
    " Build the R array using loops. "
    K, c, κ, p = params
    n_y = K + 1
    R = np.zeros((n_y, n_y, n_y))
    for y in y_vals:
        for yp in y_vals:
            for a in range(n_y - y):
                hits = f(y, a, d_vals) == yp
                R[y, a, yp] = np.sum(hits * ϕ_vals)
    return R

def build_R_vectorized(params, y_vals, d_vals, ϕ_vals):
    K, c, κ, p = params
    n_y = K + 1
    n_d = len(d_vals)
    # Create R[y, a, yp, d] and then sum out last dimension
    y  = np.reshape(y_vals, (n_y, 1, 1, 1))
    a  = np.reshape(y_vals, (1, n_y, 1, 1))
    yp = np.reshape(y_vals, (1, 1, n_y, 1))
    d  = np.reshape(d_vals, (1, 1, 1, n_d))
    ϕ  = np.reshape(ϕ_vals, (1, 1, 1, n_d))
    feasible = a <= K - y
    temp = (f(y, a, d_vals) == yp) * feasible
    R = np.sum(temp * ϕ_vals, axis=3)
    return R


@njit
def build_r(params, y_vals, d_vals, ϕ_vals):
    K, c, κ, p = params
    n_y = K + 1
    r = np.full((n_y, n_y), -np.inf)
    for y in y_vals:
        revenue = np.sum(np.minimum(y, d_vals) * ϕ_vals)
        for a in range(n_y - y):
            cost = c * a + κ * (a > 0)
            r[y, a] = revenue - cost
    return r


def build_r_vectorized(params, y_vals, d_vals, ϕ_vals):
    K, c, κ, p = params
    n_y = K + 1
    n_d = len(d_vals)
    y = np.reshape(y_vals, (n_y, 1))
    d = np.reshape(d_vals, (1, n_d))
    ϕ = np.reshape(ϕ_vals, (1, n_d))
    revenue = np.minimum(y, d) * ϕ 
    exp_revenue = np.sum(revenue, axis=1)
    exp_revenue = np.reshape(exp_revenue, (n_y, 1))
    a = np.reshape(y_vals, (1, n_y))
    cost = c * a + κ * (a > 0)
    exp_profit = exp_revenue - cost
    feasible = a <= K - y
    r = np.where(feasible, exp_profit, -np.inf)
    return r


# NamedTuple to hold arrays used to solve model
Arrays = namedtuple(
         "Arrays", ("r", "R", "y_vals", "z_vals", "Q"))

# NamedTuple to store parameters, array sizes, and arrays
Model = namedtuple("Model", ("params", "sizes", "arrays"))

def create_sdd_inventory_model(ρ=0.98,        # Z persistence
                               ν=0.002,       # Z volatility
                               n_z=25,        # size of Z grid
                               b=0.97,        # Z mean
                               K=100,         # max inventory        
                               d_max=100,     # max value of d
                               c=0.2,         # unit cost
                               κ=0.8,         # fixed cost
                               p=0.6):        # demand parameter

    n_y = K + 1               # size of state space
    y_vals = np.arange(n_y)   # inventory levels 0,...,K

    # Construct r and R arrays
    def ϕ(d):
        return (1 - p)**d * p                      
    d_vals = np.arange(d_max)
    ϕ_vals = ϕ(d_vals)

    # Build the exogenous discount process 
    mc = qe.tauchen(n_z, ρ, ν)
    z_vals, Q = mc.state_values + b, mc.P
    ρL = np.max(np.abs(np.linalg.eigvals(z_vals * Q)))     
    if ρL >= 1:
        raise NotImplementedError("Error: ρ(L) ≥ 1.")
    else:
        print(f"Building model with ρ(L) = {ρL}")

    # Build namedtuples and return them
    params = Params(K=K, c=c, κ=κ, p=p)
    r = build_r_vectorized(params, y_vals, d_vals, ϕ_vals)
    R = build_R_vectorized(params, y_vals, d_vals, ϕ_vals)

    arrays = Arrays(r=r, R=R, y_vals=y_vals, z_vals=z_vals, Q=Q)
    sizes = n_y, n_z
    return Model(params=params, sizes=sizes, arrays=arrays)

