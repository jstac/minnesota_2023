"""
Dynamic programming algorithms.

"""

import numpy as np
import jax.numpy as jnp

def value_function_iteration(v_init, 
                             T,
                             get_greedy,
                             tolerance=1e-6,        # Error tolerance
                             max_iter=10_000,       # Max iteration bound
                             print_step=25,         # Print at multiples
                             verbose=True,
                             usejax=False):
    """
        Compute v_star via VFI and then compute greedy.
    """
    array_lib = jnp if usejax else np

    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance and k <= max_iter:
        v_new = T(v)
        error = array_lib.max(array_lib.abs(v_new - v))
        if verbose and (k % print_step) == 0:
            print(f"Completed iteration {k} with error {error}.")
        v = v_new
        k += 1
    if error > tolerance:
        print(f"Warning: Iteration hit upper bound {max_iter}.")
    elif verbose:
        print(f"Terminated successfully in {k} iterations.")
    v_star = v
    σ_star = get_greedy(v_star)
    return v_star, σ_star


def optimistic_policy_iteration(v_init, 
                                T_σ,
                                get_greedy,
                                tolerance=1e-6, 
                                max_iter=1_000,
                                print_step=10,
                                m=60,
                                usejax=False):
    "Optimistic policy iteration routine."
    
    array_lib = jnp if usejax else np
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance and k < max_iter:
        last_v = v
        σ = get_greedy(v)
        for i in range(m):
            v = T_σ(v, σ)
        error = array_lib.max(array_lib.abs(v - last_v))
        if k % print_step == 0:
            print(f"Completed iteration {k} with error {error}.")
        k += 1
    return v, get_greedy(v)


def howard_policy_iteration(v_init, 
                            get_value,
                            get_greedy,
                            usejax=False):
    "Howard policy iteration routine."
    array_lib = jnp if usejax else np

    σ = get_greedy(v_init)
    i, error = 0, 1.0
    while error > 0:
        v_σ = get_value(σ)
        σ_new = get_greedy(v_σ)
        error = array_lib.max(array_lib.abs(σ_new - σ))
        σ = σ_new
        i = i + 1
        print(f"Concluded loop {i} with error {error}.")
    return v_σ, σ

