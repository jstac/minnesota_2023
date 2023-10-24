"""

Solve the inventory management problem using Python/JAX.

John Stachurski
Sun 13 Aug 2023 04:25:41

"""

from model import *
from dp_algos import *
from plot_code import *

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit  # Use JAX to JIT-compile f
def f(y, a, d):
    " Inventory update. "
    return jnp.maximum(y - d, 0) + a 


JAXModel = namedtuple("Model", 
                      ("params", "sizes", "arrays"))


def create_sdd_inventory_model_jax():

    model = create_sdd_inventory_model()
    params = model.params
    sizes = model.sizes
    arrays = [jax.device_put(a) for a in model.arrays]

    return JAXModel(params, sizes, arrays)


## Operators and functions 

def B(v, params, sizes, arrays):
    """
    B(y, z, a, v) = r(y, a) + β(z) Σ_{y′, z′} v(y′, z′) R(y, a, y′) Q(z, z′)

    """
    # Set up
    K, c, κ, p = params
    r, R, y_vals, z_vals, Q = arrays
    n_y, n_z = sizes
    # broadcasting over     (y,   z,   a)
    r = jnp.reshape(r,      (n_y, 1,   n_y))
    β = jnp.reshape(z_vals, (1,   n_z, 1))
    # broadcasting over  (y,   z,   a,   yp,   zp)
    R = jnp.reshape(R,   (n_y, 1,   n_y, n_y,  1))
    Q = jnp.reshape(Q,   (1,   n_z, 1,   1,    n_z))
    v = jnp.reshape(v,   (1,   1,   1,   n_y,  n_z))

    Ev = jnp.sum(v * R * Q, axis=(3, 4))
    return r + β * Ev


def compute_r_σ(σ, params, sizes, arrays):
    # Set up
    r, R, y_vals, z_vals, Q = arrays
    n_y, n_z = sizes
    z_idx = jnp.arange(n_z)
    # Create r_σ with indices (y,   z)
    y = jnp.reshape(y_vals,    (n_y, 1))
    z = jnp.reshape(z_idx,     (1,   n_z))
    r_σ = r[y, σ[y, z]]
    return r_σ


def compute_R_σ(σ, params, sizes, arrays):
    # Set up
    r, R, y_vals, z_vals, Q = arrays
    n_y, n_z = sizes
    z_idx = jnp.arange(n_z)
    # Create R_σ with indices (y,   z,   yp)
    y  = jnp.reshape(y_vals,   (n_y, 1,   1))
    yp = jnp.reshape(y_vals,   (1,   1,   n_y))
    z  = jnp.reshape(z_idx,    (1,   n_z, 1))
    R_σ = R[y, σ[y, z], yp]
    return R_σ 


def compute_L_σ(σ, params, sizes, arrays):
    # Set up
    r, R, y_vals, z_vals, Q = arrays
    n_y, n_z = sizes
    R_σ = compute_R_σ(σ, params, sizes, arrays)
    # Create L_σ with indices (y,   z,   yp,  zp)
    β   = jnp.reshape(z_vals,  (1,   n_z, 1,   1))
    R_σ = jnp.reshape(R_σ,     (n_y, n_z, n_y, 1))
    Q   = jnp.reshape(Q,       (1,   n_z, 1,   n_z))
    L_σ = β * R_σ * Q
    return L_σ


def T(v, params, sizes, arrays):
    "The Bellman operator."
    return jnp.max(B(v, params, sizes, arrays), axis=2)


def get_greedy(v, params, sizes, arrays):
    "Get a v-greedy policy."
    return jnp.argmax(B(v, params, sizes, arrays), axis=2)


def T_σ(v, σ, params, sizes, arrays):
    "The policy operator."
    r, R, y_vals, z_vals, Q = arrays
    n_z, n_y = len(z_vals), len(y_vals)
    r_σ = compute_r_σ(σ, params, sizes, arrays)
    L_σ = compute_L_σ(σ, params, sizes, arrays)
    v = jnp.reshape(v, (1, 1, n_y, n_z))
    return r_σ + jnp.sum(L_σ * v, axis=(2, 3))


def get_value(σ, params, sizes, arrays):
    r, R, y_vals, z_vals, Q = arrays
    n_z, n_y = len(z_vals), len(y_vals)
    r_σ = compute_r_σ(σ, params, sizes, arrays)
    L_σ = compute_L_σ(σ, params, sizes, arrays)
    # Reshape for matrix algebra
    n = n_z * n_y
    L_σ = jnp.reshape(L_σ, (n, n))
    r_σ = jnp.reshape(r_σ, n)
    # Apply matrix operations --- solve for the value of σ 
    I = jnp.identity(n)
    v_σ = jnp.linalg.solve(I - L_σ,  r_σ)
    # Return as multi-index array
    return jnp.reshape(v_σ, (n_y, n_z))


## JIT compile functions and operators 

B = jax.jit(B, static_argnums=(2,))
compute_r_σ = jax.jit(compute_r_σ, static_argnums=(2,))
compute_R_σ = jax.jit(compute_R_σ, static_argnums=(2,))
compute_L_σ = jax.jit(compute_L_σ, static_argnums=(2,))
T = jax.jit(T, static_argnums=(2,))
T_σ = jax.jit(T_σ, static_argnums=(3,))
get_greedy = jax.jit(get_greedy, static_argnums=(2,))
get_value = jax.jit(get_value, static_argnums=(2,))


## Custom solvers 

def solve_model_jax(model, algorithm="OPI", **kwargs):
    """
    General purpose solver. 

    algorithm : OPI, VFI or HPI

    """

    # Set up
    params, sizes, arrays = model
    n_y, n_z = sizes
    v_init = jnp.zeros((n_y, n_z))

    # Solve
    print(f"Solving model via using {algorithm}.")
    match algorithm:
        case "OPI":
            solver = optimistic_policy_iteration
            args = (v_init, 
                lambda v, σ: T_σ(v, σ, params, sizes, arrays), 
                lambda v: get_greedy(v, params, sizes, arrays))
        case "HPI":
            solver = howard_policy_iteration
            args = (v_init, 
                lambda σ: get_value(σ, params, sizes, arrays), 
                lambda v: get_greedy(v, params, sizes, arrays))
        case "VFI":
            solver = value_function_iteration
            args = (v_init, 
                lambda v: T(v, params, sizes, arrays), 
                lambda v: get_greedy(v, params, sizes, arrays))
        case _:
            raise ValueError("Algorithm must be in {OPI, VFI, HPI}")

    qe.tic()
    v_star, σ_star = solver(*args, **kwargs)
    run_time = qe.toc()
    print(f"Solved model using {algorithm} in {run_time:.5f} seconds.")

    return v_star, σ_star


def test_timing_jax(model,
                    m_vals=range(1, 100, 20),
                    figname="jax_timing.pdf",
                    savefig=False):
    """
    Plot relative timing of different algorithms.

    """

    qe.tic()
    _, σ_pi = solve_model_jax(model, algorithm="HPI")
    hpi_time = qe.toc()

    qe.tic()
    _, σ_vfi = solve_model_jax(model, algorithm="VFI")
    vfi_time = qe.toc()

    error = jnp.max(jnp.abs(σ_vfi - σ_pi))
    if error:
        print("Warning: VFI policy deviated with max error {error}.")

    opi_times = []
    for m in m_vals:
        qe.tic()
        _, σ_opi = solve_model_jax(model, algorithm="OPI", m=m)
        opi_times.append(qe.toc())

        error = jnp.max(jnp.abs(σ_opi - σ_pi))
        if error:
            print("Warning: OPI policy deviated with max error {error}.")

    plot_timing(hpi_time, 
                vfi_time,
                opi_times,
                m_vals, 
                figname=figname,
                savefig=False)

    return hpi_time, vfi_time, opi_times


## Simulations and plots 

model = create_sdd_inventory_model_jax()

### Solve by VFI 

v_star, σ_star = solve_model_jax(model, algorithm="VFI")

### Solve by HPI

v_star, σ_star = solve_model_jax(model, algorithm="HPI")

### Solve by OPI

v_star, σ_star = solve_model_jax(model, algorithm="OPI")


plot_ts(model, σ_star, figname="jax_ts.pdf", savefig=False)

### Plot timing test

hpi_time, vfi_time, opi_times = test_timing_jax(model)

