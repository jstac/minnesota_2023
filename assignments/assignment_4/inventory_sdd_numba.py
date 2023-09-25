"""

Solve the inventory management problem using Python/Numba.

John Stachurski
Sun 13 Aug 2023 04:25:41

"""

from model import *
from dp_algos import *
from plot_code import *

f = njit(f) # use numba to JIT-compile f

## Operators and functions 

@njit
def B(y, i_z, a, v, model):
    """
    B(y, a, v) = r(y, a) + β(z) Σ_{y′, z′} v(y′, z′) R(y, a, y′) Q(z, z′)

    """
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    β = z_vals[i_z]
    cv = 0.0
    for i_zp in range(len(z_vals)):
        for yp in y_vals:
            cv += v[yp, i_zp] * R[y, a, yp] * Q[i_z, i_zp]
    return r[y, a] + β * cv

@njit
def T(v, model):
    "The Bellman operator."
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = model.sizes
    new_v = np.empty_like(v)
    for i_z in range(n_z):
        for y in y_vals:
            Γy = range(K - y + 1)
            B_vals = [B(y, i_z, a, v, model) for a in Γy]
            new_v[y, i_z] = max(B_vals)
    return new_v


@njit
def T_σ(v, σ, model):
    "The policy operator."
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = model.sizes
    new_v = np.empty_like(v)
    for i_z in range(n_z):
        for y in y_vals:
            new_v[y, i_z] = B(y, i_z, σ[y, i_z], v, model) 
    return new_v

@njit
def get_greedy(v, model):
    "Get a v-greedy policy.  Returns indices of choices."
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = model.sizes
    σ_star = np.zeros((len(y_vals), n_z), dtype=int32)
    for i_z in range(n_z):
        for y in y_vals:
            max_val = -np.inf
            for a in range(K - y + 1):
                current_val = B(y, i_z, a, v, model)
                if current_val > max_val:
                    maximizer = a
                    max_val = current_val
            σ_star[y, i_z] = maximizer
    return σ_star

@njit
def get_value(σ, model):
    "Get the value v_σ of policy σ."
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = model.sizes
    n = n_z * n_y
    # Build L_σ and r_σ as multi-index arrays
    L_σ = np.zeros((n_y, n_z, n_y, n_z))
    r_σ = np.zeros((n_y, n_z))
    for y in y_vals:
        for i_z in range(n_z):
            a = σ[y, i_z]
            β = z_vals[i_z]
            r_σ[y, i_z] = r[y, a]
            for yp in y_vals:
                for i_zp in range(n_z):
                    L_σ[y, i_z, yp, i_zp] = β * R[y, a, yp] * Q[i_z, i_zp]
    # Reshape for matrix algebra
    L_σ = np.reshape(L_σ, (n, n))
    r_σ = np.reshape(r_σ, n)
    # Apply matrix operations --- solve for the value of σ 
    I = np.identity(n)
    v_σ = np.linalg.solve(I - L_σ,  r_σ)
    # Return as multi-index array
    return np.reshape(v_σ, (n_y, n_z))


## Custom solvers 

def solve_model_numba(model, algorithm="OPI", **kwargs):
    """
    General purpose solver. 

    algorithm : OPI, VFI or HPI

    """

    # Set up
    n_y, n_z = model.sizes
    v_init = np.zeros((n_y, n_z))

    # Solve
    print(f"Solving model using {algorithm}.")
    match algorithm:
        case "OPI":
            solver = optimistic_policy_iteration
            args = (v_init, 
                lambda v, σ: T_σ(v, σ, model), 
                lambda v: get_greedy(v, model))
        case "HPI":
            solver = howard_policy_iteration
            args = (v_init, 
                lambda σ: get_value(σ, model), 
                lambda v: get_greedy(v, model))
        case "VFI":
            solver = value_function_iteration
            args = (v_init, 
                lambda v: T(v, model), 
                lambda v: get_greedy(v, model))
        case _:
            raise ValueError("Algorithm must be in {OPI, VFI, HPI}")

    qe.tic()
    v_star, σ_star = solver(*args, **kwargs)
    run_time = qe.toc()
    print(f"Solved model using {algorithm} in {run_time:.5f} seconds.")

    return v_star, σ_star


def test_timing_numba(model,
                      m_vals=range(1, 100, 20),
                      figname="numba_timing.pdf",
                      savefig=False):
    """
    Plot relative timing of different algorithms.

    """

    qe.tic()
    _, σ_pi = solve_model_numba(model, algorithm="HPI")
    hpi_time = qe.toc()

    qe.tic()
    _, σ_vfi = solve_model_numba(model, algorithm="VFI")
    vfi_time = qe.toc()

    error = np.max(np.abs(σ_vfi - σ_pi))
    if error:
        print("Warning: VFI policy deviated with max error {error}.")

    opi_times = []
    for m in m_vals:
        qe.tic()
        _, σ_opi = solve_model_numba(model, algorithm="OPI", m=m)
        opi_times.append(qe.toc())

        error = np.max(np.abs(σ_opi - σ_pi))
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

model = create_sdd_inventory_model()

### Solve by VFI

v_star, σ_star = solve_model_numba(model, algorithm="VFI")

### Solve by HPI

v_star, σ_star = solve_model_numba(model, algorithm="HPI")

### Solve by OPI

v_star, σ_star = solve_model_numba(model, algorithm="OPI")


plot_ts(model, σ_star, figname="numba_ts.pdf", savefig=False)

### Test timing

hpi_time, vfi_time, opi_times = test_timing_numba(model)

