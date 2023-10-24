"""

Solve the inventory management problem using Python/Numpy

John Stachurski
Sun 13 Aug 2023 04:25:41

"""

from model import *
from dp_algos import *
from plot_code import *

## Operators and functions 

def B(v, model):
    """
    B(y, z, a, v) = r(y, a) + β(z) Σ_{y′, z′} v(y′, z′) R(y, a, y′) Q(z, z′)

    """
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = len(y_vals), len(z_vals)

    # broadcasting over    (y,   z,   a)
    r = np.reshape(r,      (n_y, 1,   n_y))
    β = np.reshape(z_vals, (1,   n_z, 1))

    # broadcasting over (y,   z,   a,   yp,   zp)
    R = np.reshape(R,   (n_y, 1,   n_y, n_y,  1))
    Q = np.reshape(Q,   (1,   n_z, 1,   1,    n_z))
    v = np.reshape(v,   (1,   1,   1,   n_y,  n_z))

    Ev = np.sum(v * R * Q, axis=(3, 4))
    return r + β * Ev


def compute_r_σ(σ, model):
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = len(y_vals), len(z_vals)
    z_idx = range(n_z)
    # Create r_σ with indices (y,   z)
    y = np.reshape(y_vals,    (n_y, 1))
    z = np.reshape(z_idx,     (1,   n_z))
    r_σ = r[y, σ[y, z]]
    return r_σ

def compute_R_σ(σ, model):
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = len(y_vals), len(z_vals)
    z_idx = range(n_z)
    # Create R_σ with indices (y,   z,   yp)
    y  = np.reshape(y_vals,   (n_y, 1,   1))
    yp = np.reshape(y_vals,   (1,   1,   n_y))
    z  = np.reshape(z_idx,    (1,   n_z, 1))
    R_σ = R[y, σ[y, z], yp]
    return R_σ 

def compute_L_σ(σ, model):
    r, R, y_vals, z_vals, Q = model.arrays
    n_y, n_z = len(y_vals), len(z_vals)
    R_σ = compute_R_σ(σ, model)
    # Create L_σ with indices (y,   z,   yp,  zp)
    β   = np.reshape(z_vals,  (1,   n_z, 1,   1))
    R_σ = np.reshape(R_σ,     (n_y, n_z, n_y, 1))
    Q   = np.reshape(Q,       (1,   n_z, 1,   n_z))
    L_σ = β * R_σ * Q
    return L_σ


def T(v, model):
    "The Bellman operator."
    return np.max(B(v, model), axis=2)


def get_greedy(v, model):
    "Get a v-greedy policy."
    return np.argmax(B(v, model), axis=2)


def T_σ(v, σ, model):
    "The policy operator."
    r, R, y_vals, z_vals, Q = model.arrays
    n_z, n_y = len(z_vals), len(y_vals)
    r_σ = compute_r_σ(σ, model)
    L_σ = compute_L_σ(σ, model)
    v = np.reshape(v, (1, 1, n_y, n_z))
    return r_σ + np.sum(L_σ * v, axis=(2, 3))


def get_value(σ, model):
    r, R, y_vals, z_vals, Q = model.arrays
    n_z, n_y = len(z_vals), len(y_vals)
    r_σ = compute_r_σ(σ, model)
    L_σ = compute_L_σ(σ, model)
    # Reshape for matrix algebra
    n = n_z * n_y
    L_σ = np.reshape(L_σ, (n, n))
    r_σ = np.reshape(r_σ, n)
    # Apply matrix operations --- solve for the value of σ 
    I = np.identity(n)
    v_σ = np.linalg.solve(I - L_σ,  r_σ)
    # Return as multi-index array
    return np.reshape(v_σ, (n_y, n_z))


## Custom solvers 

def solve_model_numpy(model, algorithm="OPI", **kwargs):
    """
    General purpose solver. 

    algorithm : OPI, VFI or HPI

    """

    # Set up
    n_y, n_z = model.sizes
    v_init = np.zeros((n_y, n_z))

    # Solve
    print(f"Solving model via using {algorithm}.")
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


def test_timing_numpy(model,
                      m_vals=range(1, 100, 20),
                      figname="numpy_timing.pdf",
                      savefig=False):
    """
    Plot relative timing of different algorithms.

    """

    qe.tic()
    _, σ_pi = solve_model_numpy(model, algorithm="HPI")
    hpi_time = qe.toc()

    qe.tic()
    _, σ_vfi = solve_model_numpy(model, algorithm="VFI")
    vfi_time = qe.toc()

    error = np.max(np.abs(σ_vfi - σ_pi))
    if error:
        print("Warning: VFI policy deviated with max error {error}.")

    opi_times = []
    for m in m_vals:
        qe.tic()
        _, σ_opi = solve_model_numpy(model, algorithm="OPI", m=m)
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

v_star, σ_star = solve_model_numpy(model, algorithm="VFI")

### Solve by HPI

v_star, σ_star = solve_model_numpy(model, algorithm="HPI")

### Solve by OPI

v_star, σ_star = solve_model_numpy(model, algorithm="OPI")


plot_ts(model, σ_star, figname="numpy_ts.pdf", savefig=False)

### Test timing

hpi_time, vfi_time, opi_times = test_timing_numpy(model)

