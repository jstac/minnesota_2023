import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import njit
import jax
import jax.numpy as jnp
from scipy.stats import beta

jax.config.update("jax_enable_x64", True)


# == densities == #

def f_star(x):
    " Stationary density. "
    return 1 / (np.pi * np.sqrt(x * (1-x)))


def Pf(x):
    " Pf when f is uniform. "
    return (1/2) * 1 / np.sqrt(1-x)

def P2f(x):
    " P^2 f when f is uniform. "
    z = np.sqrt(1-x)
    return 1/(4 * z) * (Pf((1-z)/2) + Pf((1+z)/2))


# == state space updates  == #

@njit
def g(x):
    return 4 * x * (1-x)


@njit
def gen_trajectory(x0, n=100):
    x = np.empty(n)
    x[0] = x0
    for t in range(n-1):
        x[t+1] = g(x[t])

    return x


@njit
def g_shifter(x0, n=20):
    " Shift x0 forward n periods. "
    x = x0
    for i in range(n):
        x = g(x)
    return x


def g_shifter_loop(x0, n):
    """
    Shift x0 forward n periods.

    """
    x = x0
    for _ in range(n):
        x = 4.0 * x * (1 - x)
    return x


def g_shifter_scan(x0, n):
    """
    Shift x0 forward n periods.

    """
    def update(x, _):
        return 4.0 * x * (1 - x), _

    x, _ = jax.lax.scan(update, x0, None, length=n)
    return x


g_shifter_scan = jax.jit(g_shifter_scan, static_argnums=(1,))

g_shifter_vmap = jax.vmap(g_shifter_loop, in_axes=(0, None))


def test_speed(m=10_000_000, n=100):

    x0 = np.random.rand(m)

    qe.tic()
    numba_out = g_shifter(x0, n)
    numba_time = qe.toc()

    x0 = jax.device_put(x0)

    qe.tic()
    jax_out = g_shifter_scan(x0, n).block_until_ready()
    jax_time_with_compile = qe.toc()

    qe.tic()
    jax_out = g_shifter_scan(x0, n).block_until_ready()
    jax_time = qe.toc()

    qe.tic()
    jax_out = g_shifter_vmap(x0, n).block_until_ready()
    jax_vmap_time_with_compile = qe.toc()

    qe.tic()
    jax_out = g_shifter_vmap(x0, n).block_until_ready()
    jax_vmap_time = qe.toc()
    qe.tic()

    print(f"Compile time without vmap = {jax_time_with_compile - jax_time}") 
    print(f"Compile time with vmap = {jax_vmap_time_with_compile - jax_vmap_time}") 
    print(f"Sped gain from JAX without vmap = {numba_time / jax_time}")
    print(f"Sped gain from JAX with vmap = {numba_time / jax_vmap_time}")


# == Plotting code == #

kwargs = dict(density=True, bins=150, alpha=0.5)
x_grid = np.linspace(0.0001, 0.9999, 200)

def plot_g():
    fig, ax = plt.subplots()
    xvec = np.linspace(0, 1, 100)
    ax.plot(xvec, [g(x) for x in xvec], label="$g(x)= 4x(1-x)$")
    ax.plot(xvec, xvec, 'k-', lw=0.5, label='45 degrees')
    ax.set_xticks((0, 1))
    ax.set_yticks((0, 1))
    ax.legend()
    plt.show()


def plot_gn(n=20):
    fig, ax = plt.subplots()
    xvec = np.linspace(0, 1, 100)
    ax.plot(xvec, [g_shifter(x, n) for x in xvec], label="$g^n$")
    ax.set_xticks((0, 1))
    ax.set_yticks((0, 1))
    ax.set_xlabel("initial condition $x_0$", fontsize=14)
    ax.set_title(f"$g^n(x_0)$ when $n = {n}$", fontsize=14)
    ax.legend()
    plt.show()


def plot_traj():
    fig, ax = plt.subplots()
    ax.plot(gen_trajectory(0.3, n=150), 'o-', alpha=0.6)
    ax.set_xlabel("$t$", fontsize=14)
    ax.set_ylabel("$x_t$", fontsize=14)
    ax.set_yticks((0, 1))
    ax.set_xticks((0, 50, 100, 150))
    plt.show()


def plot_traj_hist():
    fig, ax = plt.subplots()
    ax.hist(gen_trajectory(0.3, n=100_000), **kwargs, label='observations')
    ax.set_xlabel("state", fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_ylim(0, 6)
    ax.legend()
    plt.show()


def plot_traj_hist_and_stationary():
    fig, ax = plt.subplots()
    ax.hist(gen_trajectory(0.3, n=250_000), **kwargs, label='observations')
    y_vals = f_star(x_grid)
    ax.plot(x_grid, y_vals, label='$\psi^*$')
    ax.set_xlabel("state", fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_ylim(0, 6)
    ax.legend()
    plt.show()



def plot_dist_images():

    def plot_data(ax, y_data, hist_data, label):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 4.05)
        ax.set_yticks([])
        ax.set_xticks((0, 1))
        ax.set_title("time zero distribution")
        ax.plot(x_grid, y_data, label=label)
        ax.hist(hist_data, **kwargs, color='orange')
        ax.legend(loc='upper center')

    sample_size = 100_000

    fig, axes = plt.subplots(3, 2, figsize=(10, 6.6))
    
    beta_param_set = (2, 2), (2, 5), (5, 2)
    for i, beta_params in enumerate(beta_param_set):
        f0 = np.random.beta(*beta_params, size=sample_size)
        fT = g_shifter(f0)
        q = beta(*beta_params)
        y_data = q.pdf(x_grid)
        plot_data(axes[i, 0], q.pdf(x_grid), f0, "$\psi_0$")
        plot_data(axes[i, 1], f_star(x_grid), fT, "$\psi^*$")

    fig.tight_layout()
    plt.show()


