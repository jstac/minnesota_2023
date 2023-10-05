import quantecon as qe
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import namedtuple
import jax
import jax.numpy as jnp

# Use 64 bit floats with JAX in order to increase precision.

jax.config.update("jax_enable_x64", True)



## == Solvers == ##


default_tolerance = 1e-9
default_max_iter = int(1e6)

def successive_approx(f,
                      x_init,
                      tol=default_tolerance,
                      max_iter=default_max_iter,
                      verbose=True,
                      print_skip=1000):

    "Uses successive approximation on f."

    if verbose:
        print("Beginning iteration\n\n")

    current_iter = 0
    x = x_init
    error = tol + 1
    while error > tol and current_iter < max_iter:
        x_new = f(x)
        error = jnp.max(jnp.abs(x_new - x))
        if verbose and current_iter % print_skip == 0:
            print("iter = {}, error = {}".format(current_iter, error))
        current_iter += 1
        x = x_new

    if current_iter == max_iter:
        print(f"Warning: Hit maximum iteration number {max_iter}")
    else:
        if verbose:
            print(f"Iteration converged after {current_iter} iterations")

    return x


def newton_solver(f, 
                  x_init, 
                  tol=default_tolerance, 
                  max_iter=default_max_iter,
                  verbose=True,
                  bicgstab_atol=1e-6,
                  print_skip=1):
    """
    Apply Newton's algorithm to find a fixed point of f. 

    """

    # We use a root-finding operation on g(x) = f(x) - x.
    g = lambda x: f(x) - x

    # If g(x) = 0 then we have found a fixed point.
    #
    # Thus we need to iterate with the map
    #
    #    Q(x) = x - J_g(x)^{-1} g(x)
    #

    @jax.jit
    def Q(x):
        # First we define J_g(x) as an operator.  In particular, we 
        # define the map v -> J_g(x) v 
        jac_x_prod = lambda v: jax.jvp(g, (x,), (v,))[1]
        # Next we compute b = J(x)^{-1} g(x) using an iterative algorithm
        b = jax.scipy.sparse.linalg.bicgstab(
                jac_x_prod, g(x), 
                atol=bicgstab_atol)[0]
        # Finally we return x - b
        return x - b

    return successive_approx(Q, x_init, tol, max_iter, verbose, print_skip)



## == Model == ##


SVModel = namedtuple('SVModel',
                        ('P', 'h_grid',
                         'Q', 'z_grid',
                         'β', 'γ', 'ψ', 'bar_σ', 'μ_c'))

def create_sv_model(β=0.99,        # discount factor
                    γ=8.89,
                    ψ=1.97,
                    bar_σ=0.5,     # volatility scaling parameter
                    μ_c=0.001,     # mean growth of consumtion
                    n_h=12,        # size of state space for h
                    ρ_c=0.9,       # persistence parameter for h
                    σ_c=0.01,      # volatility parameter for h
                    n_z=60,        # size of state space for z
                    ρ_z=0.95,      # persistence parameter for z
                    σ_z=0.01):     # persistence parameter for z

    mc = qe.tauchen(n_h, ρ_c, σ_c)
    h_grid = mc.state_values
    P = mc.P

    mc = qe.tauchen(n_z, ρ_z, σ_z)
    z_grid = mc.state_values
    Q = mc.P

    return SVModel(P=P, h_grid=h_grid,
                   Q=Q, z_grid=z_grid,
                   β=β, γ=γ, ψ=ψ, bar_σ=bar_σ, μ_c=μ_c)



# Compute Tw = 1 + β * Kwθ**(1/θ)

def T(w, sv_model, shapes):
    """
    Implement the operator T via JAX.

    T takes an array w of shape (n_h, n_z) and returns a new
    array of the same shape.

    """
    # Set up
    P, h_grid, Q, z_grid, β, γ, ψ, bar_σ, μ_c = sv_model
    n_h, n_z = shapes
    θ = (1 - γ) / (1 - 1/ψ)

    # Broadcast over          (n_h, n_z, n_h', n_z')
    w = jnp.reshape(w,        (1,   1,   n_h,  n_z))
    h = jnp.reshape(h_grid,   (n_h, 1,   1,    1))
    z = jnp.reshape(z_grid,   (1,   n_z, 1,    1))
    P = jnp.reshape(P,        (n_h, 1,   n_h,  1))
    Q = jnp.reshape(Q,        (1,   n_z, 1,    n_z))

    a = jnp.exp((1 - γ ) * μ_c + z + (bar_σ * jnp.exp(h))**2 / 2)

    # Compute K(w^θ)
    Kwθ = jnp.sum(a * P * Q * w**θ , axis=(2, 3))

    # Define and return Tw
    Tw = 1 + β * Kwθ**(1/θ)
    return Tw




## == Solution and plots == ##

T = jax.jit(T, static_argnums=(2, ))

sv_model = create_sv_model()
P, h_grid, Q, z_grid, β, γ, ψ, bar_σ, μ_c = sv_model
shapes = len(h_grid), len(z_grid)
w_init = jnp.ones(shapes)

qe.tic()
w_successive = successive_approx(lambda w: T(w, sv_model, shapes), w_init)
successive_approx_time = qe.toc()

qe.tic()
w_newton = newton_solver(lambda w: T(w, sv_model, shapes), w_init)
newton_time = qe.toc()

error = jnp.max(jnp.abs(w_successive - w_newton))
relative_time = newton_time / successive_approx_time

print(f"Max absolute error = {error}")
print(f"Newton time / successive approx time = {relative_time}")

fig, ax = plt.subplots()
cs = ax.contourf(h_grid,
                 z_grid,
                 w_newton.T,
                 cmap=cm.viridis,
                 alpha=0.6)
ax.set_xlabel('$h$', fontsize=12)
ax.set_ylabel('$z$', fontsize=12)
cbar = fig.colorbar(cs)
ax.set_title('wealth-consumption ratio')
plt.savefig("foo.pdf")
plt.show()

