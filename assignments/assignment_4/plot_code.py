"""
Simulations and plots.

John Stachurski
Sun 13 Aug 2023 04:25:41

"""


from model import *
import matplotlib.pyplot as plt


def sim_inventories(model, σ_star, ts_length, Y_init=0, seed=500):
    """
        Simulate inventory dynamics and interest rates given an 
        optimal policy σ_star.
    """
    # Set up
    np.random.seed(seed)
    K, c, κ, p = model.params
    r, R, y_vals, z_vals, Q = model.arrays

    # Generate Markov chain for discount factor
    z_mc = qe.MarkovChain(Q, z_vals)
    i_z = z_mc.simulate_indices(ts_length, init=1, random_state=seed)

    # Generate corresponding inventory series
    Y = np.zeros(ts_length, dtype=int)
    Y[0] = Y_init
    for t in range(ts_length - 1):
        D = np.random.geometric(p) - 1
        a = σ_star[Y[t], i_z[t]] 
        Y[t+1] = f(Y[t],  a,  D)

    # Return both series
    return Y, z_vals[i_z]


def plot_ts(model,
            σ_star,
            ts_length=400,
            fontsize=12, 
            figname="ts.pdf",
            savefig=False):
    """
        Solve model, plot a time series of inventory and interest rates.

    """

    # Obtain inventory and discount factor series
    Y, Z = sim_inventories(model, σ_star, ts_length)
    r = (1 / Z) - 1 # calculate interest rate from discount factors

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))
    ax = axes[0]
    ax.plot(Y, label="inventory", alpha=0.7)
    ax.set_xlabel("time", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_ylim(0, np.max(Y)+3)
    ax = axes[1]
    ax.plot(r, label="$r_t$", alpha=0.7)
    ax.set_xlabel("$t$", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=False)
    plt.tight_layout()
    plt.show()
    if savefig:
        fig.savefig(figname)



def plot_timing(hpi_time, 
                vfi_time,
                opi_times,
                m_vals, 
                figname="timing.pdf",
                fontsize=12,
                savefig=False):
    """
    Plot relative timing of different algorithms.

    """
    fig, ax = plt.subplots(figsize=(9, 5.2))

    y_values = (np.full(len(m_vals), vfi_time), 
                np.full(len(m_vals), hpi_time),
                opi_times)
    labels = "VFI", "HPI", "OPI"

    for y_vals, label in zip(y_values, labels):
        ax.plot(m_vals, y_vals, lw=2, label=label)

    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_xlabel("$m$", fontsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    plt.show()

    if savefig:
        fig.savefig(figname)


