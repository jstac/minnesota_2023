# -*- coding: utf-8 -*-
"""

Solve the inventory management problem using Julia.

The discount factor takes the form β_t = Z_t, where (Z_t) is 
a discretization of the Gaussian AR(1) process 

    Z_t = ρ Z_{t-1} + b + ν W_t.

John Stachurski
Sun 13 Aug 2023 04:25:41

"""

include("s_approx.jl")
using LinearAlgebra, Random, Distributions, QuantEcon


## Primitives

f(y, a, d) = max(y - d, 0) + a  # Inventory update

function create_sdd_inventory_model(; ρ=0.98, 
                                      ν=0.002, 
                                      n_z=25, 
                                      b=0.97, 
                                      K=100, 
                                      c=0.2, 
                                      κ=0.8, 
                                      p=0.6, 
                                      d_max=100)  # truncate demand shock

    ϕ(d) = (1 - p)^d * p                      # demand pdf
    d_vals = collect(0:d_max)
    ϕ_vals = ϕ.(d_vals)
    y_vals = collect(0:K)                     # inventory levels
    n_y = length(y_vals)
    mc = tauchen(n_z, ρ, ν)
    z_vals, Q = mc.state_values .+ b, mc.p

    # test spectral radius condition
    ρL = maximum(abs.(eigvals(z_vals .* Q)))     
    @assert  ρL < 1 "Error: ρ(L) ≥ 1."    

    R = zeros(n_y, n_y, n_y)
    for (i_y, y) in enumerate(y_vals)
        for (i_y′, y′) in enumerate(y_vals)
            for (i_a, a) in enumerate(0:(K - y))
                hits = f.(y, a, d_vals) .== y′
                R[i_y, i_a, i_y′] = dot(hits, ϕ_vals)
            end
        end
    end

    r = fill(-Inf, n_y, n_y)
    for (i_y, y) in enumerate(y_vals)
        for (i_a, a) in enumerate(0:(K - y))
                cost = c * a + κ * (a > 0)
                r[i_y, i_a] = dot(min.(y, d_vals),  ϕ_vals) - cost
        end
    end

    return (; K, c, κ, p, r, R, y_vals, z_vals, Q)
end


## Operators and Functions


"""
The function 

    B(y, z, a, v) = r(y, a) + β(z) Σ_{y′, z′} v(y′, z′) R(y, a, y′) Q(z, z′)

"""
function B(i_y, i_z, i_a, v, model; d_max=100)
    (; K, c, κ, p, r, R, y_vals, z_vals, Q) = model
    β = z_vals[i_z]
    cv = 0.0
    for i_z′ in eachindex(z_vals)
        for i_y′ in eachindex(y_vals)
            cv += v[i_y′, i_z′] * R[i_y, i_a, i_y′] * Q[i_z, i_z′]
        end
    end
    return r[i_y, i_a] + β * cv
end

"The Bellman operator."
function T(v, model)
    (; K, c, κ, p, r, R, y_vals, z_vals, Q) = model
    new_v = similar(v)
    for i_z in eachindex(z_vals)
        for (i_y, y) in enumerate(y_vals)
            Γy = 1:(K - y + 1)
            new_v[i_y, i_z], _ = findmax(B(i_y, i_z, i_a, v, model) for i_a in Γy)
        end
    end
    return new_v
end

"The policy operator."
function T_σ(v, σ, model)
    (; K, c, κ, p, r, R, y_vals, z_vals, Q) = model
    new_v = similar(v)
    for (i_z, z) in enumerate(z_vals)
        for (i_y, y) in enumerate(y_vals)
            new_v[i_y, i_z] = B(i_y, i_z, σ[i_y, i_z], v, model) 
        end
    end
    return new_v
end


"Get a v-greedy policy.  Returns indices of choices."
function get_greedy(v, model)
    (; K, c, κ, p, r, R, y_vals, z_vals, Q) = model
    n_z = length(z_vals)
    σ_star = zeros(Int32, K+1, n_z)
    for (i_z, z) in enumerate(z_vals)
        for (i_y, y) in enumerate(y_vals)
            Γy = 1:(K - y + 1)
            _, i_a = findmax(B(i_y, i_z, i_a, v, model) for i_a in Γy)
            σ_star[i_y, i_z] = Γy[i_a]
        end
    end
    return σ_star
end


"Approximate lifetime value of policy σ."
function get_value_approx(v_init, σ, m, model)
    v = v_init
    for i in 1:m
        v = T_σ(v, σ, model)
    end
    return v
end


"Get the value v_σ of policy σ."
function get_value(σ, model)
    (; K, c, κ, p, r, R, y_vals, z_vals, Q) = model
    n_z, n_y = length(z_vals), length(y_vals)
    n = n_z * n_y
    # Build L_σ and r_σ as multi-index arrays
    L_σ = zeros(n_y, n_z, n_y, n_z)
    r_σ = zeros(n_y, n_z)
    for i_y in 1:n_y
        for i_z in 1:n_z 
            a = σ[i_y, i_z]
            β = z_vals[i_z]
            r_σ[i_y, i_z] = r[i_y, a]
            for i_yp in 1:n_y
                for i_zp in 1:n_z
                    L_σ[i_y, i_z, i_yp, i_zp] = β * R[i_y, a, i_yp] * Q[i_z, i_zp]
                end
            end
        end
    end
    # Reshape for matrix algebra
    L_σ = reshape(L_σ, n, n)
    r_σ = reshape(r_σ, n)
    # Apply matrix operations --- solve for the value of σ 
    v_σ = (I - L_σ) \ r_σ
    # Return as multi-index array
    return reshape(v_σ, n_y, n_z)
end


"Use successive_approx to get v_star and then compute greedy."
function value_function_iteration(v_init, model)
    v_star = successive_approx(v -> T(v, model), v_init)
    σ_star = get_greedy(v_star, model)
    return v_star, σ_star
end


"Optimistic policy iteration routine."
function optimistic_policy_iteration(v_init, 
                                     model; 
                                     tolerance=1e-6, 
                                     max_iter=1_000,
                                     print_step=10,
                                     m=60)
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance && k < max_iter
        last_v = v
        σ = get_greedy(v, model)
        v = get_value_approx(v, σ, m, model)
        error = maximum(abs.(v - last_v))
        if k % print_step == 0
            println("Completed iteration $k with error $error.")
        end
        k += 1
    end
    return v, get_greedy(v, model)
end


function howard_policy_iteration(v_init, model)
    "Howard policy iteration routine."
    v_σ = v_init
    σ = get_greedy(v_σ, model)
    i, error = 0, 1.0
    while error > 0
        v_σ = get_value(σ, model)
        σ_new = get_greedy(v_σ, model)
        error = maximum(abs.(σ_new - σ))
        σ = σ_new
        i = i + 1
        println("Concluded loop $i with error $error.")
    end
    return v_σ, σ
end


## Simulations and Plots 

using PyPlot
using LaTeXStrings
PyPlot.matplotlib[:rc]("text", usetex=true) # allow tex rendering

# Create an instance of the model and solve it
println("Create model instance.")
@time model = create_sdd_inventory_model();

(; K, c, κ, p, r, R, y_vals, z_vals, Q) = model;
n_z = length(z_vals)
v_init = zeros(Float64, K+1, n_z);

println("Solving model via OPI.")
@time v_star, σ_star = optimistic_policy_iteration(v_init, model);

println("Solving model via VFI.")
@time v_star_vfi, σ_star_vfi = value_function_iteration(v_init, model);

println("Solving model via HPI.")
@time v_star_hpi, σ_star_hpi = howard_policy_iteration(v_init, model);


"Simulate given the optimal policy."
function sim_inventories(ts_length; X_init=0, seed=500)
    Random.seed!(seed) 
    z_mc = MarkovChain(Q, z_vals)
    i_z = simulate_indices(z_mc, ts_length, init=1)
    G = Geometric(p)
    X = zeros(Int32, ts_length)
    X[1] = X_init
    for t in 1:(ts_length-1)
        D′ = rand(G)
        x_index = X[t] + 1
        a = σ_star[x_index, i_z[t]] - 1
        X[t+1] = f(X[t],  a,  D′)
    end
    return X, z_vals[i_z]
end

function plot_ts(; ts_length=400,
                   fontsize=16, 
                   figname="../figures/inventory_sdd_ts.pdf",
                   savefig=false)
    X, Z = sim_inventories(ts_length)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5))

    ax = axes[1]
    ax.plot(X, label="inventory", alpha=0.7)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_ylim(0, maximum(X)+3)

    # calculate interest rate from discount factors
    r = (1 ./ Z) .- 1

    ax = axes[2]
    ax.plot(r, label=L"r_t", alpha=0.7)
    ax.set_xlabel(L"t", fontsize=fontsize)
    ax.legend(fontsize=fontsize, frameon=false)
    #ax.set_ylim(0, maximum(X)+8)

    plt.tight_layout()
    plt.show()
    if savefig == true
        fig.savefig(figname)
    end
end

function plot_timing(; m_vals=collect(range(1, 100, step=20)),
                       fontsize=12)

    println("Running Howard policy iteration.")
    hpi_time = @elapsed _ = howard_policy_iteration(v_init, model)
    println("HPI completed in $hpi_time seconds.")

    println("Running value function iteration.")
    vfi_time = @elapsed _ = value_function_iteration(v_init, model)
    println("VFI completed in $vfi_time seconds.")

    println("Starting Howard policy iteration.")
    opi_times = []
    for m in m_vals
        println("Running optimistic policy iteration with m=$m.")
        opi_time = @elapsed σ_opi = optimistic_policy_iteration(v_init, model, m=m)
        println("OPI with m=$m completed in $opi_time seconds.")
        push!(opi_times, opi_time)
    end

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(m_vals, fill(hpi_time, length(m_vals)), 
            lw=2, label="Howard policy iteration")
    ax.plot(m_vals, fill(vfi_time, length(m_vals)), 
            lw=2, label="value function iteration")
    ax.plot(m_vals, opi_times, lw=2, label="optimistic policy iteration")
    ax.legend(fontsize=fontsize, frameon=false)
    ax.set_xlabel(L"m", fontsize=fontsize)
    ax.set_ylabel("time", fontsize=fontsize)
    plt.show()

    return (hpi_time, vfi_time, opi_times)
end

hpi_time, vfi_time, opi_times = plot_timing()
