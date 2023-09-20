using Interpolations, Optim

"""
A function that provides linear interpolation with constant extrapolation
outside the bounds.

    x : a linear grid of interpolation points
    fx : the values of the function on the grid points

"""
function lin_interp(x, fx)
    return linear_interpolation(x, fx, 
                extrapolation_bc = Interpolations.Flat())
end

"""
Create an instance of the model, stored as a namedtuple.

"""
function create_opt_savings_model(; β=0.9,      # Discount factor
                                    γ=2.0,      # CRRA utility parameter
                                    R=1.2,      # Gross rate of return
                                    w_size=200, # Grid size for wealth
                                    w_max=10)
    w_grid = LinRange(1e-8, w_max, w_size)
    u(c) = c^(1 - γ) / (1 - γ)
    v_init = u.(w_grid)
    return (; β, u, R, w_grid, v_init)
end

"""
The Bellman operator

    (Tv)(w) = min_{0 ≤ c ≤ w} { u(c) + β v(R(w - c))}

"""
function T(v, model)
    (; β, u, R, w_grid) = model

    v_new = similar(v)
    v = lin_interp(w_grid, v)

    for (i, w) in enumerate(w_grid)
        result = maximize(c -> u(c) + β * v(R * (w - c)), 0.0, w) 
        v_new[i] = Optim.maximum(result)
    end

    return v_new
end


" Get a v-greedy policy "
function get_greedy(v, model)
    (; β, u, R, w_grid) = model

    σ = similar(v) 
    v = lin_interp(w_grid, v)

    for (i, w) in enumerate(w_grid)
        result = maximize(c -> u(c) + β * v(R * (w - c)), 0.0, w) 
        σ[i] = Optim.maximizer(result)
    end

    return σ
end



function vfi(model;
               tolerance=1e-6,    
               max_iter=10_000,  
               print_step=100)      

    k = 0
    error = tolerance + 1
    v = model.v_init

    while (error > tolerance) & (k <= max_iter)
        v_new = T(v, model)
        error = maximum(abs.(v_new - v))
        if k % print_step == 0
            println("Completed iteration $k with error $error.")
        end
        v = v_new
        k += 1
    end

    if error <= tolerance
        println("Terminated successfully in $k iterations.")
    else
        println("Warning: hit iteration bound.")
    end

    σ = get_greedy(v, model)
    return σ, v
end


"""
The policy operator

    (T_σ v)(w) =  u(σ(w)) + β v[R(w - σ(w))]

"""
function T_σ(v, σ, model)
    (; β, u, R, w_grid) = model

    v_new = similar(v)
    v = lin_interp(w_grid, v)
    for (i, w) in enumerate(w_grid)
        c = σ[i]
        v_new[i] = u(c) + β * v(R * (w - c))
    end
    return v_new
end

"Approximate lifetime value of policy σ."
function get_value(v_init, σ, m, model)
    v = v_init
    for i in 1:m
        v = T_σ(v, σ, model)
    end
    return v
end

"Optimistic policy iteration routine."
function opi(model; 
              tolerance=1e-6, 
              max_iter=1_000,
              m=20,
              print_step=10)
    v = model.v_init
    error = tolerance + 1
    k = 1
    while error > tolerance && k < max_iter
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(v, σ, m, model)
        error = maximum(abs.(v - last_v))
        if k % print_step == 0
            println("Completed iteration $k with error $error.")
        end
        k += 1
    end
    return get_greedy(v, model), v
end


using PyPlot
using LaTeXStrings
fontsize=12

model = create_opt_savings_model()
(; β, u, R, w_grid, v_init) = model;

println("Solving via VFI. \n\n")
@timev σ_star, v_star = vfi(model);

println("\n\nSolving via HPI.\n\n")
@timev σ_star_opi, v_star_hpi = opi(model);

# Plot both policies
fig, ax = plt.subplots()
ax.plot(w_grid, w_grid, "k--", label=L"45")
ax.plot(w_grid, σ_star, lw=2, label="VFI")
ax.plot(w_grid, σ_star_opi, "--", lw=2, label="HPI")
ax.legend(fontsize=fontsize)
plt.show()


m_vals=collect(range(1, 600, step=20))

println("Running value function iteration.")
vfi_time = @elapsed σ, v = vfi(model)
println("VFI completed in $vfi_time seconds.")
opi_times = []
for m in m_vals
    println("Running optimistic policy iteration with m=$m.")
    opi_time = @elapsed σ_opi = opi(model, m=m)
    println("OPI with m=$m completed in $opi_time seconds.")
    push!(opi_times, opi_time)
end
fig, ax = plt.subplots()
ax.plot(m_vals, fill(vfi_time, length(m_vals)), 
        lw=2, label="value function iteration")
ax.plot(m_vals, opi_times, 
        lw=2, label="optimistic policy iteration")
ax.legend(fontsize=fontsize, frameon=false)
ax.set_xlabel(L"m", fontsize=fontsize)
ax.set_ylabel("time", fontsize=fontsize)
plt.show()


