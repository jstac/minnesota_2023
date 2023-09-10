#
#  Begin by adding the packages Interpolations, Optim
# (e.g., https://www.educative.io/answers/how-to-install-julia-packages-via-command-line)
#

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

    # Add your code here

end

"Approximate lifetime value of policy σ."
function get_value(v_init, σ, m, model)

    # Compute and return T_σ^m v_init
    
end

"Optimistic policy iteration routine."
function opi(model; 
              tolerance=1e-6, 
              max_iter=1_000,
              m=20,
              print_step=10)
    v = model.v_init

    # Put your code here
    
    return σ, v
end


using PyPlot
using LaTeXStrings
fontsize=12

model = create_opt_savings_model()
(; β, u, R, w_grid, v_init) = model;

println("Solving via VFI. \n\n")
@timev σ_star, v_star = vfi(model);

# Uncomment next two lines and get it working

#println("\n\nSolving via HPI.\n\n")
#@timev σ_star_opi, v_star_hpi = opi(model);

# Plot both policies and check that they are close to eachother

# Record the runtime for (a) VFI and (b) OPI at various choices of m
# Plot them to illustrate how OPI compares to VFI 

