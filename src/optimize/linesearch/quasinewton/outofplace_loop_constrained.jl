function minimize_constrained(objective::ObjWrapper, x0,
    scheme::QuasiNewton,
    options=OptOptions())

    minimize_constrained(objective, (x0, nothing), (scheme, Backtracking()), options)
end

function minimize_constrained(objective::ObjWrapper, x0,
    approach::Tuple, options=OptOptions())

    minimize_constrained(objective, (x0, nothing), approach, options)
end
function minimize_constrained(objective::ObjWrapper, state0::Tuple,
    scheme::QuasiNewton,
    options=OptOptions(); lower = nothing, upper = nothing)
    minimize_constrained(objective, state0, (scheme, Backtracking()), options, lower = lower, upper = upper)
end

function minimize_constrained(objective::T1, state0::Tuple, approach::Tuple{<:Any, <:LineSearch},
    options::OptOptions=OptOptions(),
    linesearch::T2 = Backtracking(); lower = nothing, upper = nothing
    ) where {T1<:ObjWrapper, T2}

    if lower === nothing && upper === nothing
        return minimize(objective, state0, approach, options, linesearch)
    elseif lower === nothing && upper !== nothing
        error()
    elseif lower !== nothing && upper === nothing
        error()
    end


    x0, B0 = state0
    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)
    # first iteration
    clamp.(x, lower, upper) == x || error("Initial guess not in the feasible region")
    x, z, fz, ∇fz, B, is_converged = iterate_constrained(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, lower, upper)

    iter = 0
    while iter <= options.maxiter && !is_converged
        iter += 1

        # take a step and update approximation
        x, z, fz, ∇fz, B, is_converged = iterate_constrained(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, lower, upper, false)
        if norm(x.-z, Inf) ≤ T(1e-16)
            break
        end
    end

    return z, fz, ∇fz, iter
end


initialh(x::StaticVector{P,T}) where {P,T} = SMatrix{P,P,T}(I)
initialh(x::AbstractVector) = [i == j ? 1. : 0. for i in 1:size(x, 1), j in 1:size(x, 1)]

function iterate_constrained(x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach, objective, options, lower, upper, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach
    epsg = 1e-8
    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    ## Update the active set here

    function cdiag(x, c, i)
        if c
            return x
        else
            if i == 1.
                return 1.
            else
                return 0.
            end
        end
    end
    isbinding(i, j) = i & j

    # Binding set 1
    # true if free

    s1l = .!( ((x .<= lower .+ epsg) .& (∇fx .>= 0)) .| ((x .>= upper .- epsg) .& (∇fx .<= 0)) )
    Ix = initialh(x) # An identity matrix of correct type
    #=
    # Binding set 2
    binding = isbinding.(s1l, s1l')
    Hbar = cdiag.(B, binding, Ix)

    Sbargrad = find_direction(Hbar, ∇fx, scheme) # solve Bd = -∇fx
    s2l = .!( ((x .<= lower) .& (Sbargrad .> 0)) .| ((x .>= upper) .& (Sbargrad .< 0)) )
    =#
    sl = s1l #.& s2l
    binding = isbinding.(sl, sl')
    Hhat = cdiag.(B, binding, Ix)

    # set jxc to 0 if var in binding set
    ∇fxc = ∇fx .* sl


    ## The gradient needs to be zeroed out for the inactive region

    ## The hessian needs to be adapted to ignore the inactive region

    # Update current gradient and calculate the search direction
    d = find_direction(Hhat, ∇fxc, scheme) # solve Bd = -∇fx
    φ = LineObjective(objective, x, d, fx, dot(∇fxc, d))
    # # Perform line search along d

    # Should the line search project at each step?
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s
    z = clamp.(z, lower, upper)
    s = @. z - x

    # Update approximation
    fz, ∇fz, B = update_obj(objective, d, s, ∇fx, z, ∇fz, B, scheme, is_first)
    ∇fzc = ∇fz .* sl
    # Check for gradient convergence
    is_converged = converged(z, ∇fzc, options.g_tol)
    #@show sl ∇fz z s

    return x, z, fz, ∇fz, B, is_converged
end
