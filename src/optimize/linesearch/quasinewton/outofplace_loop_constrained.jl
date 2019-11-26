initialh(x::StaticVector{P,T}) where {P,T} = SMatrix{P,P,T}(I)
initialh(x::AbstractVector) = [i == j ? 1. : 0. for i in 1:size(x, 1), j in 1:size(x, 1)]

function iterate(x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach, objective, options, is_first=nothing) where Tf
    if _manifold(objective) isa Box
        return _iterate_constrained(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, is_first)
    else
        return _iterate(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, is_first)
    end
end

function _iterate_constrained(x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach, objective, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search

    scheme, linesearch = approach
    epsg = 1e-8
    m = _manifold(objective)
    lower = m.lower
    upper = m.upper
    if is_first === nothing
        clamp.(x, lower, upper) == x || error("Initial guess not in the feasible region")
    end

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
    lower_e = lower .+ epsg
    upper_e = upper .- epsg
    sl = .!( ((x .<= lower_e) .& (∇fx .>= 0)) .| ((x .>= upper_e) .& (∇fx .<= 0)) )
    Ix = initialh(x) # An identity matrix of correct type

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
