function minimize(objective::ObjWrapper, x0,
                   scheme::QuasiNewton,
                   options=OptOptions())

    minimize(objective, (x0, nothing), (scheme, Backtracking()), options)
end
function minimize(objective::ObjWrapper, x0,
                   approach::Tuple, options=OptOptions())

    minimize(objective, (x0, nothing), approach, options)
end
function minimize(objective::ObjWrapper, state0::Tuple,
                  scheme::QuasiNewton,
                  options=OptOptions())
    minimize(objective, state0, (scheme, Backtracking()), options)
end
function minimize(objective::T1, state0::Tuple, approach::Tuple{<:Any, <:LineSearch},
                  options::OptOptions=OptOptions(),
                  linesearch::T2 = Backtracking()
                  ) where {T1<:ObjWrapper, T2}

    x0, B0 = state0
    T = eltype(x0)
    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)
    # first iteration
    x, z, fz, ∇fz, B, is_converged = iterate(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options)

    iter = 0
    while iter <= options.maxiter && !is_converged
        iter += 1

        # take a step and update approximation
        x, z, fz, ∇fz, B, is_converged = iterate(x, fx, ∇fx, z, fz, ∇fz, B, approach, objective, options, false)
        if norm(x.-z, Inf) ≤ T(1e-16)
            break
        end
    end

    return z, fz, ∇fz, iter
end

function iterate(x, fx::Tf, ∇fx, z, fz, ∇fz, B, approach, objective, options, is_first=nothing) where Tf
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    # Move nexts into currs
    fx = fz
    x = copy(z)
    ∇fx = copy(∇fz)

    # Update current gradient and calculate the search direction
    d = find_direction(B, ∇fx, scheme) # solve Bd = -∇fx
    φ = LineObjective(objective, x, d, fx, dot(∇fx, d))

    # # Perform line search along d
    α, f_α, ls_success = find_steplength(linesearch, φ, Tf(1))

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # Update approximation
    fz, ∇fz, B = update_obj(objective, d, s, ∇fx, z, ∇fz, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(z, ∇fz, options.g_tol)

    return x, z, fz, ∇fz, B, is_converged
end
