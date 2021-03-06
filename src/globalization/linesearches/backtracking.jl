# Notation:
# λ is the initial step length
# α current trial step length
# β current trial step length
# d is the search direction
# x is the current iterate
# f is the objective
# φ is the line search objective and is a function of the step length only

# TODO
# For non-linear systems of equations equations we generally choose the sum-of-
# squares merit function. Some useful things to remember is:
#
# f(y) = 1/2*|| F(y) ||^2 =>
# ∇_df = -d'*J(x)'*F(x)
#
# where we remember the notation x means the current iterate here. This means
# that if we step in the Newton direction such that d is defined by
#
# J(x)*d = -F(x) => -d'*J(x)' = F(x)' =>
# ∇_df = -F(x)'*F(x) = -f(x)*2
#

# This file contains several implementation of what we might call "Backtracking".
# The AbstractBacktracking line searches try to satisfy the Amijo(-Goldstein)
# condition:
#     |f(x + α*d)| < (1-c_1*α)*|f(x)|
# That is: the function should

# As per [Nocedal & Wright, pp. 37] we don't have to think about the curvature
# condition as long as we use backtracking.

abstract type AbstractBacktracking end
abstract type BacktrackingInterp end
"""
    _safe_α(α_cand, α_curr, c, ratio)
Returns the safeguarded value of α in a Amijo
backtracking line search.

σ restriction 0 < c < ratio < 1
"""
function _safe_α(α_cand, α_curr, decrease=0.1, ratio=0.5)
    α_cand < decrease*α_curr && return decrease*α_curr
    α_cand > ratio*α_curr && return ratio*α_curr

	α_cand # if the candidate is in the interval, just return it
end


struct Backtracking{T1, T2, T3} <: LineSearch
    ratio::T1
	decrease::T1
	maxiter::T2
	interp::T3
	verbose::Bool
end
Backtracking(; ratio=0.5, decrease=1e-4, maxiter=50, interp=FixedInterp(), verbose=false) = Backtracking(ratio, decrease, maxiter, interp, verbose)

struct FixedInterp <: BacktrackingInterp end
struct FFQuadInterp <: BacktrackingInterp end

function interpolate(itp::FixedInterp, φ, φ0, dφ0, α, f_α, ratio)
	β = α
	α = ratio*α
	φ_α = φ(α)
	β, α, φ_α
end


## Polynomial line search
# f(α) = ||F(xₙ+αdₙ)||²₂
# f(0) = ||F(xₙ)||²₂
# f'(0) = 2*(F'(xₙ)'dₙ)'F(xₙ) = 2*F(xₙ)'*(F'(xₙ)*dₙ) < 0
# if f'(0) >= 0, then dₙ does is not a descent direction
# for the merit function. This can happen for broyden

# two-point parabolic
# at α = 0 we know f and f' from F and J as written above
# at αc define
function twopoint(f, f_0, df_0, α, f_α, ratio)
	ρ_lo, ρ_hi = 0.1, ratio
    # get the minimum (requires df0 < 0)
	c = (f_α - f_0 - df_0*α)/α^2

	# p(α) = f0 + df_0*α + c*α^2 is the function
	# we have df_0 < 0. Then if  f_α > f(0) then c > 0
	# by the expression above, and p is convex. Then,
	# we have a minimum between 0 and α at

	γ = -df_0/(2*c) # > 0 by df0 < 0 and c > 0
    # safeguard α
    return max(min(γ, α*ρ_hi), α*ρ_lo) # σs
end

function interpolate(itp::FFQuadInterp, φ, φ0, dφ0, α, f_α, ratio)
	β = α
	α = twopoint(φ, φ0, dφ0, α, f_α, ratio)
	φ_α = φ(α)
	β, α, φ_α
end


"""
    find_steplength(---)

Returns a step length, (merit) function value at step length and succes flag.
"""
function find_steplength(ls::Backtracking, φ::T, λ) where T
 	#== unpack ==#
	φ0, dφ0 = φ.φ0, φ.dφ0
	Tf = typeof(φ0)
	ratio, decrease, maxiter, verbose = Tf(ls.ratio), Tf(ls.decrease), ls.maxiter, ls.verbose

	#== factor in Armijo condition ==#
    t = -decrease*dφ0

    iter, α, β = 0, λ, λ # iteration variables
	f_α = φ(α)   # initial function value

	if verbose
		println("Entering line search with step size: ", λ)
		println("Initial value: ", φ0)
		println("Value at first step: ", f_α)
	end
	is_solved = isfinite(f_α) && f_α <= φ0 + α*t
    while !is_solved && iter <= maxiter
        iter += 1
        β, α, f_α = interpolate(ls.interp, φ, φ0, dφ0, α, f_α, ratio)
		is_solved = isfinite(f_α) && f_α <= φ0 + α*t
    end

	ls_success = iter >= maxiter ? false : true

    if verbose
		!ls_success && println("maxiter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end

# The
struct ThreePointQuadratic{T1, T2} <: LineSearch
    ratio::T1
	decrease::T1
	maxiter::T2
	verbose::Bool
end
ThreePointQuadratic(; ratio=0.5, decrease=1e-4, maxiter=50, verbose=false) = ThreePointQuadratic(ratio, decrease, maxiter, verbose)
