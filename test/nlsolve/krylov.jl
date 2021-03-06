using NLSolvers
using LinearAlgebra
using SparseDiffTools
using SparseArrays

# Start with a stupid example
n = 10
A = sprand(10, 10, 0.1)
A = A*A' + I
F(Fx, x) = mul!(Fx, A, x)

x0, = rand(10)
xp = copy(x0)
Fx = copy(xp)



function theta(x)
   if x[1] > 0
       return atan(x[2] / x[1]) / (2.0 * pi)
   else
       return (pi + atan(x[2] / x[1])) / (2.0 * pi)
   end
end

function F_powell!(Fx, x)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        Fx[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        Fx[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        Fx[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
    Fx
end

krylov_problem = ResidualKrylovProblem(F_powell!; seed=rand(3))
nlsolve!(krylov_problem, rand(3), ResidualKrylov(FixedForceTerm(0.4)))

function f_2by2!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
    F
end

function g_2by2!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
    J
end

nlsolve!(f_2by2!, rand(2), Krylov(FixedForceTerm(0.4)))




function f_sparse!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end

function g_sparse!(J, x)
    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u
end










function helical_valley()
    tpi = 8*atan(1)
    c7 = 2.5e-1
    c8 = 5e-1

    function f!(fvec, x)
        if x[1] > 0
            temp1 = atan(x[2]/x[1])/tpi
        elseif x[1] < 0
            temp1 = atan(x[2]/x[1])/tpi + c8
        else
            temp1 = c7*sign(x[2])
        end
        temp2 = sqrt(x[1]^2+x[2]^2)
        fvec[1] = 10(x[3] - 10*temp1)
        fvec[2] = 10(temp2 - 1)
        fvec[3] = x[3]
    end
    function fj!(F, J, x)
        if x[1] > 0
            temp1 = atan(x[2]/x[1])/tpi
        elseif x[1] < 0
            temp1 = atan(x[2]/x[1])/tpi + c8
        else
            temp1 = c7*sign(x[2])
        end
        temp2 = sqrt(x[1]^2+x[2]^2)
        if !isa(F, Nothing)
            F[1] = 10*(x[3] - 10*temp1)
            F[2] = 10*(temp2 - 1)
            F[3] = x[3]
        end
        if !isa(J, Nothing)
            temp = x[1]^2 + x[2]^2
            temp1 = tpi*temp
            temp2 = sqrt(temp)
            J[1,1] = 100*x[2]/temp1
            J[1,2] = -100*x[1]/temp1
            J[1,3] = 10
            J[2,1] = 10*x[1]/temp2
            J[2,2] = 10*x[2]/temp2
            J[2,3] = 0
            J[3,1] = 0
            J[3,2] = 0
            J[3,3] = 1
            if !isa(F, nothing)
                return F, J
            else
                return J
            end
        end

    end
    f!, fj!, rand(3), [-1.; 0; 0]), [-1.; 0; 0], "Helical Valley")
end
