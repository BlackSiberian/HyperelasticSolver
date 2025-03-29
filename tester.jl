# Include all modules that is used in main.jl and in imported modules
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
# include("./Hyperelasticity.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")

# Только то, что нужно в main.jl
using .EquationsOfState: EoS, Barton2009
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh: initial_states, cons2prim_mph, prim2cons_mph, get_eigvals, flux_mph, noncons_flux
using .NumFluxes: lxf, hll

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff: derivative

eos = (Barton2009(_rho0=8.93, _c0=4.6, _cv=0.00039, _t0=300, _b0=2.1, _alpha=1, _beta=3, _gamma=2), Barton2009(_rho0=8.93, _c0=4.6, _cv=0.00039, _t0=300, _b0=2.1, _alpha=1, _beta=3, _gamma=2))

alpha_1 = alpha_2 = 0.5
den_1 = den_2 = 8.93

u_1 = [-0.21738, -5.27757, 6.89893]
F_1 = [1.0527 -1.19581e-20 2.4884e-20;
        -3.12631 1.0 0.1;
        3.225 -3.08217e-20 1.0]
  
S_1 = 0.001

u_2 = [0, 0.5, 1.0]
F_2 = [0.98 0.0 0.0;
        0.02 1.0 0.1;
        0.0 0.0 1.0]
S_2 = 1e-3

den_1 = den_1 / det(F_1)
den_2 = den_2 / det(F_2)

P = [alpha_1, den_1, u_1..., S_1, F_1...,
  alpha_2, den_2, u_2..., S_2, F_2...]

Q = prim2cons_mph(eos, P)

# Q_l = [0.1, 0.911224, 0, 0.455612, 0.911224, 1.92656, 0.893, 0.0182245, 0, 0, 0.911224, 0, 0, 0.0911224, 0.911224, 0.9, 8.20102, 0, 4.10051, 8.20102, 17.339, 8.037, 0.16402, 0, 0, 8.20102, 0, 0, 0.820102, 8.20102]
# Q_r = [0.9, 8.037, 0, 0, 0, 0.177807, 8.037, 0, 0, 0, 8.037, 0, 0, 0.8037, 8.037, 0.1, 0.893, 0, 0, 0, 0.0197563, 0.893, 0, 0, 0, 0.893, 0, 0, 0.0893, 0.893]
# Q0 = hcat(Q_l, Q_r)
Q0 = Q

# eos = (Barton2009(), Barton2009())

nx = 1

lambda = Array{Float64}(undef, nx)
eigvals = Array{Array{Float64}}(undef, nx)
Threads.@threads for i = 1:nx
  Q = Q0[:, i]
  n = [1, 0, 0]
  eigvals[i] = get_eigvals(eos, Q, n)
  lambda[i] = maximum(abs.(eigvals[i]))
end

display(eigvals)



# function my2r(A::Array{<:Any,1})
#   Ph1 = A[1:15]
#   Ph2 = A[16:30]
#   return [Ph1[1], Ph2[1], Ph1[2], Ph2[2], Ph1[3:5]..., Ph2[3:5]..., reshape(transpose(reshape(Ph1[7:15], (3, 3))), 9)..., reshape(transpose(reshape(Ph2[7:15], (3, 3))), 9)..., Ph1[6], Ph2[6]]
# end
# Ph1 = D_m[1:15]
# Ph2 = D_m[16:30]
#
# R = [Ph1[1], Ph2[1], Ph1[2], Ph2[2], Ph1[3:5]..., Ph2[3:5]..., reshape(transpose(reshape(Ph1[7:15], (3, 3))), 9)..., reshape(transpose(reshape(Ph2[7:15], (3, 3))), 9)..., Ph1[6], Ph2[6]]

# display(my2r(flux_mph(eos, Q_l)))
# display(my2r(flux_mph(eos, Q_r)))

# F_r, D_m, D_p= hll(eos, Q_l, Q_r, eigvals)
# display(my2r(D_m))
# display(my2r(D_p))
# s = 0.0337652
# s = 0.169395

# display(my2r(noncons_flux(eos, Q)[:, 1]))

# nodes, weights = gausslegendre(6)                       # for [-1,+1] interval
# nodes, weights = (nodes .+ 1.0) / 2.0, weights ./ 2.0  # for [0,1] interval
# path(Q_l, Q_r, s) = Q_l .* (1 - s) + Q_r .* s # define path
# dpath(Q_l, Q_r, s) = derivative(s -> path(Q_l, Q_r, s), s)

# display(my2r(Q_l))
# display(my2r(Q_r))
# println("d alpha_1 = $(dpath(Q_l, Q_r, 0)[1]), d alpha_2 = $(dpath(Q_l, Q_r, 0)[16])")

# for node in nodes
#   println("node = $node")
#   Q = path(Q_l, Q_r, node)
#   noncons_flux(eos, Q)
# end


