#
# NumFluxes.jl
#
# Numerical fluxes

module NumFluxes

# using ..Hyperelasticity: flux
using ..HyperelasticityMPh: flux_mph, noncons_flux
using ..EquationsOfState: EoS
using ForwardDiff: derivative
using FastGaussQuadrature: gausslobatto
using LinearAlgebra: I

export lxf

"""
    lxf(eos::T, Q_l::Array{<:Any,1}, Q_r::Array{<:Any,1}, lambda)::Array{<:Any,1} where {T <: EoS}

Returns the value of the numerical flux between cells using 
the Lax-Friedrichs method and `eos` equation of state

`lambda` is the value of `Δx/Δt`
"""
function lxf(eos::Tuple{T,T}, Q_l::Array{<:Any,1}, Q_r::Array{<:Any,1}, lambda) where {T<:EoS}
  # return 0.5 * (flux(Q_l) + flux(Q_r)) - 0.5 * lambda * (Q_r - Q_l)

  path(Q_l, Q_r, s) = Q_l .* (1 - s) + Q_r .* s # define path

  cons = 0.5 * (flux_mph(eos, Q_l) + flux_mph(eos, Q_r)) - 0.5 * lambda * (Q_r - Q_l)
  noncons_minus, noncons_plus = lxf_pathcons(eos, Q_l, Q_r, path, lambda)
  return cons, noncons_minus, noncons_plus
end

function lxf_pathcons(eos::Tuple{T,T}, Q_l::Array{<:Any,1}, Q_r::Array{<:Any,1}, path::Function, lambda) where {T<:EoS}
  nodes, weights = gausslobatto(5)                       # for [-1,+1] interval
  nodes, weights = (nodes .+ 1.0) / 2.0, weights ./ 2.0  # for [0,1] interval

  A = Q -> noncons_flux(eos, Q)
  # Path derivative, consider to define globally
  dpath(Q_l, Q_r, s) = derivative(s -> path(Q_l, Q_r, s), s)
  uvals = [path(Q_l, Q_r, s) for s in nodes]     # values of ψ in quad. points
  dvals = [dpath(Q_l, Q_r, s) for s in nodes]    # values of ∂ψ/∂s in quad. points
  avals = A.(uvals)                              # values of matrix in quad. points

  # Integration
  flux_path = sum([weights[i] * avals[i] * dvals[i] for i in 1:length(weights)])

  # <<Потоки>>, $D^+$ и $D^-$
  dp = (1.0 / 2.0) * flux_path #+ (1.0 / 2.0) * lambda * (Q_r - Q_l)
  dm = (1.0 / 2.0) * flux_path #- (1.0 / 2.0) * lambda * (Q_r - Q_l)

  # aplusvals = [1 / 2 .* (avals[i] + lambda * I) for i in 1:length(avals)]
  # aminusvals = [1 / 2 .* (avals[i] - lambda * I) for i in 1:length(avals)]
  #
  # dp = sum([weights[i] * aplusvals[i] * dvals[i] for i in 1:length(weights)])
  # dm = sum([weights[i] * aminusvals[i] * dvals[i] for i in 1:length(weights)])

  return dm, dp
end

end # module NumFluxes

# EOF
