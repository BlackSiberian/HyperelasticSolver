#
# NumFluxes.jl
#


module NumFluxes


#include("./Hyperelasticity.jl")
# using ..Hyperelasticity: flux
using ..HyperelasticityMPh: flux_mph
using ..EquationsOfState: EoS

export lxf

"""
Returns the value of the numerical flux between cells using the
Lax-Friedrichs method.
"""
function lxf(eos::T, Q_l::Array{<:Any,1}, Q_r::Array{<:Any,1}, lambda) where {T <: EoS}
    return 0.5 * (flux_mph(eos, Q_l) + flux_mph(eos, Q_r)) - 0.5 * lambda * (Q_r - Q_l)
    # return 0.5 * (flux(Q_l) + flux(Q_r)) - 0.5 * lambda * (Q_r - Q_l)
end

end # module NumFluxes
# EOF