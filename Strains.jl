#
# Strains.jl
#
# General definitions for strains, invariants, etc.

module Strains

using LinearAlgebra: inv, tr, det, I

export finger, invariants#, di1dg, di2dg, di3dg

# TODO: Сделать типы для мер деформаций, нужно в hyperelasticity.jl и eos.jl.

"""
    finger(a::Array{<:Any,1})::Array{<:Any,1}

Computes the symmetric Finger's tensor for a distortion tensor
"""
function finger(a::Array{<:Any,1})::Array{<:Any,1}
  a = reshape(a, (3, 3))
  g = inv(a * transpose(a))[:]
  return g
end

"""
    invariants(g::Array{<:Any,1})::Array{<:Any,1}

Returns the triplet of invariants for a Finger's symmetric tensor
"""
function invariants(g::Array{<:Any,1})::Array{<:Any,1}
  g = reshape(g, (3, 3))
  i1 = tr(g)
  i2 = 0.5 * (tr(g)^2 - tr(g^2))
  i3 = det(g)
  return [i1, i2, i3]
end

end # Module Strains

# EOF
