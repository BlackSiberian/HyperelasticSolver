#
# Strains.jl
#
# General definitions for strains, invariants, etc.

module Strains


using ..SimpleLA: inv_, tr_, det_
# using LinearAlgebra: inv, tr, det, I

export finger, invariants

# TODO: Сделать типы для мер деформаций, нужно в hyperelasticity.jl и eos.jl.

"""
    finger(a::Array{<:Any,1})::Array{<:Any,1}

Compute the symmetric Finger's tensor for a distortion tensor
"""
# function finger(a::Array{<:Any,1})::Array{<:Any,1}
#   a = reshape(a, (3, 3))
#   g = inv(a * transpose(a))[:]
#   return g
# end
function finger(a::Array{T,1})::Array{T,1} where {T}
  #function finger(a::Array{<:Any,1})::Array{<:Any,1}
  #function finger(a::Array{<:Any,1})
  a = reshape(a, (3, 3))
  g = inv_(a * transpose(a))[:]
  return g
end

"""
    invariants(g::Array{<:Any,1})::Array{<:Any,1}

Return the triplet of invariants for a Finger's symmetric tensor
"""
# function invariants(g::Array{<:Any,1})::Array{<:Any,1}
#   g = reshape(g, (3, 3))
#   i1 = tr(g)
#   i2 = 0.5 * (tr(g)^2 - tr(g^2))
#   i3 = det(g)
#   return [i1, i2, i3]
# end
function invariants(g::Array{<:Any,1})::Array{<:Any,1}
  g = reshape(g, (3, 3))
  i1 = tr_(g)
  i2 = 0.5 * (tr_(g)^2 - tr_(g^2))
  i3 = det_(g)
  return [i1, i2, i3]
end

end # Module Strains

# EOF
