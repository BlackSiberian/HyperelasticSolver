#
# Strains.jl
#
# General definitions for strains, invariants, etc.

module Strains

using LinearAlgebra: inv, tr, det, I

export finger, invariants, di1dg, di2dg, di3dg

# TODO: Сделать типы для мер деформаций, нужно в hyperelasticity.jl и eos.jl.

# finger(f::Array{<:Any,2})::Array{<:Any,2} = inv(f * transpose(f))

"""
    finger(f::Array{<:Any,1})::Array{<:Any,1}

Returns the symmetric Finger's tensor

# Arguments

- `f`: a gradient deformations tensor represented as 1-by-9 vector.
"""
function finger(f::Array{<:Any,1})::Array{<:Any,1}
    f = reshape(f, (3, 3))
    g = inv(f * transpose(f))
    g = reshape(g, length(g))
    return g
end


# """
#     Returns the triplet of symmetric tensor invariants.
#     @param G is an arbitrary symmetric tensor represented as 3-by-3 array.
# """
# function invariants(g::Array{<:Any,2})::Array{<:Any,1}
#     i1 = tr(g)
#     i2 = 0.5 * (tr(g)^2 - tr(g^2))
#     i3 = det(g)
#     return [i1, i2, i3]
# end


"""
    invariants(g::Array{<:Any,1})::Array{<:Any,1}

Returns the triplet of symmetric tensor invariants.

# Arguments 

- `G`: an arbitrary symmetric tensor represented as 1-by-9 vector.
"""
function invariants(g::Array{<:Any,1})::Array{<:Any,1}
    g = reshape(g, (3, 3))
    i1 = tr(g)
    i2 = 0.5 * (tr(g)^2 - tr(g^2))
    i3 = det(g)
    return [i1, i2, i3]
end

# di1dg(G, i1, i2, i3) = I            # Returns the derivative of I with respect to G
# di2dg(G, i1, i2, i3) = i1 .* I - G  # @param G is the symmetric tensor
# di3dg(G, i1, i2, i3) = i3 .* inv(G) # @param I1, I2, I3 are the invariants

end # Module Strains

# EOF