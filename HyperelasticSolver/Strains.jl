#
# Strains.jl
#
# General definitions for strains, invariants, etc.
module Strains
using LinearAlgebra

export finger, invariants, di1dg, di2dg, di3dg


# TODO: Сделать типы для мер деформаций, нужно в hyuperelasticity.jl и eos.jl.


# Теперь:
# invariants:
#     матрица -> вектор
#     вектор  -> вектор
#
# finger: 
#     матрица -> матрица
#     вектор  -> матрица
#     вектор  -> вектор   (?)
#


"""
    Returns the symmetric Finger's tensor 
"""
finger(f::Array{<:Any,2})::Array{<:Any,2} = inv(f * transpose(f))

function finger(f::Array{<:Any,1})::Array{<:Any,2}
    f2 = [ f[1] f[2] f[3];
           f[4] f[5] f[6];
           f[7] f[8] f[9] ]
    return inv(f2 * transpose(f2))
end


"""
    Returns the triplet of symmetric tensor invariants.
    @param G is an arbitrary symmetric tensor represeted as 3-by-3 array.
"""
function invariants(g::Array{<:Any,2})
    i1 = tr(g)
    i2 = 0.5 * (tr(g)^2 - tr(g^2))
    i3 = det(g)
    return [i1, i2, i3]
end


"""
    Returns the triplet of symmetric tensor invariants.
    @param G is an arbitrary symmetric tensor represented as 1-by-9
    vector.
    TODO: Внутри можно вызывать предыдущий метод, --- нолучше
          переписать эта без LinearAlgebra.
"""
function invariants(g::Array{<:Any,1})
    g2 = [ g[1] g[2] g[3];                    # This isn't the natural
           g[4] g[5] g[6];                    # order in Julia!
           g[7] g[7] g[9] ]
    i1 = tr(g2)
    i2 = 0.5 * (tr(g2)^2 - tr(g2^2))
    i3 = det(g2)
    return [i1, i2, i3]
end



di1dg(G, i1, i2, i3) = I            # Returns the derivative of I with respect to G
di2dg(G, i1, i2, i3) = i1 .* I - G  # @param G is the symmetric tensor
di3dg(G, i1, i2, i3) = i3 .* inv(G) # @param I1, I2, I3 are the invariants

end # Module Strains


# EOF
