#
# SimpleLA.jl
#
# Simple linear algebra implementation to use with ForwardDiff
#

# TODO:
#     1. Сделать более кореектные типы, как массивы 3 на 3
#
#     2. Сделать все функции как фугкции векторов 1 на 9 и
#     как матриц 3 на 3.
#
#     3. Сделать статические массивы, так как:
#        julia> @btime inv(SMatrix{3,3}(F))
#        171.179 ns (2 allocations: 160 bytes)       <---- !!!
#        3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
#        0.949122     0.0        0.0
#        0.00965773   0.999076  -0.0222017
#        0.0158187   -0.0        1.05458
#
#        julia> @btime inv(F)
#        656.722 ns (4 allocations: 1.86 KiB)       <---- !!!
#        3×3 Matrix{Float64}:
#        0.949122    -0.0       -0.0
#        0.00965773   0.999076  -0.0222017
#        0.0158187    0.0        1.05458
#
#    4. Переместить сюда вычисление инвариантов.
#


module SimpleLA
export inv_, det_, norm_, tr_

#using StaticArrays


"""
    2-norm for an arbitrary matrix/tensor.
"""
function norm_(a)
  return sqrt(sum(a .^ 2))
end # norm_()


"""
    Computes the inverse of a matrix m.
"""
#function finger(a::Array{T,1})::Array{T,1} where T
#function inv_(m::Array{<:Any,2})::Array{<:Any,2}
function inv_(m::Array{T,2})::Array{T,2} where {T}

  #minv = similar(m) # typeof(minv) = typeof(m)

  minv = Array{<:Any,2}(undef, 3, 3) # ОК! --- Работает:  334.677 ns, FD: 183.054 μs (2271 allocations: 94.20 KiB)
  #minv = Array{T, 2}(undef, 3, 3)      # 
  #minv = SMatrix{2,2}(m)              # ОК! --- Работает: 306.931ns, FD: 158.997 μs (2271 allocations: 94.20 KiB)

  dt = m[1, 1] * (m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3]) -
       m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) +
       m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])

  invdet = 1.0 / dt

  #minv = zeros(3,3)
  minv[1, 1] = (m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3]) * invdet
  minv[1, 2] = (m[1, 3] * m[3, 2] - m[1, 2] * m[3, 3]) * invdet
  minv[1, 3] = (m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]) * invdet

  minv[2, 1] = (m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3]) * invdet
  minv[2, 2] = (m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]) * invdet
  minv[2, 3] = (m[2, 1] * m[1, 3] - m[1, 1] * m[2, 3]) * invdet

  minv[3, 1] = (m[2, 1] * m[3, 2] - m[3, 1] * m[2, 2]) * invdet
  minv[3, 2] = (m[3, 1] * m[1, 2] - m[1, 1] * m[3, 2]) * invdet
  minv[3, 3] = (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) * invdet

  return minv
end # inv_()

"""
    Determinant of a matrix
"""
function det_(m::Array{<:Any,2})
  dt = m[1, 1] * (m[2, 2] * m[3, 3] - m[3, 2] * m[2, 3]) -
       m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) +
       m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])
  return dt
end # det_()

"""
    Trace of a matrix.
"""
function tr_(m::Array{<:Any,2})
  trm = m[1, 1] + m[2, 2] + m[3, 3]
end # tr_()


"""
    Convert matrix to 4th rank tensor.
"""
function mat2ten(x::Array{T,2}) where {T}
  y = zeros(3, 3, 3, 3)

  # Обходим компоненты тензора 4-го ранга
  for i in 1:3
    for j = 1:3
      for k = 1:3
        for l = 1:3
          row = 3 * (i - 1) + j
          col = 3 * (k - 1) + l
          y[i, j, k, l] = x[row, col]
        end
      end
    end
  end
  return y
end # mat2ten()

end # Module SimpleLA

# EOF
