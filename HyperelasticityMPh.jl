#
# HyperelasticityMPh.jl
#
# Multiphase hyperelasticity (Savenkov, Alekseev, 2022)

module HyperelasticityMPh

using LinearAlgebra: inv, transpose, det, tr, I
using ForwardDiff: derivative
using ..EquationsOfState: energy, entropy, stress, EoS, Barton2009, Hank2016
using ..Strains: finger, invariants

export prim2cons_mph, cons2prim_mph, flux_mph, noncons_flux, initial_states #, postproc_arrays

# Number of phases
# const nph = 2

# 1. Для многофазной задачи номер фазы --- всегда параметр в массиве.
#
# 2. cons2prim и prim2cons --- принимают вектора нужной длины и отдают
#    вектора такой же длины (не так, как сейчас для простой
#    гиперупругости).
#                               
# 3. flux() --- принимает <<консервативные>> переменные
#    (которые под производной по времени), отдает три величины:
#    физический поток, его якобиан и неконсервативную матрицу B  
#
# 4. Так как фазы независимы по УрС, то переход от первичных к
#    консервативным и наоборот, вычисление энтропии, энергии и так далее
#    --- всегда можно сделать локально для каждой фазы.
#    
# 5. Как можно реже переводить вектор длины 9 в тензор и наоборот.
#    На самом деле, нам практически никогда не нужны тензоры как
#    матрицы 3 на 3.
#
#    Возможно, потребуется дописать какие то еще методы с тем же именем
#    и другими интерфейсами.
#
# 6. Помнить, что теперь EquationsOfStates.jl --- для всех задач.
#    Поэтому нельзя ломать совместимость. Дописываем методы, не трогая
#    старые! 
#
# 7. Функции для задания начальных условий и визуализации задаются в
#    соответствующих модулях с "физикой".
#    Потом нужно будет отцепить их во вспомогательные, чтобы не
#    мешать с содержательной частью.
#    
#  8. Общее правило: main.jl --- без изменений при смене модели,
#     EquationsOfState.jl, Strains.jl --- тоже.
#     минимальные изменения могут быть, но по возможности,
#     максимально, без них.
#
#  9. Имена соответственных функций  должны совпадать --- в какой-то
#     момент параметризуем их типом, соответствующим задаче.
#     Пока не нужно, но лучше заложиться.

"""
    prim2cons_mph(eos::T, P::Array{<:Any, 1}) where {T<:EoS}

Converts primitive variables to conservative variables for multiphase
hyperelasticity with `eos` equation of state.
"""
function prim2cons_mph(eos::T, P::Array{<:Any,1}) where {T<:EoS}
  ph = Tuple(P[i:i+15-1] for i in 1:15:length(P))

  function prim2cons(eos::T, P::Array{<:Any,1}) where {T<:EoS}
    Q = similar(P)
    frac = P[1]
    true_den = P[2]
    den = frac * true_den
    vel = P[3:5]
    entropy = P[6]
    def_grad = P[7:15]

    # G = finger(inv(reshape(distortion, (3, 3))))
    # G = finger(distortion)
    G = finger(def_grad)
    e_int = energy(eos, entropy, G)
    e_kin = sum(vel .^ 2) / 2
    e_total = e_int + e_kin

    Q[1] = frac
    Q[2] = den
    Q[3:5] = den * vel
    Q[6] = den * e_total
    Q[7:15] = den * def_grad

    return Q
  end
  Q = vcat([prim2cons(eos, P) for P in ph]...)
  return Q
end


"""
    cons2prim_mph(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}

Converts conservative variables to primitive variables
for multiphase hyperelasticity with `eos` equation of state.
"""
function cons2prim_mph(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  ph = Tuple(Q[i:i+15-1] for i in 1:15:length(Q))

  function cons2prim(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
    P = similar(Q)

    frac = Q[1]
    den = Q[2]
    true_den = den / frac
    vel = Q[3:5] / den
    e_total = Q[6] / den
    e_kin = sum(vel .^ 2) / 2
    e_int = e_total - e_kin
    def_grad = Q[7:15] / den

    G = finger(def_grad)
    # i = invariants(G)
    ent = entropy(eos, e_int, G)

    P[1] = frac
    P[2] = true_den
    P[3:5] = vel
    P[6] = ent
    P[7:15] = def_grad

    return P
  end
  P = vcat([cons2prim(eos, Q) for Q in ph]...)
  return P
end


"""
    flux_mph(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}

Computes the physical flux for multiphase hyperelasticity with `eos` equation of state.
"""
function flux_mph(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  ph = Tuple(Q[i:i+15-1] for i in 1:15:length(Q))

  F = vcat([flux(eos, Q) for Q in ph]...)
  return F
end

function flux(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  frac = Q[1]
  den = Q[2]
  true_den = den / frac
  vel = Q[3:5] / den
  e_total = Q[6] / den
  e_kin = sum(vel .^ 2) / 2
  e_int = e_total - e_kin
  def_grad = Q[7:15] / den

  strs = stress(eos, den, e_int, def_grad)

  flux = similar(Q)

  flux[1] = 0
  flux[2] = den * vel[1]
  flux[3:5] = den * vel[1] * vel - strs[begin:3:end]
  flux[6] = den * vel[1] * e_total - sum(vel .* strs[begin:3:end])
  flux[7:15] = den .* (vel[1] .* def_grad - (vel*transpose(def_grad[begin:3:end]))[:])

  return flux
end

function noncons_flux(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  Q = [Q[p:p+14] for p in 1:15:length(Q)]
  nph = length(Q)
  frac = [Q[p][1] for p in 1:nph]
  den = [Q[p][2] for p in 1:nph]
  true_den = den ./ frac
  vel = [Q[p][3:5] / den[p] for p in 1:nph]
  e_total = [Q[p][6] / den[p] for p in 1:nph]
  e_kin = [sum(vel[p] .^ 2) / 2 for p in 1:nph]
  e_int = e_total - e_kin
  def_grad = [Q[p][7:15] / den[p] for p in 1:nph]

  strs = [reshape(stress(eos, den[p], e_int[p], def_grad[p]), 3, 3) for p in 1:nph]

  # Омега должна быть равна нулю, поскольку иначе возникает нефизичный импульс
  # omega = 1 / 2
  omega = 0
  k = 1 / 2
  k = [k, 1 - k]
  beta = zeros(2)

  G = [finger(def_grad[p]) for p in 1:nph]
  S = [entropy(eos, e_int[p], G[p]) for p in 1:nph]
  temp = [derivative(S -> energy(eos, S, G[p]), S[p]) for p in 1:nph]
  vel_i = k[1] .* vel[1] + k[2] .* vel[2]
  # vel_i = vel[2]
  # vel_i = frac[1] .* vel[1] + frac[2] .* vel[2]
  K = [1 / frac[p] .* (omega / 3 .* tr(strs[p]) * I + (1 - omega) .* strs[p]) + beta[p] .* I for p in 1:nph]
  strs_i = (k[2] * temp[2] .* K[1] + k[1] * temp[1] .* K[2]) / (k[1] * temp[1] + k[2] * temp[2])
  # strs_i = frac[1] .* K[1] + frac[2] .* K[2]
  # strs_i = K[1]

  B = [zeros(Float64, length(Q[p]), length(Q[p])) for p in 1:nph]
  for p in 1:nph
    B[p][1, 1] = vel_i[1]
    B[p][3:5, 1] = strs_i[:, 1]
    B[p][6, 1] = sum(strs_i[:, 1] .* vel_i)

    for i in 0:3:8
      B[p][6+i.+(1:3), 1] = omega * true_den[p] / 3 .* (vel_i[1] - vel[p][1]) .* def_grad[p][i.+(1:3)] + true_den[p] * def_grad[p][i+1] .* vel[p]
    end
    B[p][7:3:15, 1] += (1 - omega) * true_den[p] .* transpose(reshape(def_grad[p], 3, 3)) * (vel_i - vel[p])
    # for i in 1:3
    #   B[p][7+(i-1)*3, 1] = (1 - omega) * true_den[p] * sum(def_grad[p][1:3] .* vel[p])
    # end

    # bad ones
    # for i in 0:2
    #   B[p][6+3*i.+(1:3), 1] = omega * true_den[p] / 3 * (vel_i[1] - vel[p][1]) .* def_grad[p][3*i.+(1:3)]
    # end
    # B[p][7:9, 1] += den[p] * (vel_i[1] - vel[p][1]) .* def_grad[p][1:3]
  end


  function combine_blocks(matrices::AbstractVector{<:AbstractMatrix})
    n = length(matrices)
    m = size(matrices[1], 1)
    big_matrix = zeros(eltype(matrices[1]), n * m, n * m)

    for i in 1:n, j in 1:n
      if i == j
        big_matrix[(i-1)*m+1:(i-1)*m+m, (j-1)*m+1:(j-1)*m+m] = matrices[i]
      end
    end

    return big_matrix
  end

  B = combine_blocks(B)
  return B
end

"""
    initial_states(eos::T, testcase::Int) where {T<:EoS}

Sets left and right states for the Riemann problem for multiphase hyperelasticity with `eos` equation of state

'testcase' is the test case number
"""
function initial_states(eos::T, testcase::Int) where {T<:EoS}
  if testcase == 1
    alpha_l_1 = alpha_r_1 = 0.5
    alpha_l_2 = alpha_r_2 = 0.5

    den_1 = den_2 = 5.0

    u_l_1 = u_l_2 = u_r_1 = u_r_2 = [0, 0, 0]

    S_l_1 = S_l_2 = S_r_1 = S_r_2 = 0

    F_l_1 = F_l_2 = F_r_1 = F_r_2 = [1 0 0; 0 1 0; 0 0 1]
  elseif testcase == 2
    alpha_l_1 = alpha_r_1 = 0.5
    alpha_l_2 = alpha_r_2 = 0.5

    den_1 = den_2 = 5.0

    u_l_1 = u_l_2 = u_r_1 = u_r_2 = [1.0, 0, 0]

    S_l_1 = S_l_2 = S_r_1 = S_r_2 = 0

    F_l_1 = F_l_2 = F_r_1 = F_r_2 = [1 0 0; 0 1 0; 0 0 1]
  elseif testcase == 3
    alpha_l_1 = alpha_r_1 = 0.5
    alpha_l_2 = alpha_r_2 = 0.5

    den_1 = den_2 = 8.9

    u_l_1 = u_l_2 = u_r_1 = u_r_2 = [2.0, 0.0, 0.1]

    S_l_1 = S_l_2 = S_r_1 = S_r_2 = 0.0

    F = [1.0 0.0 0.0;
      -0.01 0.95 0.02;
      -0.015 0.0 0.9]
    F_l_1 = F_l_2 = F_r_1 = F_r_2 = F
  elseif testcase == 4
    alpha_l_1 = alpha_l_2 = 0.5
    alpha_r_1 = alpha_r_2 = 0.5
    den_1 = den_2 = 8.9

    u_l_1 = u_l_2 = [0.0, 0.5, 1.0]
    F_l_1 = F_l_2 = [
      0.98 0.0 0.0;
      0.02 1.0 0.1;
      0.0 0.0 1.0]
    S_l_1 = S_l_2 = 1e-3

    u_r_1 = u_r_2 = [0.0, 0.0, 0.0]
    F_r_1 = F_r_2 = [
      1.0 0.0 0.0;
      0.0 1.0 0.1;
      0.0 0.0 1.0]
    S_r_1 = S_r_2 = 0.0
  elseif testcase == 5
    alpha_l_1 = alpha_l_2 = 0.5
    alpha_r_1 = alpha_r_2 = 0.5
    den_1 = den_2 = 8.9

    u_l_1 = u_l_2 = [2.0, 0.0, 0.1] # [km/s]
    F_l = [1.0 0.0 0.0;
      -0.01 0.95 0.02;
      -0.015 0.0 0.9]
    F_l_1 = F_l_2 = F_l
    S_l_1 = S_l_2 = 0.0 # [kJ/(g*K)]

    u_r_1 = u_r_2 = [0.0, -0.03, -0.01] # [km/s]
    F_r = [1.0 0.0 0.0;
      0.015 0.95 0.0;
      -0.01 0.0 0.9]
    F_r_1 = F_r_2 = F_r
    S_r_1 = S_r_2 = 0.0 # [kJ/(g*K)]
  elseif testcase == 6
    alpha_l_1 = 0.4
    alpha_l_2 = 0.6
    alpha_r_1 = 0.6
    alpha_r_2 = 0.4
    den_1 = den_2 = 8.9

    u_l_1 = u_l_2 = [2.0, 0.0, 0.1] # [km/s]
    F_l = [1.0 0.0 0.0;
      -0.01 0.95 0.02;
      -0.015 0.0 0.9]
    F_l_1 = F_l_2 = F_l
    S_l_1 = S_l_2 = 0.0 # [kJ/(g*K)]

    u_r_1 = u_r_2 = [0.0, -0.03, -0.01] # [km/s]
    F_r = [1.0 0.0 0.0;
      0.015 0.95 0.0;
      -0.01 0.0 0.9]
    F_r_1 = F_r_2 = F_r
    S_r_1 = S_r_2 = 0.0 # [kJ/(g*K)]
  elseif testcase == 7
    alpha_l_1 = 0.4
    alpha_l_2 = 0.6
    alpha_r_1 = 0.6
    alpha_r_2 = 0.4
    den_1 = den_2 = 8.9

    u_l_1 = u_l_2 = [2.0, 0.0, 0.1] # [km/s]
    F_l = [1.0 0.0 0.0;
      -0.01 0.95 0.02;
      -0.015 0.0 0.9]
    F_l_1 = F_l_2 = F_l
    S_l_1 = S_l_2 = 0.0 # [kJ/(g*K)]

    u_r_1 = u_r_2 = [2.0, 0.0, 0.1] # [km/s]
    F_r = [1.0 0.0 0.0;
      -0.01 0.95 0.02;
      -0.015 0.0 0.9]
    F_r_1 = F_r_2 = F_r
    S_r_1 = S_r_2 = 0.0
  else
  end

  den_l_1 = den_1 / det(F_l_1)
  den_r_1 = den_1 / det(F_r_1)
  den_l_2 = den_2 / det(F_l_2)
  den_r_2 = den_2 / det(F_r_2)

  Pl = [alpha_l_1, den_l_1, u_l_1..., S_l_1, F_l_1...,
    alpha_l_2, den_l_2, u_l_2..., S_l_2, F_l_2...]
  Pr = [alpha_r_1, den_r_1, u_r_1..., S_r_1, F_r_1...,
    alpha_r_2, den_r_2, u_r_2..., S_r_2, F_r_2...]

  Ql = prim2cons_mph(eos, Pl)
  Qr = prim2cons_mph(eos, Pr)

  return Ql, Qr
end # initial_states(eos::T, testcase::Int) where {T<:EoS}

"""
    Расчет значений массивов для визуализации.
    Возвращает сам массив tuple с аннотациями для переменных.
"""
# function postproc_arrays(eos, Q0)
#   nx = size(Q0)[2]
#
#   alpha = Array{Float64}(undef, nph, nx)
#   true_den = Array{Float64}(undef, nph, nx)
#   ent = Array{Float64}(undef, nph, nx)
#   vel = Array{Float64}(undef, 3, nph, nx)
#   A = Array{Float64}(undef, 9, nph, nx)
#
#   for (i, Q) in enumerate(eachcol(Q0))
#     P = cons2prim_mph(eos, Q)
#
#     alpha[:, i] = P[begin:15:end]
#     true_den[:, i] = P[begin+1:15:end]
#     for j = 1:3
#       vel[j, :, i] = P[begin+1+j:15:end]
#     end
#     for j = 1:9
#       A[j, :, i] = P[begin+5+j:15:end]
#     end
#     ent[:, i] = P[begin+14:15:end]
#
#     ent[i] = entropy(e_int, invariants(finger(F)))
#     eint[i] = e_int
#   end
#
#   info = ("Fraction", "True Density", "Velocity", "Distortion", "Entropy", "info")
#   return alpha, true_den, vel, A, ent, info
# end

end # module HyperelasticityMPh

# EOF
