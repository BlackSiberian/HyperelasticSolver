#
# Hyperelasticity.jl
#
# Hyperelastic GPR model as in Barton2002.

module Hyperelasticity

import LinearAlgebra: det, eigvals, dot

using ..EquationsOfState: density, acoustic, energy, entropy, stress, EoS, Barton2009
using ..Strains: finger, invariants

export prim2cons, cons2prim, flux, initial_states, postproc_arrays, get_eigvals


"""
    cons2prim(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}

Converts conservative variables to primitive variables
for onephase hyperelasticity with `eos` equation of state.
"""
function cons2prim(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  P = similar(Q)

  FQ = reshape(Q[5:13], (3, 3))
  den = sqrt(det(FQ) / eos.rho0)
  vel = Q[1:3] / den
  e_total = Q[4] / den
  e_kin = sum(vel .^ 2) / 2
  e_int = e_total - e_kin
  def_grad = Q[5:13] / den

  G = finger(def_grad)
  ent = entropy(eos, e_int, G)

  P[1:3] = vel
  P[4] = ent
  P[5:13] = def_grad
  return P
end


"""
    prim2cons(eos::T, P::Array{<:Any, 1}) where {T<:EoS}

Converts primitive variables to conservative variables for onephase
hyperelasticity with `eos` equation of state.
"""
function prim2cons(eos::T, P::Array{<:Any,1}) where {T<:EoS}
  Q = similar(P)

  vel = P[1:3]
  entropy = P[4]
  def_grad = P[5:13]
  den = eos.rho0 / det(reshape(def_grad, (3,3)))

  G = finger(def_grad)
  e_int = energy(eos, entropy, G)
  e_kin = sum(vel .^ 2) / 2
  e_total = e_int + e_kin

  Q[1:3] = den * vel
  Q[4] = den * e_total
  Q[5:13] = den * def_grad
  return Q
end

"""
    flux(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}

Computes the physical flux for onephase hyperelasticity with `eos` equation of state.
"""
function flux(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
  flux = similar(Q)

  FQ = reshape(Q[5:13], (3, 3))
  den = sqrt(det(FQ) / eos.rho0)

  vel = Q[1:3] / den
  e_total = Q[4] / den
  e_kin = sum(vel .^ 2) / 2
  e_int = e_total - e_kin
  def_grad = Q[5:13] / den

  G = finger(def_grad)
  ent = entropy(eos, e_int, G)
  strs = stress(eos, ent, def_grad)

  flux[1:3] = den * vel[1] * vel - strs[begin:3:end]
  flux[4] = den * vel[1] * e_total - sum(vel .* strs[begin:3:end])
  flux[5:13] = den .* (vel[1] .* def_grad - (vel*transpose(def_grad[begin:3:end]))[:])

  return flux
end

function get_eigvals(eos::T, Q::Array{<:Any,1}, n::Array{<:Any,1}) where {T<:EoS}
  P = cons2prim(eos, Q)
  vel = P[1:3]
  ent = P[4]
  def_grad = P[5:13]

  ac = acoustic(eos, ent, def_grad, n)
  # WARNING: No abs should be here. Eigvals must be non-negative
  sound_spd = sqrt.(abs.(eigvals(ac)))
  # sound_spd = sqrt.(eigvals(ac))
  spd = dot(vel, n)
  return vcat(spd .+ sound_spd, spd .- sound_spd)
end



# ##############################################################################
# Начальные условия и вывод.


"""
    Задает левые и правые состояния для НУ, присваивание --- в основном коде.
    Эта фунция ничего не знает про сетку, но знает про физику.    
"""
function initial_states(eos::T, testcase::Int) where {T <: EoS}
    if testcase == 1
        u_l = [0.0, 0.5, 1.0]       # velocity on the left boundary [km/s]
        F_l = [ 0.98  0.0   0.0;    # elastic deformation gradient tensor
                0.02  1.0   0.1;    # on the left boundary
                0.0   0.0   1.0]
        S_l = 1e-3                  # entropy on the left boundary [kJ/(g*K)]
        
        u_r = [0.0, 0.0, 0.0]       # velocity on the right boundary [km/s]
        F_r = [ 1.0    0.0   0.0;   # elastic deformation gradient tensor
                0.0    1.0   0.1;   # on the right boundary
                0.0    0.0   1.0]
        S_r = 0                     # entropy on the right boundary [kJ/(g*K)]
    elseif testcase == 2
        u_l = [2.0, 0.0, 0.1] # [km/s]
        F_l = [ 1.0     0.0   0.0 ;
                -0.01   0.95  0.02; 
                -0.015  0.0   0.9 ]
        S_l = 0.0 # [kJ/(g*K)]
        
        u_r = [0.0, -0.03, -0.01] # [km/s]
        F_r = [ 1.0     0.0     0.0;
                0.015   0.95    0.0;
                -0.01   0.0     0.9]
        S_r = 0.0 # [kJ/(g*K)]
    elseif testcase == 3
        u_l = [1.0, 0.0, 0.0] # [km/s]
        F_l = [0.5      -0.5*3^0.5      0.0;
               0.5*3^0.5    0.5         0.0;
               0.0          0.0         1.0]
        S_l = 0.0 # [kJ/(g*K)]
        u_r = [1.0, 0.0, 0.0] # [km/s]
        F_r = [0.5       -0.5*3^0.5     0.0;
               0.5*3^0.5    0.5         0.0;
               0.0          0.0         1.0]
        S_r = 0.0 # [kJ/(g*K)]
  elseif testcase == 4
    u_r = [0, -5, 0]
    F_r = [1 0 0; 0 1 0; 0 0 1]
    S_r = 1e-3

    u_l = [0, 5, 0]
    F_l = [1 0 0; 0 1 0; 0 0 1]
    S_l = 1e-3
    else
        u_l = u_r = zeros(3)
        F_l = F_r = [1 0 0; 0 1 0; 0 0 1]
        S_l = S_r = 0.0
    end

    P_l = [u_l..., S_l, F_l...]
    P_r = [u_r..., S_r, F_r...]

    Q_l = prim2cons(eos, P_l)
    Q_r = prim2cons(eos, P_r)

    return Q_l, Q_r
end # initial_states(eos::T, testcase::Int) where {T<:EoS}



"""
    Расчет значений массивов для визуализации.
    Возвращает сам массив и тюпл с аннотациями для переменных.
"""
# function postproc_arrays(Q0)
#     nx = size(Q0)[2] 
#     den  = Array{Float64}(undef, nx)
#     ent  = Array{Float64}(undef, nx)
#     vel  = Array{Float64, 2}(undef, 3, nx)
#     strs = Array{Float64, 3}(undef, 3, 3, nx)
#     eint = Array{Float64}(undef, nx)
#
#     for i in 1:size(Q0,2)
#         Q = Q0[:, i]
#         den[i], vel[:, i], F, e_int = cons2prim(Q)    # cons2prim должно возвращать вектор
#         local sigma = stress(den[i], e_int, F)        # stress должно возвращать вектор
#
#         # Это не нужно,
#         # invariants _уже умеет_ тензор как массив 3 на 3 и как строку длины 9
#         for j in 1:3
#             for k in 1:3
#                 strs[j, k, i] = sigma[k, j]
#             end
#         end
#         ent[i] = entropy(e_int, invariants(finger(F)))
#         eint[i] = e_int
#     end
#
#     info = ("den", "ent", "vel", "strs", "eint", "info")
#     return den, ent, vel, strs, eint, info
# end


end # module Hyperelasticity

# EOF

