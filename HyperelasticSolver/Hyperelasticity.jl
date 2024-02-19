#
# Hyperelasticity.jl
#
#
# Hyperelastic GPR model as in Barton2002.
#
#
#

module Hyperelasticity
import LinearAlgebra: det

#include("./EquationsOfState.jl")
using ..EquationsOfState: density, energy, entropy, stress, EoS, Barton2009

using ..Strains: finger, invariants

export flux, initial_states, postproc_arrays


"""
    Translates conservative variables to primitive variables.
    @param Q is the conservative variables vector
"""
function cons2prim(Q::Array{<:Any,1})
    FQ = [ Q[4]  Q[5]  Q[6];                              
           Q[7]  Q[8]  Q[9];
           Q[10] Q[11] Q[12]] 
    den = density(Q[4:12])
    vel = Q[1:3] ./ den
    F = FQ ./ den
    E = Q[13] / den
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    e_int = E - e_kin
    return [den, vel, F, e_int]
end


"""
    Translates primitive variables to conservative variables
    @param vel is the velocity vector
    @param S is the entropy
    FIXME: Здесь F --- 3 на 3, но некоторые функции требуют линейного массива
    длиной 9. Нужно все согласовать, то есть сверху передавать толко
    линейный
    или только массив 3 на 3.

    Линейный массив упорядочен по строкам.

    Это должно быть отражено в типе параметра.

    Если тут передавать линейный массив, то можно использовать density
    из EquationsOfState.jl 
    ---- нельзя, так как требуется rho * F, а здесь просто F.
    Но можно переопределить ее по типу, если сделать типы тензоров.

    Текущее решение:
        Не меняем интерфейсы, делаем reshape каждый раз, когда нужно

    @param: F is [3,3] array

    !
    ! Вообще, эта функция должна получать на вход дин вектор длиной 13
    ! и одавать второй вектор длиной 13
    !

"""
# Cложные типы для скорости и тензора нужны, чтобы проверялись размерности.
# Далее это бдует нужно для правильной параметризации.
function prim2cons(eos::T, vel::Array{<:Any,1}, F::Array{<:Any,2}, S) where {T <: EoS}
    Q = Array{Float64}(undef, 13)

    # ALERT:
    # TODO:
    # FIXME:
    #     Из-за rho0 сецйча сюда трябуется тянуть ::EoS.
    #     Этого можно избежать, если иметь  два метода:
    #     density(F::DefGrad) и density(F:Finger)
    den = eos.rho0 / det(F)
    
    Q[1:3] = den .* vel
    for i in 1:3
        for j in 1:3
            Q[3*i + j] = den * F[i, j]
        end
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    G = finger(F) 
    i = invariants(G)
    E = energy(S, i) + e_kin
    Q[13] = den * E
    return Q
end

"""
    Returns the value of the physical flux in Q.
    @param Q is the conservative variables vector.
"""
function flux(Q::Array{<:Any,1}) 
    den, vel, F, e_int = cons2prim(Q)
    sigma = stress(den, e_int, F)

    flux = similar(Q)
    for i in 1:3
        flux[i] = den * vel[1] * vel[i] - sigma[1, i]
        flux[i+3] = 0
        flux[i+6] = den * (F[2, i] * vel[1] - F[1, i] * vel[2])
        flux[i+9] = den * (F[3, i] * vel[1] - F[1, i] * vel[3])
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    E = e_int + e_kin
    flux[13] = den * vel[1] * E - vel[1] * sigma[1, 1] - vel[2] * sigma[1, 2] - vel[3] * sigma[1, 3]
    return flux
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
    else 
        u_l = u_r = zeros(3)
        F_l = F_r = Array{Float64}(I, 3, 3)
        S_l = S_r = 0.0
    end


    Ql = prim2cons(eos, u_l, F_l, S_l)
    Qr = prim2cons(eos, u_r, F_r, S_r)

    return Ql, Qr
end # initial_states(eos::T, testcase::Int) where {T<:EoS}



"""
    Расчет значений массивов для визуализации.
    Возвращает сам массив и тюпл с аннотациями для переменных.
"""
function postproc_arrays(Q0)
    nx = size(Q0)[2] 
    den  = Array{Float64}(undef, nx)
    ent  = Array{Float64}(undef, nx)
    vel  = Array{Float64, 2}(undef, 3, nx)
    strs = Array{Float64, 3}(undef, 3, 3, nx)
    eint = Array{Float64}(undef, nx)
    
    for i in 1:size(Q0,2)
        Q = Q0[:, i]
        den[i], vel[:, i], F, e_int = cons2prim(Q)    # cons2prim должно возвращать вектор
        local sigma = stress(den[i], e_int, F)        # stress должно возвращать вектор
        
        # Это не нужно,
        # invariants _уже умеет_ тензор как массив 3 на 3 и как строку длины 9
        for j in 1:3
            for k in 1:3
                strs[j, k, i] = sigma[k, j]
            end
        end
        ent[i] = entropy(e_int, invariants(finger(F)))
        eint[i] = e_int
    end

    info = ("den", "ent", vel, "strs", "eint", "info")
    return den, ent, vel, strs, eint, info
end


end # module Hyperelasticity

# EOF

