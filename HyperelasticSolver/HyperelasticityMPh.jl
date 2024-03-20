#
# HyperelasticityMPh.jl
#
# Multiphase hyperelasticity (Savenkov, Alekseev, 2022)

module HyperelasticityMPh

using LinearAlgebra: inv, transpose, det
using ..EquationsOfState: energy, entropy, stress, EoS, Barton2009, Hank2016
using ..Strains: finger, invariants

export prim2cons_mph, cons2prim_mph, flux_mph, initial_states, postproc_arrays 

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
function prim2cons_mph(eos::T, P::Array{<:Any, 1}) where {T<:EoS}
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
function cons2prim_mph(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}
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
        
        # G = finger(inv(reshape(distortion, (3, 3))))
        # G = finger(distortion)
        G = finger(def_grad)
        i = invariants(G)
        # pressure = pressure(eos, true_den, e_int, i) # TODO: rewrite pressure to entropy
        # entropy = entropy(eos, e_int, G)
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

    # FQ = reshape(Q[7:15], 3, 3)   
    # den = sqrt(det(FQ) / 8.93)

    true_den = den / frac
    vel = Q[3:5] / den
    e_total = Q[6] / den
    e_kin = sum(vel .^ 2) / 2
    e_int = e_total - e_kin
    def_grad = Q[7:15] / den
    
    # strs = stress(eos, true_den, e_int, distortion)
    strs = stress(eos, true_den, e_int, def_grad)

    flux = similar(Q)

    flux[1] = 0
    flux[2] = den * vel[1]
    flux[3:5] = den * vel[1] * vel - strs[begin:3:end]
    flux[6] = den * vel[1] * e_total - sum(vel .* strs[begin:3:end])
    # flux[7:15] = den * (def_grad * vel[1] - reshape(def_grad[begin:3:end] * transpose(vel), 9))
    # flux[7:15] = den .* (vel[1] .* def_grad - (def_grad[begin:3:end] * transpose(vel))[:])
    flux[7:15] = den .* (vel[1] .* def_grad - (vel * transpose(def_grad[begin:3:end]))[:])
    
    return flux
end

"""
    initial_states(eos::T, testcase::Int) where {T<:EoS}

Sets left and right states for the Riemann problem for multiphase hyperelasticity with `eos` equation of state

'testcase' is the test case number
"""
function initial_states(eos::T, testcase::Int) where {T <: EoS}
# Эта функция ничего не знает про сетку, но знает про физику. 
    # rho_1 = 2.7
    # rho_2 = 1e-3

    # alpha_l_1 = 1.0
    # alpha_l_2 = 0.0

    # den_l_1 = rho_1 * alpha_l_1
    # den_l_2 = rho_2 * alpha_l_2

    # u_l_1 = [400.0, 0.0, 0.0]
    # u_l_2 = [400.0, 0.0, 0.0]

    # A_l_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # A_l_2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    # P_l_1 = 1e5
    # P_l_2 = 1e5


    # alpha_r_1 = 0.0
    # alpha_r_2 = 1.0

    # den_r_1 = rho_1 * alpha_r_1
    # den_r_2 = rho_2 * alpha_r_2

    # u_r_1 = [400.0, 0.0, 0.0]
    # u_r_2 = [400.0, 0.0, 0.0]
    
    # A_r_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # A_r_2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # P_r_1 = 1e5
    # P_r_2 = 1e5


    # Pl = [alpha_l_1, den_l_1, u_l_1..., A_l_1..., P_l_1, alpha_l_2, den_l_2, u_l_2..., A_l_2..., P_l_2]
    # Pr = [alpha_r_1, den_r_1, u_r_1..., A_r_1..., P_r_1, alpha_r_2, den_r_2, u_r_2..., A_r_2..., P_r_2]

    # alpha_l_1 = alpha_l_2 = 0.5
    # alpha_r_1 = alpha_r_2 = 0.5

    
    # alpha_l_1 = alpha_r_1 = 0.999
    # alpha_l_2 = alpha_r_2 = 0.001
    
    # u_l_1 = u_l_2 = [2.0, 0.0, 0.1] # [km/s]

    # F_l = [ 1.0     0.0     0.0;
    #         -0.01   0.95    0.02;
    #         -0.015  0.0     0.9]
    # F_l_1 = F_l_2 = reshape(F_l, length(F_l))
    
    # S_l_1 = S_l_2 = 0.0 # [kJ/(g*K)]
    
    
    # u_r_1 = u_r_2 = [0.0, -0.03, -0.01] # [km/s]

    # F_r = [ 1.0     0.0     0.0;
    #         0.015   0.95    0.0;
    #         -0.01   0.0     0.9]
    # F_r_1 = F_r_2 = reshape(F_r, length(F_r))
    
    # S_r_1 = S_r_2 = 0.0 # [kJ/(g*K)]

    # den_l_1 = den_l_2 = 8.93 / det(F_l)   
    # den_r_1 = den_r_2 = 8.93 / det(F_r)
    

    
    # alpha_l_1 = alpha_r_1 = 0.5
    # alpha_l_2 = alpha_r_2 = 0.5
    
    # u_l_1 = u_l_2 = [1.0, 0.0, 0.0] # [km/s]
    
    # F_l = [ 1.0     0.0     0.0;
    #         0.0     1.0     0.0;
    #         0.0     0.0     1.0 ]
    # F_l_1 = F_l_2 = reshape(F_l, length(F_l))
    
    # S_l_1 = S_l_2 = 10.0 # [kJ/(g*K)]
    
    
    # u_r_1 = u_r_2 = [1.0, 0.0, 0.0] # [km/s]

    # F_r = [ 1.0     0.0     0.0;
    #         0.0     1.0     0.0;
    #         0.0     0.0     1.0 ]
    # F_r_1 = F_r_2 = reshape(F_r, length(F_r))
    
    # S_r_1 = S_r_2 = 1.0 # [kJ/(g*K)]

    # den_l_1 = den_l_2 = 8.93 # Maybe that isn't correct  
    # den_r_1 = den_r_2 = 8.93 # Maybe that isn't correct
    
    u_l_1 = u_l_2 = [2.0, 0.0, 0.1] # [km/s]
    F_l = [ 1.0     0.0   0.0 ;
            -0.01   0.95  0.02; 
            -0.015  0.0   0.9 ]
    F_l_1 = F_l_2 = F_l
    # F_l_1 = F_l_2 = reshape(F_l, length(F_l))
    S_l_1 = S_l_2 = 0.0 # [kJ/(g*K)]
    
    u_r_1 = u_r_2 = [0.0, -0.03, -0.01] # [km/s]
    F_r = [ 1.0     0.0     0.0;
            0.015   0.95    0.0;
            -0.01   0.0     0.9]
    F_r_1 = F_r_2 = F_r
    # F_r_1 = F_r_2 = reshape(F_r, length(F_r))
    S_r_1 = S_r_2 = 0.0 # [kJ/(g*K)]

    alpha_l_1 = alpha_l_2 = 0.5
    alpha_r_1 = alpha_r_2 = 0.5

    den_l_1 = den_l_2 = 8.93 / det(F_l)
    den_r_1 = den_r_2 = 8.93 / det(F_r)


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
function postproc_arrays(eos, Q0)
    nx = size(Q0)[2] 

    alpha = Array{Float64}(undef, nph, nx)
    true_den = Array{Float64}(undef, nph, nx)
    ent = Array{Float64}(undef, nph, nx)
    vel = Array{Float64}(undef, 3, nph, nx)
    A = Array{Float64}(undef, 9, nph, nx)
    
    for (i, Q) in enumerate(eachcol(Q0))
        P = cons2prim_mph(eos, Q)

        alpha[:, i] = P[begin:15:end]
        true_den[:, i] = P[begin+1:15:end]
        for j = 1:3 
            vel[j, :, i] = P[begin+1+j:15:end]
        end
        for j = 1:9 
            A[j, :, i] = P[begin+5+j:15:end]
        end
        ent[:, i] = P[begin+14:15:end]        
        
        ent[i] = entropy(e_int, invariants(finger(F)))
        eint[i] = e_int
    end

    info = ("Fraction", "True Density", "Velocity", "Distortion", "Entropy", "info")
    return alpha, true_den, vel, A, ent, info
end

end # module HyperelasticityMPh

# EOF