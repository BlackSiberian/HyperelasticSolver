#
# HyperelasticityMPh.jl
#
#
# Многофазная гиперупругость по Савенков, Алексеев
#

module HyperelasticityMPh

# FIXME: Implement it.

using ..EquationsOfState: density, energy, entropy, stress, EoS, Barton2009, Hank2016, pressure

using ..Strains: finger, invariants

export flux, initial_states, postproc_arrays, prim2cons_mph, cons2prim_mph

# Number of phases
const nph = 2

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

function prim2cons_mph(eos::T, P::Array{<:Any, 1}) where {T<:EoS}
    ph = Tuple(P[i:i+15-1] for i in 1:15:length(P))
    # Q = vcat(prim2cons.(eos, ph)...)
    Q = vcat([prim2cons(eos, P) for P in ph]...)
    return Q
end

function cons2prim_mph(eos::T, Q::Array{<:Any, 1}) where {T<:EoS}
    ph = Tuple(Q[i:i+15-1] for i in 1:15:length(Q))
    # P = vcat(cons2prim.(eos, ph)...)
    P = vcat([cons2prim(eos, Q) for Q in ph]...)
    return P
end

# ALERT: Temporary 15th primitive is pressure
function prim2cons(eos::T, P::Array{<:Any,1}) where {T<:EoS}
    Q = similar(P)
    frac = P[1]
    true_den = P[2]
    vel = [P[3], P[4], P[5]]
    distortion = P[6:14]
    # entropy = P[15]
    pressure = P[15]
    e_kin = 0.5 * (vel[1] + vel[2] + vel[3])

    G = finger(inv(reshape(distortion, (3, 3))))
    e_int = energy(eos, frac * true_den, pressure, G)

    e_total = e_int + e_kin

    Q[1] = frac
    Q[2] = frac * true_den
    for i in 1:3
        Q[2+i] = frac * true_den * vel[i]
    end
    Q[6:14] = distortion
    Q[15] = frac * true_den * e_total
    return Q
end


function cons2prim(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
    P = similar(Q)

    frac = Q[1] 
    true_den = Q[2] / frac
    vel = [Q[3], Q[4], Q[5]] / (frac * true_den)
    distortion = Q[6:14]
    e_kin = 0.5 * (vel[1] + vel[2] + vel[3])
    e_total = Q[15] / (frac * true_den)
    e_int = e_total - e_kin

    G = finger(inv(reshape(distortion, (3, 3))))
    i = invariants(G)
    pressure = pressure(eos, frac * true_den, e_int, i)
    # entropy = entropy() # TODO: write proper function call
    
    P[1] = frac
    P[2] = true_den
    for i in 1:3
        P[2+i] = vel[i]
    end
    for i in 1:9
        P[5+i] = Q[5+i]
    end
    P[15] = e_int
    
    return P
end

function flux_mph(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
    ph = Tuple(Q[i:i+15-1] for i in 1:15:length(Q))
    # flux = vcat(flux.(eos, ph)...)    
    ans = vcat([flux(eos, Q) for Q in ph]...)
    return ans
end

function flux(eos::T, Q::Array{<:Any,1}) where {T<:EoS}
    den = Q[2]
    vel = [Q[3], Q[4], Q[5]]
    distortion = Q[6:14]
    e_int = Q[15]
    
    strs = stress(eos, den, e_int, distortion)

    flux = similar(Q)

    flux[1] = 0
    flux[2] = den * vel[1]
    for i in 1:3
        flux[2+i] = den * vel[i] * vel[1] - strs[i]
    end
    for i in 1:9
        flux[5+i] = 0
    end
    flux[15] = den * vel[1] * e_int - strs[1] * vel[1] - strs[4] * vel[2] - strs[7] * vel[3] #TODO: check this out
    
    return flux
end

dname = "./hank_data/"
"""
    Задает левые и правые состояния для НУ, присваивание --- в основном коде.
    Эта фунция ничего не знает про сетку, но знает про физику.    
"""
function initial_states(eos::T, testcase::Int) where {T <: EoS}
    rho_1 = 2.7
    rho_2 = 1e-3

    alpha_l_1 = 1.0
    alpha_l_2 = 0.0

    den_l_1 = rho_1 * alpha_l_1
    den_l_2 = rho_2 * alpha_l_2

    u_l_1 = [400.0, 0.0, 0.0]
    u_l_2 = [400.0, 0.0, 0.0]

    A_l_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    A_l_2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    P_l_1 = 1e5
    P_l_2 = 1e5


    alpha_r_1 = 0.0
    alpha_r_2 = 1.0

    den_r_1 = rho_1 * alpha_r_1
    den_r_2 = rho_2 * alpha_r_2

    u_r_1 = [400.0, 0.0, 0.0]
    u_r_2 = [400.0, 0.0, 0.0]
    
    A_r_1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    A_r_2 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    P_r_1 = 1e5
    P_r_2 = 1e5


    Pl = [alpha_l_1, den_l_1, u_l_1..., A_l_1..., P_l_1, alpha_l_2, den_l_2, u_l_2..., A_l_2..., P_l_2]
    Pr = [alpha_r_1, den_r_1, u_r_1..., A_r_1..., P_r_1, alpha_r_2, den_r_2, u_r_2..., A_r_2..., P_r_2]

    Ql = prim2cons_mph(eos, Pl)
    Qr = prim2cons_mph(eos, Pr)

    return Ql, Qr
end # initial_states(eos::T, testcase::Int) where {T<:EoS}

"""
    Расчет значений массивов для визуализации.
    Возвращает сам массив и тюпл с аннотациями для переменных.
"""
function postproc_arrays(eos, Q0)
    nx = size(Q0)[2] 

    alpha = Array{Float64}(undef, nph, nx)
    true_den = Array{Float64}(undef, nph, nx)
    ent = Array{Float64}(undef, nph, nx)
    vel = Array{Float64}(undef, 3, nph, nx)
    A = Array{Float64}(undef, 9, nph, nx)
    
    for i in 1:size(Q0,2)
        P = cons2prim_mph(eos, Q0[:, i])

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
