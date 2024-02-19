#
# HyperelasticityMPh.jl
#
#
# Многофазная гиперупругость по Савенков, Алексеев
#

module HyperelasticityMPh

# FIXME: Implement it.


# Number of phases
const nph = 2

# 1. Для многофазной задачи номер фазы --- всегда параметр в массиве.
#
# 2. cons2prim и prim2cons --- принмают вектора нужной длины и отдают
#    вектора такой же длины (не так, как сейчас для простой
#    гиперупугсти).
#                               
# 3. flux() --- принимает <<консервативные>> переменные
#    (которые под производной по времени), отдает три величины:
#    физический поток, его якобиан и неконсервативную матрицу B  
#
# 4. Так как фазы независимы по УрС, то переход от первичных к
#    консерватиыным и наоборот, вычисление энтропии, энергии и так далее
#    --- всегда можно сделать локально для каждой фазы.
#    
# 5. Как можно реже переводить вектор длины 9 в тензор и наоборот.
#    На самом деле, нам практически никогда не нужны тензоры как
#    матрицы 3 на 3.
#
#    Возможно, потребуется дописать какие то еще методы с тем же имененм
#    и другими интерфейсами.
#
# 6. Помнить, что теперь EquationsOfStates.jl --- для всех задач.
#    Поэтому неоьзя ломать совместимость. Дописываем методы, не трогая
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

function prim2cons_mph(P::Array{<:Any, 1})
    ph = Tuple(P[i:i+nph-1] for i in 1:nph:length(P))
    Q = vcat(prim2cons.(ph)...)
    return Q
end

function cons2prim_mph(Q::Array{<:Any, 1})
    ph = Tuple(Q[i:i+nph-1] for i in 1:nph:length(Q))
    P = vcat(cons2prim.(ph)...)
    return P
end

function prim2cons(P::Array{<:Any,1})
    Q = similar(P)
    frac = P[1]
    true_den = P[2]
    vel = [P[3], P[4], P[5]]
    entropy = P[15]
    e_kin = 0.5 * (vel[1] + vel[2] + vel[3])
    e_int = energy(eos, frac * true_den, ) # TODO: write proper function call

    Q[1] = frac
    Q[2] = frac * true_den
    for i in 1:3
        Q[2+i] = frac * true_den * vel[i]
    end
    for i in 1:9
        Q[5+i] = P[5+i]
    end
    Q[15] = frac * true_den * e_int
    return Q
end


function cons2prim(Q::Array{<:Any,1})
    P = similar(Q)

    frac = Q[1] 
    true_den = Q[2] / frac
    vel = [Q[3], Q[4], Q[5]] / (frac * true_den)
    e_int = Q[15] / (frac * true_den)
    entropy = entropy() # TODO: write proper function call
    
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

function flux_mph(Q::Array{<:Any,1})
    ph = Tuple(Q[i:i+nph-1] for i in 1:nph:length(Q))
    flux = vcat(flux.(ph)...)    
end

function flux(Q::Array{<:Any,1})
    den = Q[2]
    vel = [Q[3], Q[4], Q[5]]
    e_int = Q[15]
    
    stress = stress() #TODO: write proper function call

    flux = similar(Q)

    flux[1] = 0
    flux[2] = den * vel[1]
    for i in 1:3
        flux[2+i] = den * vel[i] * vel[1] - stress[i]
    end
    for i in 1:9
        flux[5+i] = 0
    end
    flux[15] = den * vel[1] * e_int - stress[1] * vel[1] - stress[4] * vel[2] - stress[7] * vel[3] #TODO: check this out
    
    return flux
end

end # module HyperelasticityMPh

# EOF
