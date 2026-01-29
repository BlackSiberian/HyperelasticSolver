#
# RelaxationInst.jl
#

module RelaxationInst

using ..EquationsOfState
using ..HyperelasticityMPh
using ..Strains
using ..SimpleLA

using LinearAlgebra: dot, I
using ..SimpleLA
using NLsolve

export relaxation_inst

function relaxation_inst(eos::Tuple{T,T}, Q::Array{<:Any,1}) where {T <: EoS}
    relaxed = similar(Q)

    relaxed_vel = relaxation_vel(eos, Q)

    relaxed = relaxation_pres(eos, relaxed_vel)

    return relaxed
end

function relaxation_vel(eos::Tuple{T,T}, Q::Array{<:Any,1}) where {T <: EoS}
    # Преобразование консервативных переменных в примитивные
    P = cons2prim_mph(eos, Q)
    # Разделение вектора P на два компонента
    P = [P[i:i+14] for i in 1:15:length(P)]

    # Извлечение компонентов из вектора P
    frac = [P[i][1] for i in 1:2] # volume fraction
    true_den = [P[i][2] for i in 1:2] # true density
    den = frac .* true_den # density
    vel = [P[i][3:5] for i in 1:2] # velocity
    ent = [P[i][6] for i in 1:2] # entropy
    def_grad = [P[i][7:15] for i in 1:2] # deformation gradient
    # Вычисление инварианта finger для deformation_gradient один раз
    G = [finger(def_grad[i]) for i in 1:2] # first invariant of finger
    e_int = [energy(eos[i], ent[i], G[i]) for i in 1:2] # internal energy

    # Вычисление релаксированной скорости
    rel_vel = (den[1] * vel[1] + den[2] * vel[2]) / (den[1] + den[2])
    # Вычисление промежуточной скорости
    int_vel = 1/2 * (vel[1] + vel[2])

    # Вычисление релаксированной внутренней энергии
    rel_en = [e_int[i] + 1/2 * dot(int_vel + rel_vel, rel_vel - vel[i]) - 1/2 * (dot(rel_vel, rel_vel) - dot(vel[i], vel[i])) for i in 1:2]

    # Обновление компонентов вектора P
    for i in 1:2
        P[i][3:5] = rel_vel
        P[i][6] = entropy(eos[i], rel_en[i], G[i])
    end

    # Сборка вектора P
    P = vcat(P[1], P[2])
    # Преобразование примитивных переменных обратно в консервативные
    Q = prim2cons_mph(eos, P)

    return Q
end

function relaxation_vel_newton(eos::Tuple{T,T}, Q::Array{<:Any,1}) where {T<:EoS}
    # Преобразование консервативных переменных в примитивные
    P = cons2prim_mph(eos, Q)

    P0 = copy(P)
    # Разделение вектора P на два компонента
    P0 = [P0[i:i+14] for i in 1:15:length(P)]

    # Извлечение компонентов из вектора P
    frac0 = [P0[i][1] for i in 1:2] # volume fraction
    true_den0 = [P0[i][2] for i in 1:2] # true density
    den0 = frac0 .* true_den0 # density
    vel0 = [P0[i][3:5] for i in 1:2] # velocity
    ent0 = [P0[i][6] for i in 1:2] # entropy
    def_grad0 = [P0[i][7:15] for i in 1:2] # deformation gradient
    # Вычисление инварианта finger для deformation_gradient один раз
    G = [finger(def_grad0[i]) for i in 1:2] # first invariant of finger
    e_int0 = [energy(eos[i], ent0[i], G[i]) for i in 1:2] # internal energy

    function f!(F, P)
        # Разделение вектора P на два компонента
        P = [P[i:i+14] for i in 1:15:length(P)]

        # Извлечение компонентов из вектора P
        frac = [P[i][1] for i in 1:2] # volume fraction
        true_den = [P[i][2] for i in 1:2] # true density
        den = frac .* true_den # density
        vel = [P[i][3:5] for i in 1:2] # velocity
        ent = [P[i][6] for i in 1:2] # entropy
        def_grad = [P[i][7:15] for i in 1:2] # deformation gradient
        # Вычисление инварианта finger для deformation_gradient один раз
        G = [finger(def_grad[i]) for i in 1:2] # first invariant of finger
        e_int = [energy(eos[i], ent[i], G[i]) for i in 1:2] # internal energy

        # Вычисление релаксированной скорости
        rel_vel = (den[1] * vel0[1] + den[2] * vel0[2]) / (den[1] + den[2])
        # Вычисление промежуточной скорости
        int_vel = 1/2 * (vel[1] + vel[2])
        # Вычисление релаксированной внутренней энергии
        rel_en = [e_int[i] + 1/2 * dot(int_vel + rel_vel, rel_vel - vel[i]) - 1/2 * (dot(rel_vel, rel_vel) - dot(vel[i], vel[i])) for i in 1:2]

        F[1] = frac[1] - frac0[1]
        F[2] = den[1] - den0[1]
        F[3:5] = vel[1] - rel_vel
        F[6] = e_int[1] - e_int0[1] - 1/2 * dot(int_vel + rel_vel, rel_vel - vel0[1]) + 1/2 * (dot(rel_vel, rel_vel) - dot(vel0[1], vel0[1]))
        F[7:15] = def_grad[1] - def_grad0[1]
        F[16] = frac[2] - frac0[2]
        F[17] = den[2] - den0[2]
        F[18:20] = vel[2] - rel_vel
        F[21] = e_int[2] - e_int0[2] - 1/2 * dot(int_vel + rel_vel, rel_vel - vel0[2]) + 1/2 * (dot(rel_vel, rel_vel) - dot(vel0[2], vel0[2]))
        F[22:30] = def_grad[2] - def_grad0[2]
    end

    # Сборка вектора P0
    P_guess = vcat(P0[1], P0[2])

    sol = nlsolve(f!, P_guess; method=:newton, autodiff=:forward)

    check_sol = similar(P_guess)
    f!(check_sol, sol.zero)
    println(check_sol)

    # Преобразование примитивных переменных обратно в консервативные
    Q = prim2cons_mph(eos, sol.zero)

    return Q
end


function relaxation_pres(eos::Tuple{T,T}, Q::Array{<:Any,1}) where {T<:EoS}
    # Преобразование консервативных переменных в примитивные
    P = cons2prim_mph(eos, Q)

    P0 = copy(P)
    # Разделение вектора P на два компонента
    P0 = @views [P0[i:i+14] for i in 1:15:length(P)]

    # Извлечение компонентов из вектора P
    frac0 = [P0[i][1] for i in 1:2] # volume fraction
    true_den0 = [P0[i][2] for i in 1:2] # true density
    den0 = frac0 .* true_den0 # density
    vel0 = [P0[i][3:5] for i in 1:2] # velocity
    ent0 = [P0[i][6] for i in 1:2] # entropy
    def_grad0 = [P0[i][7:15] for i in 1:2] # deformation gradient
    # Вычисление инварианта finger для deformation_gradient один раз
    G = [finger(def_grad0[i]) for i in 1:2] # first invariant of finger
    e_int0 = [energy(eos[i], ent0[i], G[i]) for i in 1:2] # internal energy
    strs0 = [reshape(frac0[i] * stress(eos[i], ent0[i], def_grad0[i]), (3, 3)) for i in 1:2]
    pres0 = [- 1/3 * tr_(strs0[i]) for i in 1:2]

    beta = zeros(2)
    K0 = [1 / frac0[i] * strs0[i] + beta[i] * I for i in 1:2]

    mu0 = 1/2
    mu0 = [mu0, 1 - mu0]
    # pi0 = - 1/3 * tr_(mu0[1] * K0[1] + mu0[2] * K0[2])
    pi0 = pres0[1]

    function f!(F, P)
        # Разделение вектора P на два компонента
        P = @views [P[i:i+14] for i in 1:15:length(P)]

        # Извлечение компонентов из вектора P
        frac = [P[i][1] for i in 1:2] # volume fraction
        true_den = [P[i][2] for i in 1:2] # true density
        den = frac .* true_den # density
        vel = [P[i][3:5] for i in 1:2] # velocity
        ent = [P[i][6] for i in 1:2] # entropy
        def_grad = [P[i][7:15] for i in 1:2] # deformation gradient
        # Вычисление инварианта finger для deformation_gradient один раз
        G = [finger(def_grad[i]) for i in 1:2] # first invariant of finger
        e_int = [energy(eos[i], ent[i], G[i]) for i in 1:2] # internal energy
        # pres = [pressure(eos[1], true_den[1], e_int[1]) for i in 1:2] # pressure
        strs = [reshape(stress(eos[i], ent[i], def_grad[i]), (3, 3)) for i in 1:2]
        pres = [- 1/3 * tr_(strs[i]) for i in 1:2]

        beta = zeros(2)
        K = [1 / frac[i] * strs[i] + beta[i] * I for i in 1:2]

        mu = 1/2
        mu = [mu, 1 - mu]
        pi = - 1/3 * tr_(mu[1] * K[1] + mu[2] * K[2])

        pi_i = 1/2 * (pi + pi0)

        F[1] = 1 - frac[1] - frac[2]
        F[2] = den[1] - den0[1]
        F[3:5] = den[1] * vel[1] - den0[1] * vel0[1]
        F[6] = e_int[1] - e_int0[1] + pi_i / den[1] * (frac[1] - frac0[1])
        F[7:15] = den[1] * def_grad[1] - (den0[1] * def_grad0[1]) * cbrt(frac[1] / frac0[1])
        F[16] = pres[1] - pres[2]
        F[17] = den[2] - den0[2]
        F[18:20] = den[2] * vel[2] - den0[2] * vel0[2]
        F[21] = e_int[2] - e_int0[2] + pi_i / den[2] * (frac[2] - frac0[2])
        F[22:30] = den[2] * def_grad[2] - (den0[2] * def_grad0[2]) * cbrt(frac[2] / frac0[2])
    end

    function g!(F, R)
        # Разделение вектора R на два компонента
        R = [R[i:i+10] for i in 1:11:length(R)]

        # Извлечение компонентов из вектора R
        frac = [R[i][1] for i in 1:2] # volume fraction
        # true_den = [den0[i] / frac[i] for i in 1:2]
        # den = frac .* true_den # density
        den = den0
        ent = [R[i][2] for i in 1:2] # entropy
        def_grad = [R[i][3:11] / den0[i] for i in 1:2] # deformation gradient
        # Вычисление инварианта finger для deformation_gradient один раз
        G = [finger(def_grad[i]) for i in 1:2] # first invariant of finger
        e_int = [energy(eos[i], ent[i], G[i]) for i in 1:2] # internal energy
        # pres = [pressure(eos[1], true_den[1], e_int[1]) for i in 1:2] # pressure
        strs = [reshape(stress(eos[i], ent[i], def_grad[i]), (3, 3)) for i in 1:2]
        pres = [- 1/3 * tr_(strs[i]) for i in 1:2]

        beta = zeros(2)
        K = [1 / frac[i] * strs[i] + beta[i] * I for i in 1:2]

        mu = 1/2
        mu = [mu, 1 - mu]
        pi = - 1/3 * tr_(mu[1] * K[1] + mu[2] * K[2])

        pi_i = 1/2 * (pi + pi0)
        pi = pres[1]

        F[1] = 1 - frac[1] - frac[2]
        F[2] = pres[1] - pres[2]
        F[3] = e_int[1] - e_int0[1] + pi_i / den[1] * (frac[1] - frac0[1])
        F[4] = e_int[2] - e_int0[2] + pi_i / den[2] * (frac[2] - frac0[2])
        F[5:13] = den[1] * def_grad[1] - (den0[1] * def_grad0[1]) * cbrt(frac[1] / frac0[1])
        F[14:22] = den[2] * def_grad[2] - (den0[2] * def_grad0[2]) * cbrt(frac[2] / frac0[2])
    end


    # Сборка вектора P0
    P_guess = vcat(P0[1], P0[2])
    R_guess = Array{Float64}(undef, 22)
    R_guess[1] = P_guess[1]
    R_guess[2] = P_guess[6]
    R_guess[3:11] = Q[7:15]
    R_guess[12] = P_guess[16]
    R_guess[13] = P_guess[21]
    R_guess[14:22] = Q[22:30]

    # TODO: Check if converge
    # sol = nlsolve(f!, P_guess; method=:newton, autodiff=:forward, store_trace=false, show_trace=false, extended_trace=false)
    # sol = nlsolve(g!, R_guess; method=:newton, autodiff=:forward, store_trace=false, show_trace=false, extended_trace=false, ftol=1e-12)
    sol = nlsolve(g!, R_guess)

    # check_sol = similar(P_guess)
    # f!(check_sol, sol.zero)
    # check_sol = similar(R_guess)
    # g!(check_sol, sol.zero)
    # println(check_sol)
    println(sol)

    R_exit = sol.zero
    P_exit = similar(P)
    P_exit[1] = R_exit[1]
    P_exit[2] = den0[1] / R_exit[1]
    P_exit[3:5] = vel0[1]
    P_exit[6] = R_exit[2]
    P_exit[7:15] = R_exit[3:11] / den0[1]
    P_exit[16] = R_exit[12]
    P_exit[17] = den0[2] / R_exit[12]
    P_exit[18:20] = vel0[2]
    P_exit[21] = R_exit[13]
    P_exit[22:30] = R_exit[14:22] / den0[2]
    println("Fraction = ", P_exit[1], " ", P_exit[16])
    println("True density = ", P_exit[2], " ", P_exit[17])
    println("Velocity = ", P_exit[3:5], P_exit[18:20])
    println("Entropy = ", P_exit[6], P_exit[21])
    display(reshape(P_exit[7:15], 3, 3))
    display(reshape(P_exit[22:30], 3, 3))
    println(P_exit[1:15])
    println(P_exit[16:30])

    # P_exit = sol.zero

    return prim2cons_mph(eos, P_exit)
end

end # module RelaxationInst

# EOF
