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

    beta = zeros(2)
    K0 = [1 / frac0[i] * strs0[i] + beta[i] * I for i in 1:2]

    mu0 = 1/2
    mu0 = [mu0, 1 - mu0]
    pi0 = - 1/3 * tr_(mu0[1] * K0[1] + mu0[2] * K0[2])

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
        strs = [reshape(frac[i] * stress(eos[i], ent[i], def_grad[i]), (3, 3)) for i in 1:2]
        pres = [- 1/3 * tr_(strs[i]) for i in 1:2]

        beta = zeros(2)
        K = [1 / frac[i] * strs[i] + beta[i] * I for i in 1:2]

        mu = 1/2
        mu = [mu, 1 - mu]
        pi = - 1/3 * tr_(mu[1] * K[1] + mu[2] * K[2])

        pi_i = 1/2 * (pi - pi0)

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

    # Сборка вектора P0
    P_guess = vcat(P0[1], P0[2])

    # TODO: Check if converge
    sol = nlsolve(f!, P_guess; method=:newton, autodiff=:forward, store_trace=false, show_trace=false, extended_trace=false)

    check_sol = similar(P_guess)
    f!(check_sol, sol.zero)
    println(check_sol)

    check_R = similar(P_guess)
    R = [0.538048, 10.2284, 0.378661, -0.00591818, 0.0255848, 8.58408e-05, 0.938767, -0.0012452, -0.00445142, 0.0112462, 0.990819, 0.00172915, -0.0133126, -3.32515e-07, 0.938671,
        0.461952, 8.749, 0.378661, -0.00591818, 0.0255848, 0.00106794, 0.931328, -0.00137763, -0.00484703, -0.00870159, 1.07564, 0.0199322, -0.0216253, -6.56284e-06, 1.019]
    f!(check_R, R)
    println(check_R)
# a1 = 0.538048
# a2 = 0.461952
# gamma1 = 10.2284
# gamma2 = 8.749
# velocity 1 = 0.378661, -0.00591818, 0.0255848,
# velocity 2 = 0.378661, -0.00591818, 0.0255848,
# F1 = 0.938767 0.0112462 -0.0133126
    # -0.0012452 0.990819 -3.32515e-07
    # -0.00445142 0.00172915 0.938671
# F2 = 0.931328 -0.00870159 -0.0216253
#     -0.00137763 1.07564 -6.56284e-06
#     -0.00484703 0.0199322 1.019
# S1 = 8.58408e-05
# s2 = 0.00106794

# p1 = 27.5525
# p2 = 27.5525
    println(sol)

    # Преобразование примитивных переменных обратно в консервативные
    Q = prim2cons_mph(eos, sol.zero)

    return Q
end

end # module RelaxationInst

# EOF
