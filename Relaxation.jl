#
# Relaxation.jl
#

module Relaxation

using ForwardDiff: derivative, tr, norm
using DifferentialEquations
using LinearAlgebra
using ..EquationsOfState: energy, entropy, stress, EoS
using ..Strains: finger
using ..SimpleLA

using Plots
using LaTeXStrings
using Printf

export relaxation

pyplot()

function relaxation(eos::Tuple{T,T}, initial::Array{<:Any,1}, dt) where {T<:EoS}
    # toggle_print = false
    toggle_print = true

    t_span = (0, dt)

    tau_a = dt * 100
    # tau_u = dt / 1000
    # tau_u = dt / 200000
    tau_u = dt / 200
    tau_t = dt * 50

    # tau_a = 1e10
    # tau_t = 1e10

    function get_vars(eos::Tuple{<:EoS,<:EoS}, Q::Array{<:Any,1})
        Q = [Q[p:p+14] for p in [1, 16]]
        nph = length(Q)
        frac = [Q[p][1] for p in 1:nph]

        FQ = [reshape(Q[p][7:15] ./ frac[p], (3, 3)) for p in 1:nph]
        true_den = [sqrt(det(FQ[p]) / eos[p].rho0) for p in 1:nph]
        den = frac .* true_den

        vel = [Q[p][3:5] / den[p] for p in 1:nph]
        e_total = [Q[p][6] / den[p] for p in 1:nph]
        e_kin = [sum(vel[p] .^ 2) / 2 for p in 1:nph]
        e_int = e_total - e_kin
        def_grad = [Q[p][7:15] / den[p] for p in 1:nph]

        G = [finger(def_grad[p]) for p in 1:nph]
        ent = [entropy(eos[p], e_int[p], G[p]) for p in 1:nph]
        strs = [reshape(frac[p] .* stress(eos[p], ent[p], def_grad[p]), (3, 3)) for p in 1:nph]

        temp = [derivative(S -> energy(eos[p], S, G[p]), ent[p]) for p in 1:nph]
        pres = [-1 / 3 * tr(s) for s in strs]

        return (vel[1], vel[2]), (pres[1], pres[2]), (temp[1], temp[2]), (e_total[1], e_total[2]), (e_kin[1], e_kin[2]), (e_int[1], e_int[2])
    end

    function print_tol(pair::Tuple{<:Any,<:Any}, name::String)
        a_tol = norm(pair[1] - pair[2])
        r_tol = 2a_tol / norm(pair[1] + pair[2])
        @printf("%s: \ta_tol = %.2e \t r_tol = %.2e\n", name, a_tol, r_tol)
    end

    if toggle_print
        println("Before relaxation")
        vel, pres, temp = get_vars(eos, initial)
        print_tol(vel, "velocity")
        print_tol(pres, "pressure")
        print_tol(temp, "temperature")
    end

    problem = ODEProblem(init_relaxation, initial, t_span, (eos, tau_a, tau_u, tau_t, 1.0, 1.0, 1.0))
    # solution = solve(problem, TRBDF2(), dtmax= min(tau_a, tau_u, tau_t) / 10)
    # solution = solve(problem, TRBDF2(), abstol=1e-4, reltol=1e-1)
    # solution = solve(problem, TRBDF2(), abstol=1e-3, reltol=1.0)

    solution = solve(problem, TRBDF2())
    # relaxed = solution.u[end]
    # problem = ODEProblem(init_relaxation, relaxed, t_span, (eos, tau_a, tau_u, tau_t, 0.0, 0.0, 1.0))
    # solution = solve(problem, TRBDF2())

    # solution = solve(problem, TRBDF2(), abstol=1e-3, reltol=1.0)
    # solution = solve(problem, TRBDF2(), abstol=1e-2, reltol=1.0)
    # solution = solve(problem, TRBDF2(), abstol=1e-1, reltol=1.0)
    # solution = solve(problem, TRBDF2(autodiff=AutoFiniteDiff()))
    # solution = solve(problem, Rosenbrock23(autodiff = AutoFiniteDiff()))
    # solution = solve(problem, Rosenbrock23(autodiff = AutoFiniteDiff()), abstol=1e-5, reltol=1e-2)
    # solution = solve(problem, Rosenbrock23(autodiff = AutoFiniteDiff()), abstol=1e-4, reltol=1e-1)
    # solution = solve(problem, Rosenbrock23(autodiff = AutoFiniteDiff()), abstol=1e-3, reltol=1.0)
    # solution = solve(problem, Rosenbrock23(autodiff = AutoFiniteDiff()), abstol=1e-2, reltol=1.0)
    # return solution

    if toggle_print
        println("After relaxation")
        vel, pres, temp = get_vars(eos, solution.u[end])
        print_tol(vel, "velocity")
        print_tol(pres, "pressure")
        print_tol(temp, "temperature")
        println(solution.stats)

        vars = [get_vars(eos, u) for u in solution.u]

        vel = [v[1] for v in vars]
        pres = [v[2] for v in vars]
        temp = [v[3] for v in vars]
        e_total = [v[4] for v in vars]
        e_kin = [v[5] for v in vars]
        e_int = [v[6] for v in vars]

        coords = ("X", "Y", "Z")
        for i in 1:3
            plot(
                solution.t, [v[1][i] for v in vel];
                title="Скорость по координате $(coords[i])",
                ylabel=L"$u_%$(lowercase(coords[i])), м/с$",
                xlabel=L"t, с",
                xlims=(t_span[1], t_span[2]),
                minorgrid=true, grid=true,
                legend=false
            )
            plot!(
                solution.t, [v[2][i] for v in vel]
            )
            savefig("relaxation/1$i.png")
        end

        p_num = 2
        titles = ("Давление", "Температура", "Полная энергия", "Кин. энергия", "Внутр. энергия")
        ylabels = (L"P, Па", L"\Theta, К", L"e_{total}, Дж", L"e_{kin}, Дж", L"e_{int}, Дж")
        for var in (pres, temp, e_total, e_kin, e_int)
            plot(
                solution.t, [v[1] for v in var];
                title=titles[p_num-1],
                ylabel=ylabels[p_num-1],
                xlabel=L"t, с",
                xlims=(t_span[1], t_span[2]),
                minorgrid=true, grid=true,
                legend=false
            )
            plot!(
                solution.t, [v[2] for v in var]
            )
            savefig("relaxation/$p_num.png")
            p_num += 1
        end
    end

    # return solution.u[end]
    return solution
end

function init_relaxation(Q::Array{<:Any,1}, params, t::Float64)
    eos, tau_a, tau_u, tau_t, enable_v, enable_p, enable_t = params
    S = similar(Q)
    Q = [Q[p:p+14] for p in 1:15:length(Q)]
    nph = length(Q)
    frac = [Q[p][1] for p in 1:nph]

    FQ = [reshape(Q[p][7:15] ./ frac[p], (3, 3)) for p in 1:nph]
    if (det(FQ[1]) / eos[1].rho0 < 0)
        print("Negative sqrt: ", det(FQ[1]) / eos[1].rho0)
        exit()
    end
    if (det(FQ[2]) / eos[2].rho0 < 0)
        print("Negative sqrt: ", det(FQ[2]) / eos[2].rho0)
        exit()
    end
    true_den = [sqrt(det(FQ[p]) / eos[p].rho0) for p in 1:nph]
    den = frac .* true_den

    vel = [Q[p][3:5] / den[p] for p in 1:nph]
    if enable_v < 0.5
        vel = [0.5 * (vel[1] + vel[2]) for _ in 1:2]
    end
    e_total = [Q[p][6] / den[p] for p in 1:nph]
    e_kin = [sum(vel[p] .^ 2) / 2 for p in 1:nph]
    e_int = e_total - e_kin
    def_grad = [Q[p][7:15] / den[p] for p in 1:nph]

    G = [finger(def_grad[p]) for p in 1:nph]
    ent = [entropy(eos[p], e_int[p], G[p]) for p in 1:nph]
    strs = [reshape(frac[p] .* stress(eos[p], ent[p], def_grad[p]), (3, 3)) for p in 1:nph]

    beta = zeros(2)
    temp = [derivative(S -> energy(eos[p], S, G[p]), ent[p]) for p in 1:nph]

    # WARNING: Возможно требуется деление на объемную долю
    pres = [-1 / 3 * tr(s) for s in strs]

    K = [1 / frac[p] .* strs[p] + beta[p] .* I for p in 1:nph]

    v = 1 / 2
    v = [v, 1 - v]
    w = v[1] .* vel[1] + v[2] .* vel[2]

    mu = 1 / 2
    mu = [mu, 1 - mu]
    pi = -1 / 3 * tr_(mu[1] .* K[1] + mu[2] .* K[2])

    temp_hat = sum(frac .* temp)
    Q_t = [temp_hat - temp[p] for p in 1:nph]
    a_1 = 1.0
    theta_t = 1.0 # Скорость релаксации по температуре
    k_t = (temp[2] * pres[1] - temp[1] * pres[2]) / (a_1 * temp[1] * temp[2] * Q_t[1])
    k_t = [k_t, -k_t * Q_t[2] / Q_t[1]]

    for p in 1:nph
        shift = (p - 1) * 15
        S[shift+1] = enable_p * 1 / 3 / tau_a * tr_(K[3-p] - K[p]) + enable_t * theta_t * Q_t[p] / k_t[p]
        S[shift+2] = 0
        S[shift+3:shift+5] = [enable_v * 1 / tau_u * (vel[3-p][i] - vel[p][i]) for i in 1:3]
        S[shift+6] = enable_v * 1 / tau_u * sum([w[k] * (vel[3-p][k] - vel[p][k]) for k in 1:3]) + enable_p * 1 / 3 / tau_a * pi * tr_(K[p] - K[3-p]) + 0.0 * 1 / tau_t * (temp[3-p] - temp[p]) + enable_t * theta_t * Q_t[p]
        S[shift+7:shift+15] = [enable_p * 1 / 9 / tau_a * true_den[p] * def_grad[p][i] * tr_(K[3-p] - K[p]) + enable_t * den[p] * def_grad[p][i] / 3 / frac[p] * theta_t * Q_t[p] / k_t[p] for i in 1:9]
    end
    # Sum syncronnically
    # println("Sum of righthand side is ", (sum([S[i] + S[i + 15] for i in 1:15])))
    # println("Sum of righthand side is ", [S[i] + S[i + 15] for i in 1:15])
    # TODO: Exception if not zero
    return S
    # TODO: control sum of S at start and at end
end

end # module Relaxation

# EOF
