#
# Relaxation.jl
#

module Relaxation

using ForwardDiff: derivative
using DifferentialEquations: ODEProblem, solve, TRBDF2
using LinearAlgebra
using ..EquationsOfState: energy, entropy, stress, EoS
using ..Strains: finger
using ..SimpleLA

export relaxation

function relaxation(eos::Tuple{T,T}, initial::Array{<:Any,1}, dt) where {T <: EoS}
    t_span = (0, dt)

    # TODO: Calculate tau from dt
    # tau_a = 1e-6
    # tau_u = 1e-6
    # tau_t = 1e-6
    tau_a = dt * 50
    tau_u = dt / 10
    tau_t = dt * 50

    # tau_a = 1e10
    # tau_t = 1e10

    problem = ODEProblem(init_relaxation, initial, t_span, (eos, tau_a, tau_u, tau_t))
    solution = solve(problem, TRBDF2(), dtmax= min(tau_a, tau_u, tau_t) / 10)
    # return solution
    return solution.u[end]
end

function init_relaxation(Q::Array{<:Any,1}, params, t::Float64)
    eos, tau_a, tau_u, tau_t = params
    S = similar(Q)
    Q = [Q[p:p+14] for p in 1:15:length(Q)]
    nph = length(Q)
    frac = [Q[p][1] for p in 1:nph]

    FQ = [reshape(Q[p][7:15] ./ frac[p], (3, 3)) for p in 1:nph]
    if (det(FQ[1]) / eos[1].rho0 < 0)
        print("Negative sqrt: ", det(FQ[1]) / eos[1].rho0)
    end
    if (det(FQ[2]) / eos[2].rho0 < 0)
        print("Negative sqrt: ", det(FQ[2]) / eos[2].rho0)
    end
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

    beta = zeros(2)
    temp = [derivative(S -> energy(eos[p], S, G[p]), ent[p]) for p in 1:nph]

    K = [1 / frac[p] .* strs[p] + beta[p] .* I for p in 1:nph]

    v = 1/2
    v = [v, 1 - v]
    w = v[1] .* vel[1] + v[2] .* vel[2]

    mu = 1/2
    mu = [mu, 1 - mu]
    pi = - 1/3 * tr_(mu[1] .* K[1] + mu[2] .* K[2])

    for p in 1:nph
        shift = (p - 1) * 15
        S[shift + 1] = 1/3 / tau_a * tr_(K[3-p] - K[p])
        S[shift + 2] = 0
        S[shift + 3: shift + 5] = [1/tau_u* (vel[3-p][i] - vel[p][i]) for i in 1:3]
        S[shift + 6] = 1/tau_u * sum([w[k] * (vel[3-p][k] - vel[p][k]) for k in 1:3]) + 1/3 / tau_a * pi * tr_(K[p] - K[3-p]) + 1/tau_t * (temp[3-p] - temp[p])
        S[shift + 7: shift + 15] = [1/9 / tau_a * true_den[p] * def_grad[p][i] * tr_(K[3-p] - K[p]) for i in 1:9]
    end
    # Sum syncronnically
    # println("Sum of righthand side is ", (sum([S[i] + S[i + 15] for i in 1:15])))
    # TODO: Exception if not zero
    return S
    # TODO: control sum of S at start and at end
end

end # module Relaxation

# EOF
