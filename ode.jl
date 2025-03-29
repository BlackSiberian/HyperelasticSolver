# Include all modules that is used in main.jl and in imported modules
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
# include("./Hyperelasticity.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")

# Только то, что нужно в main.jl
using .EquationsOfState
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh
using .NumFluxes
using .Strains
using .SimpleLA

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff: derivative


using DifferentialEquations

# Table of results of different solers
# Name | Work time | Steps to Δt
# Δt must be greater than relax time

using Plots
gr()

#Setup
eos = (Barton2009(), Barton2009())
Q0, _ = initial_states(eos, 7)
Q0[3] *= 1.1
Q0[18] *= 0.9
dt = 1e-5
tspan = (0, dt)

xi = 1e6
chi = 1e6
phi = 1e6

relaxation(Q0, 0.0, 0.0)

#Define the problem

function relaxation(Q::Array{<:Any,1}, p, t::Float64)
    S = similar(Q)
    Q = [Q[p:p+14] for p in 1:15:length(Q)]
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
        S[shift + 1] = 1/3 * xi * tr_(K[3-p] - K[p])
        S[shift + 2] = 0
        S[shift + 3: shift + 5] = [chi * (vel[3-p][i] - vel[p][i]) for i in 1:3]
        S[shift + 6] = chi * sum([w[k] * (vel[3-p][k] - vel[p][k]) for k in 1:3]) + 1/3 * xi * pi * tr_(K[p] - K[3-p]) + phi * (temp[3-p] - temp[p])
        S[shift + 7:shift + 15] = [1/9 * xi * true_den[p] * def_grad[p][i] * tr_(K[3-p] - K[p]) for i in 1:9]
    end
    # Sum syncronnically
    println("Sum of righthand side is ", (sum([S[i] + S[i + 15] for i in 1:15])))
    # TODO: Exception if not zero
    return S
end

# TODO: min tau / 10
#Pass to solver
problem = ODEProblem(relaxation, Q0, tspan)
solution = solve(problem, TRBDF2(), dtmax=1 / chi / 10)

display(solution)
#Plot
# plot(sol, linewidth = 2, title = "Carbon-14 half-life",
#     xaxis = "Time in thousands of years", yaxis = "Percentage left",
#     label = "Numerical Solution")
# plot!(sol.t, t -> exp(-C₁ * t), lw = 3, ls = :dash, label = "Analytical Solution")

# plot(solution)

plot(solution.t, [u[3] for u in solution.u] ./ [u[2] for u in solution.u], title="uₓ", label="Phase 1")
plot!(solution.t, [u[18] for u in solution.u] ./ [u[17] for u in solution.u], label="Phase 2")

# TODO: control sum of S at start and at end

# TODO: graph vel, stresses
# TODO: graph norma of vel[1] and vel[2]
# TODO: graph of temps or enthropy
# TODO: graph of stress of pressure and deviator (shear stress (Von-Mises effective yield criterion))
# TODO: nice graphs for articles
# TODO: describe 