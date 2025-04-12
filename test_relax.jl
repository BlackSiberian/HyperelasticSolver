# Include all modules that is used in main.jl and in imported modules
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
# include("./Hyperelasticity.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")
include("./Relaxation.jl")

# Только то, что нужно в main.jl
using .EquationsOfState: entropy, EoS, Barton2009, stress, energy
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh
using .NumFluxes
using .Strains: finger
using .SimpleLA
using .Relaxation

using FastGaussQuadrature
using LinearAlgebra
using ForwardDiff: derivative

# Table of results of different solers
# Name | Work time | Steps to Δt
# Δt must be greater than relax time

using Plots
gr()

#Setup
eos = (Barton2009(), Barton2009())
Ql, Qr = initial_states(eos, 7)
Q0 = Array{Float64}(undef, 30)
Q0[1:15] = Ql[1:15]
Q0[16:30] = Qr[1:15]
dt = 1e-5

sol = relaxation(eos, Q0, dt)
# display(sol)


# plot(sol.t, [u[3] for u in sol.u] ./ [u[2] for u in sol.u], title="uₓ", label="Phase 1")
# plot!(sol.t, [u[18] for u in sol.u] ./ [u[17] for u in sol.u], label="Phase 2")


Q = [[Q[p:p+14] for p in 1:15:length(Q)] for Q in sol.u]
nt = length(sol.u)
nph = length(Q[1])

frac = [[Q[t][p][1] for p in 1:nph] for t in 1:nt]

FQ = [[reshape(Q[t][p][7:15] ./ frac[t][p], (3, 3)) for p in 1:nph] for t in 1:nt]
true_den = [[sqrt(det(FQ[t][p]) / eos[p].rho0) for p in 1:nph] for t in 1:nt]
den = [frac[t] .* true_den[t] for t in 1:nt]

vel = [[Q[t][p][3:5] / den[t][p] for p in 1:nph] for t in 1:nt]
e_total = [[Q[t][p][6] / den[t][p] for p in 1:nph] for t in 1:nt]
e_kin = [[sum(vel[t][p] .^ 2) / 2 for p in 1:nph] for t in 1:nt]
e_int = [e_total[t] - e_kin[t] for t in 1:nt]
def_grad = [[Q[t][p][7:15] / den[t][p] for p in 1:nph] for t in 1:nt]

G = [[finger(def_grad[t][p]) for p in 1:nph] for t in 1:nt]
ent = [[entropy(eos[p], e_int[t][p], G[t][p]) for p in 1:nph] for t in 1:nt]
strs = [[reshape(frac[t][p] .* stress(eos[p], ent[t][p], def_grad[t][p]), (3, 3)) for p in 1:nph] for t in 1:nt]

temp = [[derivative(S -> energy(eos[p], S, G[t][p]), ent[t][p]) for p in 1:nph] for t in 1:nt]
println(temp[1][1], "\t", temp[1][2])
println(temp[end][1], "\t", temp[end][2])

pres = [[-1 / 3 / frac[t][p] * tr(strs[t][p]) for p in 1:nph] for t in 1:nt]

vonMises = [[sqrt(
  ((strs[t][p][1][1] - strs[t][p][2][2])^2
   + (strs[t][p][2][2] - strs[t][p][3][3])^2
   + (strs[t][p][3][3] - strs[t][p][1][1])^2
   + 6 * (strs[t][p][1][2]^2 + strs[t][p][2][3]^2 + strs[t][p][3][1]^2))
  /
  2) for p in 1:nph] for t in 1:nt]

for i in 1:3
  plot(sol.t, [_vel[1][i] for _vel in vel[:]], title="u_$i", label="Phase 1")
  plot!(sol.t, [_vel[2][i] for _vel in vel[:]], label="Phase 2")
  savefig("Velocity_$i.png")
end

for i in 1:3
  for j in 1:i
    plot(sol.t, [_strs[1][i][j] for _strs in strs[:]], title="stress_$i$j", label="Phase 1")
    plot!(sol.t, [_strs[2][i][j] for _strs in strs[:]], label="Phase 2")
    savefig("Stress_$i$j")
  end
end

plot(sol.t, [_temp[1] for _temp in temp[:]], title="temperature", label="Phase 1")
plot!(sol.t, [_temp[2] for _temp in temp[:]], label="Phase 2")
savefig("Temperature.png")

plot(sol.t, [_pres[1] for _pres in pres[:]], title="pressure", label="Phase 1")
plot!(sol.t, [_pres[2] for _pres in pres[:]], label="Phase 2")
savefig("Pressure.png")

plot(sol.t, [_true_den[1] for _true_den in true_den[:]], title="true density", label="Phase 1")
plot!(sol.t, [_true_den[2] for _true_den in true_den[:]], label="Phase 2")
savefig("True_density.png")

plot(sol.t, [_vonMises[:][1] for _vonMises in vonMises[:]], title="Von-Mises", label="Phase 1")
plot!(sol.t, [_vonMises[:][2] for _vonMises in vonMises[:]], label="Phase 2")

plot(sol.t, [_frac[1] for _frac in frac[:]], title="Fraction", label="Phase 1")
plot!(sol.t, [_frac[2] for _frac in frac[:]], label="Phase 2")
savefig("fracture.png")

# TODO: graph of stress of pressure and deviator (shear stress (Von-Mises effective yield criterion))
# TODO: nice graphs for articles
# TODO: describe
