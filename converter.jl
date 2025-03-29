include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")

using .EquationsOfState: EoS, Barton2009
using .HyperelasticityMPh: initial_states, cons2prim_mph, prim2cons_mph, cons2data_mph, get_eigvals, flux_mph, noncons_flux
using .NumFluxes: lxf, hll

function save_data_plt(fname::String, Q::Array{<:Any,2})
  io = open(fname, "w")
  nx = size(Q)[2]
  write(io, "a1\tr1\tu11\tu21\tu31\tS1\tT111\tT211\tT311\tT121\tT221\tT321\tT131\tT231\tT331\ta2\tr2\tu12\tu22\tu32\tS2\tT112\tT212\tT312\tT122\tT222\tT322\tT132\tT232\tT332", "\n")
  for i in 1:nx
    D = cons2data_mph(eos, Q[:, i])
    write(io, join(D, "\t"), "\n")
  end
  close(io)
end

"""
    read_data(fname::String)

Read the solution array from a `fname` file.
"""
function read_data(fname::String)
  data = readlines(fname)
  nx = length(data) - 1
  P = Array{Float64}(undef, 30, nx)
  for (indx, line) in enumerate(data[2:end])
      P[:, indx] = parse.(Float64, split(line))
  end
  return P, nx
end

cd(@__DIR__)
dir_name = "plot_data/"

eos = (Barton2009(), Barton2009())
choosen_file = joinpath(dir_name, "advanced1800.csv")
println("choosen_file = $choosen_file")

# P0 = Array{Float64}(undef, 30, nx)
P0, nx = read_data(choosen_file) # Read the last file
global Q0 = similar(P0)
for i in 1:nx
    Q0[:, i] = prim2cons_mph(eos, P0[:, i])
end

save_data_plt(joinpath(dir_name, "advanced1800conv.csv"), Q0)