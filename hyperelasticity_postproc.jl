using CSV
using DataFrames
using LinearAlgebra
using Plots

if !isdefined(Main, :SimpleLA)
  include("SimpleLA.jl")
end
if !isdefined(Main, :Strains)
  include("Strains.jl")
end
if !isdefined(Main, :EquationsOfState)
  include("EquationsOfState.jl")
  using .EquationsOfState
end

if !isdefined(Main, :Hyperelasticity)
  include("Hyperelasticity.jl")
  using .Hyperelasticity
end

theme(:default)
script_path = @__FILE__
script_dir = dirname(script_path)
println("Script path: $script_path")
println("Changing working directory to: $script_dir")
if isdir(script_dir)
  cd(script_dir)
else
  error("Directory does not exist: $script_dir")
end

const DATAPATH = "./barton_data/"
const PLOTPATH = "./plots/"
const DATAFILE = "result.csv"

den = Float64[]
vel = Matrix{Float64}(undef, 0, 3)
ent = Float64[]
strs = Vector{Matrix{Float64}}()

if isdefined(Main, :Q0)
  println("Q0 detected. Using in-program data.")
  global data = [cons2data(eos, q) for q in Q0]
else
  println("Reading data from file: $(DATAPATH * DATAFILE)")
  global data = CSV.read(DATAPATH * DATAFILE, DataFrame; delim='\t', header=3)
  println("Loaded $(size(data, 1)) rows and $(size(data, 2)) columns from CSV.")
end

den = data[:, 1]
vel = Matrix(data[:, 2:4])
ent = data[:, 5]
strs = [reshape(Vector(data[i, 6:14]), 3, 3) for i in 1:size(data, 1)]
temp = data[:, 15]

eos = Stiffened()
pres = [-1 / 3 * tr(strs[i]) for i in eachindex(strs)]
println("Calculating temperature...")
# temp2 = [300 * exp(ent[i] / (9.3e-4) - 2 * (2.78 / den[i])) for i in eachindex(ent)]
X = range(0, 1, length=length(ent))
println("Generated X grid with $(length(X)) points.")

const TITLES = ["Плотность", "Скорость X", "Скорость Y", "Скорость Z",
  "Энтропия", "Давление", "Температура"]
const YLABELS = ["ρ, г/см^3", "u_x, км/c", "u_y, км/c", "u_z, км/c",
  "η, кДж/(г K)", "P, Па", "Θ, K"]
const COLORS = [:blue, :black]

function plot_and_save(x, y, title_str, ylabel_str, filename, color)
  println("Plotting: $title_str -> $filename")
  p = plot(x, y;
    label="Фаза 1",
    color=color,
    grid=true,
    minorgrid=true,
    xticks=0:0.1:1,
    yticks=:auto,
    yformatter=:plain,
    size=(800, 600)
  )
  title!(p, title_str)
  ylabel!(p, ylabel_str)
  mkpath(PLOTPATH)
  savefig(p, PLOTPATH * filename)
  println("Saved lot to $(PLOTPATH * filename)")
end

plots = [
  (den, TITLES[1], YLABELS[1], "density.png"),
  [(vel[:, i], TITLES[i+1], YLABELS[i+1], "velocity_1.png") for i in 1:3]...,
  # (vel[:, 1], TITLES[2], YLABELS[2], "velocity_1.png"),
  # (vel[:, 2], TITLES[3], YLABELS[3], "velocity_2.png"),
  # (vel[:, 3], TITLES[4], YLABELS[4], "velocity_3.png"),
  (ent, TITLES[5], YLABELS[5], "entropy.png"),
  (pres, TITLES[6], YLABELS[6], "pressure.png"),
]
for (y, title, ylabel, fname) in plots
  plot_and_save(X, y, title, ylabel, fname, COLORS[1])
end
println("All plots generated successfully.")
