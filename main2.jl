using Printf
# Logging in terminal and file
# https://julialogging.github.io/how-to/tee/#Send-messages-to-multiple-locations
using Logging, LoggingExtras
using BenchmarkTools

# Include all modules that is used in main.jl and in imported modules
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")
include("./Relaxation.jl")

# Только то, что нужно в main.jl
using .EquationsOfState: EoS, Barton2009, Stiffened
using .HyperelasticityMPh: initial_states, cons2prim_mph, prim2cons_mph, get_eigvals, cons2data_mph#, postproc_arrays
using .NumFluxes: lxf, hll
using .Relaxation: relaxation

# ##############################################################################
# ### Main driver ##############################################################
# ##############################################################################

# Set equation of state for each phase
eos = (Stiffened(rho0=2780, s=1.338, c0=5330, cv=9.3e2, mu=27.6e9, T0=300, G0=2.13, S0=0),
       Stiffened(rho0=8930, s=1.49,  c0=3970, cv=3.9e2, mu=45.0e9, T0=300, G0=2,    S0=0))

X = 0.1     # Coordinate boundary [m]
T = 2.5e-6  # Time boundary [1e-5 s]

nx = 1000   # Number of steps on dimension coordinate
cfl = 0.95  # Courant-Friedrichs-Levy number
dt = 5 * 1e-6

dx = X / nx # Coordinate step

# ##############################################################################

# Logging settings
log_filename = "comparison.log"
@info @sprintf("Starting log at: %s", log_filename)
logger = TeeLogger(
  global_logger(),          # Current global logger (stderr)
  FileLogger(log_filename)  # FileLogger writing to logfile.log
)
global_logger(logger) # Set logger as global logger


# Prepare the directory where data files will be saved
cd(@__DIR__)

# Initialize initial conditions

Q0 = Array{Float64}(undef, 30)
# nx = 5
# Q0 = [ 0.52, 1445.65, 2569.8, -9911.14, 0.0, 2.266e7, 1445.813, 1.0107, 0.0, 1.0338, 1445.65, 0.0, 0.0, 0.0, 1445.649, 0.48, 4286.24, 6504.6957, 31581.07, 0.0, 8.7, 4285.72, 3.2466, 0.0, 3.32, 4286.2417, 0.0, 0.0, 0.0, 4286.2432 ]
# nx = 1000
Q0 = [ 0.48, 1334.35, 3778.1, -8512.54, 0.0, 2.05e7, 1334.52, 1.01059, 0.0, 1.03, 1334.35, 0.0, 0.0, 0.0, 1334.35, 0.52, 4643.75, 3689.20, 36042.61, 0.0, 2.6, 4643.19, 3.24, 0.0, 3.31, 4643.75, 0.0, 0.0, 0.0, 4643.75]
@info "Initial array initialized"


# ##############################################################################


@info "Evaluating time step"
lambda = Array{Float64}(undef, 1)
eigvals = Array{Array{Float64}}(undef, 1)
n = [1, 0, 0]
eigvals[1] = get_eigvals(eos, Q0, n)
lambda[1] = maximum(abs.(eigvals[1]))
dt = cfl * dx / maximum(lambda)

@info @sprintf("Time step: dt = %.3e", dt)
@info "Starting measurements"

# @btime begin
Q1 = relaxation(eos, Q0, dt)
# end

@info @sprintf("Done!")

# EOF
