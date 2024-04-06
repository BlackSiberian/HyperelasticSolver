using LinearAlgebra
using Plots, LaTeXStrings
using Printf
using ForwardDiff
# Logging in terminal and file
# https://julialogging.github.io/how-to/tee/#Send-messages-to-multiple-locations
using Logging, LoggingExtras

# Include all modules that is used in main.jl and in imported modules
include("./Strains.jl");
include("./EquationsOfState.jl")
# include("./Hyperelasticity.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")

# Только то, что нужно в main.jl
using .EquationsOfState: Barton2009, Hank2016, EoS
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh: initial_states, cons2prim_mph#, postproc_arrays
using .NumFluxes: lxf


"""
    update_cell(Q::Array{<:Any,2}, flux_num::Function, lambda, eos::T) where {T <: EoS}

Returns the value of quantity in the cell to the next time layer using
`flux_num` numerical flux method and `eos` equation of state

`lambda` is the value of `Δx/Δt`
"""
function update_cell(Q::Array{<:Any,2}, flux_num::Function, lambda, eos::T) where {T<:EoS}
  Q_l, Q, Q_r = Q[:, 1], Q[:, 2], Q[:, 3]

  # F_l = flux_num(eos, Q_l, Q, lambda)
  # F_r = flux_num(eos, Q, Q_r, lambda)
  # return Q - 1.0 / lambda * (F_r - F_l)

  F_l, NF_l, _ = flux_num(eos, Q_l, Q, lambda)
  F_r, _, NF_r = flux_num(eos, Q, Q_r, lambda)
  return Q - 1.0 / lambda * ((F_r - F_l) + (NF_r + NF_l))
end

"""
    save_data(fname, Q::Array{<:Any,2})

Save the solution array to a `fname` file at `time`.
"""
function save_data(fname, Q::Array{<:Any,2})
  io = open(fname, "w")
  # write(io, "$t\n")
  nx = size(Q)[2]
  write(io, "a1\tr1\tu11\tu21\tu31\tS1\tF111\tF211\tF311\tF121\tF221\tF321\tF131\tF231\tF331\ta2\tr2\tu12\tu22\tu32\tS2\tF112\tF212\tF312\tF122\tF222\tF322\tF132\tF232\tF332", "\n")
  for i in 1:nx
    # println(Q[:, i])
    P = cons2prim_mph(eos, Q[:, i])
    write(io, join(P, "\t"), "\n")
    # write(io, join(Q[:, i], "\t"), "\n")
  end
  close(io)
end


"""
    initial_condition(Ql, Qr, nx)

Sets the initial condition with two states for the Riemann problem with `nx` cells.
"""
function initial_condition(Ql, Qr, nx)
  # Эта функция ничего не знает про физику, но знает про сетку.   
  Q = Array{Float64}(undef, 30, nx)
  for i in 1:nx
    Q[:, i] = (i - 1) < nx / 2 ? Ql : Qr
  end
  return Q
end


# ##############################################################################
# ### Main driver ##############################################################
# ##############################################################################


# Выносим сюда, в одно место, постепенно, все основные параметры расчета.
# Потом завернуть в структуру?
eos = Barton2009()              # Equation of state
dname = "./barton_data/"        # directory name -- directory where data files are saved
# eos = Hank2016()
# dname = "./hank_data/"

# cd(@__DIR__)

# Prepare the directory where the data files will be saved
dname = mkpath(dname)

foreach(rm, readdir(dname, join=true)) # Remove all files in the directory
@printf("Cleaning data directory  %s\n", dname)

get_fname(nstep) = @sprintf("sol_%06i.csv", nstep) # Padded with zeros


# ---
# Logging settings
fname_log = "solution.log"
logger = TeeLogger(
  global_logger(),           # Current global logger (stderr)
  FileLogger(fname_log)      # FileLogger writing to logfile.log
)

with_logger(logger) do
  @info @sprintf("Starting log at: %s", fname_log)
  @info @sprintf("Data directory: %s", dname)
end

# ##############################################################################

testcase = 5    # Select the test case
Ql, Qr = initial_states(eos, testcase)

log_freq = 10   # Log frequency


X = 1.0     # Coordinate boundary [m]
T = 0.06    # Time boundary [1e-5 s]

nx = 500    # Number of steps on dimension coordinate
cfl = 0.6   # Courant-Friedrichs-Levy number

dx = X / nx # Coordinate step


# Initialize initial conditions
Q0 = initial_condition(Ql, Qr, nx)


t = 0.0     # Initilization of time
nstep = 0   # Initilization of timestep counter

fname = joinpath(dname, get_fname(nstep))
save_data(fname, Q0)


# ##############################################################################
# Timestepping
while t < T
  # Computing the time step using CFL condition
  lambda = 0
  for i in 1:nx
    Q = Q0[:, i]
    # local den = density(Q[4:12])
    # lambda = max(abs(Q0[1, i]) / den + 5.0, lambda) # TODO: replace 5 with max eigenvalue
    # lambda = max(abs(Q[3]) / Q[2] + 5.0, abs(Q[18]) / Q[17] + 5.0, lambda)
    lambda = max(lambda, (abs.(Q[3:15:end] ./ Q[2:15:end]) .+ 5.0)...)
  end

  global dt = cfl * dx / lambda
  global t += dt                  # Updating the time
  global nstep += 1               # Updating the step counter

  # Computing the next time layer
  Q1 = similar(Q0)
  Q1[:, begin] = Q0[:, begin]
  Q1[:, end] = Q0[:, end]
  for i in 2:nx-1
    Q1[:, i] = update_cell(Q0[:, i-1:i+1], lxf, dx / dt, eos)
  end

  global Q0 = copy(Q1)

  if nstep % log_freq == 0
    # Saving the solution array to a file

    global fname = joinpath(dname, get_fname(nstep))
    save_data(fname, Q0)

    # Logging
    msg = @sprintf("Solution saved to: %s\n", fname)
    with_logger(logger) do
      @info msg
    end
  end

  # Logging
  msg = @sprintf("nstep = %d,\t t = %.6f / %.6f,\t dt = %.6f\n", nstep, t, T, dt)
  with_logger(logger) do
    @info msg
  end

end  # while t < T


# ##############################################################################
# ### Plotting #################################################################
# ##############################################################################

#
# Этот кусок ниже знает про сетку и про физику (по значениям переменных).
# Для разных моделей он разный.
# Проще всего включать сюда какой-нибудь файл, который специфичен для модели,
# видит весь контекст, не ограничивает область видимости. Пользователь
# делает в нем все, что хочет, под свою ответственность.
#
# Можно как то параметризовать модели (в каждом "физическом" модуле определить
# значение переменной model или тит или еще что, и сделать соответствующий
# switch/if/...) --- но смысла нет пока особо.
#
# Поэтому кусок ниже засовываем в файл с говорящим названием
# и включаем его "как есть".
# Стандартное название фиксируем ---
#     [название модуля с физикой маленькими буквами]_postproc.jl.
#
# Сейчас это:
#     hyperelasticity_postproc.jl
#     hyperelasticitymph_postproc.jl
#    

# include("hyperelasticity_postproc.jl")    
include("hyperelasticitymph_postproc.jl")

with_logger(logger) do
  @info @sprintf("Done!")
end

# EOF
