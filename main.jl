using Printf
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
using .EquationsOfState: EoS, Barton2009
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh: initial_states, cons2prim_mph, prim2cons_mph#, postproc_arrays
using .NumFluxes: lxf


"""
    update_cell(Q::Array{<:Any,2}, flux_num::Function, lambda, eos::T) where {T <: EoS}

Returns the value of quantity in the cell to the next time layer using
`flux_num` numerical flux method and `eos` equation of state

`lambda` is the value of `Δx/Δt`
"""
function update_cell(Q::Array{<:Any,2}, flux_num::Function, lambda, eos::Tuple{T,T}) where {T<:EoS}
  Q_l, Q, Q_r = Q[:, 1], Q[:, 2], Q[:, 3]

  # F_l = flux_num(eos, Q_l, Q, lambda)
  # F_r = flux_num(eos, Q, Q_r, lambda)
  # return Q - 1.0 / lambda * (F_r - F_l)

  F_l, _, NF_l = flux_num(eos, Q_l, Q, lambda)
  F_r, NF_r, _ = flux_num(eos, Q, Q_r, lambda)
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
    P = cons2prim_mph(eos, Q[:, i])
    write(io, join(P, "\t"), "\n")
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

get_filename(step_num) = @sprintf("sol_%06i.csv", step_num) # Padded with zeros

function initial_condition_tanh(eos, Ql, Qr, nx, eps)
  Q = Array{Float64}(undef, 30, nx)
  Pl = cons2prim_mph(eos, Ql)
  Pr = cons2prim_mph(eos, Qr)
  for i in 1:nx
    x = (i - 0.5) / nx
    P = x < 0.5 ? Pl : Pr
    P[1] = 0.2 / 2 * (tanh(4 * (x - 0.5) / eps) + 1) + 0.4
    P[16] = 1 - P[1]

    Q[:, i] = prim2cons_mph(eos, P)
  end
  return Q
end

# ##############################################################################
# ### Main driver ##############################################################
# ##############################################################################


# Prepare the directory where data files will be saved
cd(@__DIR__)
dir_name = "barton_data/"
dir_name = mkpath(dir_name)
foreach(rm, readdir(dir_name, join=true)) # Remove all files in the directory
@info @sprintf("Cleaning data directory  %s\n", dir_name)

# Logging settings
log_filename = "solution.log"
@info @sprintf("Starting log at: %s", log_filename)
logger = TeeLogger(
  global_logger(),          # Current global logger (stderr)
  FileLogger(log_filename)  # FileLogger writing to logfile.log
)
global_logger(logger) # Set logger as global logger

@info @sprintf("Data directory: %s", dir_name)

# ##############################################################################

# Выносим сюда, в одно место, постепенно, все основные параметры расчета.
# Потом завернуть в структуру?
# Set equation of state for each phase
eos = (Barton2009(), Barton2009(_rho0=8.93, _c0=6.22, _cv=9.0e-4, _t0=300, _b0=3.16, _alpha=1, _beta=3.577, _gamma=2.088))
testcase = 6    # Select the test case

log_freq = 10   # Log frequency


X = 1.0     # Coordinate boundary [m]
T = 0.06    # Time boundary [1e-5 s]

nx = 500    # Number of steps on dimension coordinate
cfl = 0.6   # Courant-Friedrichs-Levy number

dx = X / nx # Coordinate step

eps = 0.2

# Initialize initial conditions
Ql, Qr = initial_states(eos, testcase)
Q0 = initial_condition(Ql, Qr, nx)
# Q0 = initial_condition_tanh(eos, Ql, Qr, nx, eps)

t = 0.0        # Initilization of time
step_num = 0   # Initilization of timestep counter

fname = joinpath(dir_name, get_filename(step_num))
save_data(fname, Q0)
@info @sprintf("Initial state saved to: %s\n", fname)

# ##############################################################################
# Timestepping
while t < T
  # Computing the time step using CFL condition
  lambda = 0
  for i in 1:nx
    Q = Q0[:, i]
    # local den = density(Q[4:12])
    # lambda = max(abs(Q0[1, i]) / den + 5.0, lambda)
    # TODO: replace 5 with max eigenvalue
    lambda = max(lambda, (abs.(Q[3:15:end] ./ Q[2:15:end]) .+ 5.0)...)
  end

  global dt = cfl * dx / lambda
  global t += dt                  # Updating the time
  global step_num += 1            # Updating the step counter

  # Computing the next time layer
  Q1 = similar(Q0)
  Q1[:, begin] = Q0[:, begin]
  Q1[:, end] = Q0[:, end]
  Threads.@threads for i in 2:nx-1
    Q1[:, i] = update_cell(Q0[:, i-1:i+1], lxf, dx / dt, eos)
  end

  global Q0 = copy(Q1)

  # Saving the solution array to a file
  msg = @sprintf("Step = %d,\t t = %.6f / %.6f,\t Δt = %.6f\n",
    step_num, t, T, dt)

  if step_num % log_freq == 0
    global fname = joinpath(dir_name, get_filename(step_num))
    save_data(fname, Q0)
    msg *= @sprintf("Solution saved to: %s\n", fname)
  end

  @info msg

end  # while t < T

fname = joinpath(dir_name, "result.csv")
save_data(fname, Q0)
@info @sprintf("Result solution saved to: %s\n", fname)

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

# include("hyperelasticitymph_postproc.jl")

@info @sprintf("Done!")

# EOF
