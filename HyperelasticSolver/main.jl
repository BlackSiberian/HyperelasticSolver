using LinearAlgebra
using Plots, LaTeXStrings
using Printf

# Logging in termical and file
# https://julialogging.github.io/how-to/tee/#Send-messages-to-multiple-locations
using Logging, LoggingExtras

# #############################################################################
# Если используется модуль (module), но не исползуется пакет (package), то включение
# модуля не тривиально. В частнсоти, нельзя сделать <<using Strains>>,
# но можно сделать <<using .Strains>>
#
# Вариант с <<using Strains>> требует явной коррекции пути вида
# push!(LOAD_PATH, pwd()), что в сообществе считается идейно плохим решением.
#
# Детали см. тут:
#     https://copyprogramming.com/howto/how-to-import-custom-module-in-julia
#     https://stackoverflow.com/questions/52519625/loading-a-module-from-the-local-directory-in-julia
#     https://discourse.julialang.org/t/how-to-load-a-module-from-the-current-directory/21727/3
#
# Этих проблем нет, если импортируетя пакет, а не просто модуль.
#
# Далее, есть неудобная особенность, когда по цепочке импортов в
# модулях импортируется один и тот же модуль. В этом случае имена типов и функций
# дополняются префиксом в виде имени модуля, и таким образом, неправильно происходит
# диспетчеризация.
#
# Детали и текущее решение см. тут:
#     https://discourse.julialang.org/t/referencing-the-same-module-from-multiple-files/77775
# ##############################################################################


# Делаем include на все модули, уоторые используются в main.jl
# и в импортируемых модулях.
include("./Strains.jl");
include("./EquationsOfState.jl")
# include("./Hyperelasticity.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl");

# Только то, что нужно в main.jl
using .EquationsOfState: Barton2009, Hank2016, EoS
# using .Hyperelasticity: prim2cons, cons2prim, initial_states, postproc_arrays
using .HyperelasticityMPh: initial_states, postproc_arrays
using .NumFluxes: lxf


"""
    Returns the value of quantity in the cell to the next time layer
    @param Q is the quantity in the left, central and right cells
    @param F is the numerical flux method
    @param lambda is the ratio of the spatial mesh stepsize to the temporal meshstep size
"""
function update_cell(Q::Array{<:Any,2}, flux_num::Function, lambda, eos::T) where {T <: EoS}
    Q_l, Q, Q_r = Q[:, 1], Q[:, 2], Q[:, 3]
    F_l = flux_num(eos, Q_l, Q, lambda)
    F_r = flux_num(eos, Q, Q_r, lambda)
    return Q - 1.0 / lambda * (F_r - F_l)
end



# ##############################################################################
# ### Main driver below
# ##############################################################################


# Выносим сюда, в одно место, постепенно, все основные параметры расчета.
# Потом завернуть в структуру?
# eos = Barton2009()              # EoS
# dname = "./barton_data/"        # directory name -- directory where data files are saved
eos = Hank2016()
dname = "./hank_data/"

# ---
# cd(@__DIR__)
# Подготовить директорию, где сохраняются файлы расчета.
# isdir(dir) || mkdir(dir)               # Проверить что путь существует,
#                                       # если нет --- то создать.
dname = mkpath(dname)                   # fname = joinpath(prefix,fname)

#foreach(rm, filter(endswith(".txt"), readdir(dir,join=true)))
foreach(rm, readdir(dname, join=true) ) # Удалить все файлы в директории
@printf("Cleaning data directory  %s\n", dname)

get_fname(nstep) = @sprintf("sol_%06i.dat", nstep) # Padded with zeros

"""
    Save solution array to file
    fname    : File name
    time     : Time value.
    Q        : Solution array [15] x [ncells]
"""
function save_data(fname, time, Q)
    io = open(fname, "w")
    write(io, "$t\n")
    nx = size(Q)[2]
    for i in 1:nx
        write(io, join(Q[:, i], "\t"), "\n")
    end
    close(io)
end

# ---
# Настройка логгирования
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
"""
    Задает НУ с известными состояниями для задачи Римана.
    Эта фунция ничего не знает про физику, но знает про сетку.    
"""
function initial_condition(Ql,Qr,nx)
    Q = Array{Float64}(undef, 30, nx)
    for i in 1:nx
        if (i-1) * dx < 0.5 * X
            Q[:, i] = Ql
        else
            Q[:, i] = Qr
        end
    end
    return Q
end

testcase = 2    # Select the test case
Ql, Qr = initial_states(eos, testcase)

log_freq = 10   # Log frequency


X = 1.0     # Coordinate boundary [m]
T = 0.06    # Time boundary [1e-5 s]

nx = 500    # Number of steps on dimension coordinate
cfl = 0.6   # Courant-Friedrichs-Levy number

dx = X / nx # Coordinate step


# Initialize initial conditions
Q0 = initial_condition(Ql,Qr,nx)


t = 0.0     # Initilization of time
nstep = 0   # Initilization of timestep counter

fname = joinpath(dname, get_fname(nstep))
save_data(fname, t, Q0)

# ##############################################################################
# Timestepping
while t < T
    # Computing the time step using CFL condition
    lambda = 0
    for i in 1:nx
        Q = Q0[:, i]
        # local den = density(Q[4:12])
        # lambda = max(abs(Q0[1, i]) / den + 5.0, lambda) # TODO: replace 5 with max eigenvalue
        lambda = 5.0
    end
    
    global dt = cfl * dx / lambda    
    global t += dt                  # Updating the time
    global nstep += 1               # Updating the step counter
    
    # Computing the next time layer
    Q1 = similar(Q0)                
    Q1[:, begin] = Q0[:, begin]
    Q1[:, end] = Q0[:, end]
    for i in 2:nx-1
        Q1[:, i] = update_cell(Q0[:, i-1:i+1], lxf, dx/dt, eos)
    end

    global Q0 = copy(Q1)

    if nstep % log_freq == 0
        # Saving the solution array to a file
        
        global fname = joinpath(dname, get_fname(nstep))
        save_data(fname, t, Q0)

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
# Plotting #####################################################################
# ##############################################################################

#
# Этот кусок ниже знает про сетку и про физику (по значениям переменных).
# Для разных моделей он разный.
# Проще всего вулючать сюда какой-нибьудь файл, который специфичен для модели,
# видит весь контекст, не ограничивает область видимости. Пользователь
# делает в нем все, что хочет, под свою ответственность.
#
# Можно как то параметризовать модели (в каждом "физическом" модуле определить
# значение переменной model или тит или еще что, и сделать соотвтествующий
# switch/if/...) --- но смысла нет пока особо.
#
# Поэтому кусок ниже засовываем в файл с говорящим названием
# и включаем его "как есть".
# Стандартное название фиксируем ---
#     [название модуля с физикой маленькми буквами]_postproc.jl.
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