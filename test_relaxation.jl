# Подключаем все модули, как в test_relaxation_inst_full.jl
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")
include("./Relaxation.jl") # Подключаем наш модуль

# Импортируем используемые функции
using .EquationsOfState: entropy, EoS, Barton2009, stress, energy, pressure
using .HyperelasticityMPh: prim2cons_mph, cons2prim_mph, initial_states
using .NumFluxes
using .Strains: finger
using .SimpleLA: tr_
using .Relaxation: relaxation

using LinearAlgebra # Для dot, tr, I
using ForwardDiff: derivative # Для вычисления температуры
using Plots
pyplot()  # Используем matplotlib backend - лучше для публикационных графиков с настройкой шрифтов
Plots.default(show = false)  # Отключаем интерактивный вывод

# Настройка шрифтов с засечками для публикационных графиков
Plots.default(
    fontfamily = "serif",
    titlefont = ("serif", 14),
    guidefont = ("serif", 12),    # Подписи осей
    tickfont = ("serif", 10),     # Цифры на осях
    legendfont = ("serif", 10)
)

# --- Подготовка данных ---
println("=== Подготовка данных ===")

# Создаём EoS
eos = (Barton2009(), Barton2009())

# Создаём начальное состояние Q0 (длины 30)
Ql, Qr = initial_states(eos, 73)
Q0 = Array{Float64}(undef, 30)
Q0[1:15] = Ql[1:15]
Q0[16:30] = Qr[1:15]
dt = 1e-5

println(cons2prim_mph(eos, Q0))

println("Входной вектор Q0 (до релаксации):")
println(Q0)

# Преобразуем в примитивные переменные для анализа
P0 = cons2prim_mph(eos, Q0)
P0_parts = [P0[i:i+14] for i in 1:15:length(P0)]

frac0 = [P0_parts[i][1] for i in 1:2]
true_den0 = [P0_parts[i][2] for i in 1:2]
den0 = frac0 .* true_den0
vel0 = [P0_parts[i][3:5] for i in 1:2]
ent0 = [P0_parts[i][6] for i in 1:2]
def_grad0 = [P0_parts[i][7:15] for i in 1:2]
G0 = [finger(def_grad0[i]) for i in 1:2]
e_int0 = [energy(eos[i], ent0[i], G0[i]) for i in 1:2]
strs0 = [reshape(stress(eos[i], ent0[i], def_grad0[i]), (3, 3)) for i in 1:2]
pres0 = [- 1/3 * tr_(strs0[i]) for i in 1:2]

println("\n--- Анализ до релаксации ---")
println("Фракции: ", frac0)
println("Плотности (rho): ", den0)
println("Скорости: ", vel0)
println("Давления: ", pres0)
println("Энергии: ", e_int0)
println("Температуры (через derivative): ", [derivative(S -> energy(eos[i], S, G0[i]), ent0[i]) for i in 1:2])

# --- Выполнение релаксации ---
println("\n=== Выполнение релаксации ===")

# Вызов relaxation
solution = relaxation(eos, Q0, dt)

println("Релаксация завершена.")
println("Количество шагов: ", length(solution.t))
println("Временной интервал: ", solution.t[1], " - ", solution.t[end])

# Получаем расслабленное состояние
Q_relaxed = solution.u[end]

println("\nВыходной вектор Q_relaxed (после релаксации):")
println(Q_relaxed)

# Преобразуем результат в примитивные переменные для анализа
P_relaxed = cons2prim_mph(eos, Q_relaxed)
P_relaxed_parts = [P_relaxed[i:i+14] for i in 1:15:length(P_relaxed)]

frac_relaxed = [P_relaxed_parts[i][1] for i in 1:2]
true_den_relaxed = [P_relaxed_parts[i][2] for i in 1:2]
den_relaxed = frac_relaxed .* true_den_relaxed
vel_relaxed = [P_relaxed_parts[i][3:5] for i in 1:2]
ent_relaxed = [P_relaxed_parts[i][6] for i in 1:2]
def_grad_relaxed = [P_relaxed_parts[i][7:15] for i in 1:2]
G_relaxed = [finger(def_grad_relaxed[i]) for i in 1:2]
e_int_relaxed = [energy(eos[i], ent_relaxed[i], G_relaxed[i]) for i in 1:2]
strs_relaxed = [reshape(stress(eos[i], ent_relaxed[i], def_grad_relaxed[i]), (3, 3)) for i in 1:2]
pres_relaxed = [- 1/3 * tr_(strs_relaxed[i]) for i in 1:2]
temp_relaxed = [derivative(S -> energy(eos[i], S, G_relaxed[i]), ent_relaxed[i]) for i in 1:2]

println("\n--- Анализ после релаксации ---")
println("Фракции: ", frac_relaxed)
println("Плотности (rho): ", den_relaxed)
println("Скорости: ", vel_relaxed)
println("Энергии: ", e_int_relaxed)
println("Энтропии: ", ent_relaxed)
println("Давления: ", pres_relaxed)
println("Температуры: ", temp_relaxed)
println("Градиент деформации 1: ", def_grad_relaxed[1])
println("Градиент деформации 2: ", def_grad_relaxed[2])

# --- Проверка релаксации ---
println("\n=== Проверка релаксации ===")

println("Разница скоростей до релаксации: ", vel0[1] - vel0[2])
println("Разница скоростей после релаксации: ", vel_relaxed[1] - vel_relaxed[2])

println("Разница давлений до релаксации: ", pres0[1] - pres0[2])
println("Разница давлений после релаксации: ", pres_relaxed[1] - pres_relaxed[2])

println("Разница температур до релаксации: ", [derivative(S -> energy(eos[i], S, G0[i]), ent0[i]) for i in 1:2][1] - [derivative(S -> energy(eos[i], S, G0[i]), ent0[i]) for i in 1:2][2])
println("Разница температур после релаксации: ", temp_relaxed[1] - temp_relaxed[2])

println("Разница энтропий до релаксации: ", ent0[1] - ent0[2])
println("Разница энтропий после релаксации: ", ent_relaxed[1] - ent_relaxed[2])

println("\nТест завершён.")

# --- Построение графиков схождения ---
println("\n=== Построение графиков схождения ===")

# Извлекаем историю из solution
times = solution.t
states = solution.u
n_steps = length(times)

# Массивы для хранения истории значений
vel_x = zeros(n_steps, 2)  # [time, phase]
vel_y = zeros(n_steps, 2)
vel_z = zeros(n_steps, 2)
pres = zeros(n_steps, 2)
temp = zeros(n_steps, 2)

# Вычисляем значения для каждого временного шага
for (i, Q) in enumerate(states)
    P = cons2prim_mph(eos, Q)
    P_parts = [P[j:j+14] for j in 1:15:length(P)]

    # Скорости по компонентам
    vel_x[i, :] = [P_parts[1][3], P_parts[2][3]]
    vel_y[i, :] = [P_parts[1][4], P_parts[2][4]]
    vel_z[i, :] = [P_parts[1][5], P_parts[2][5]]

    # Давления
    def_grad = [P_parts[p][7:15] for p in 1:2]
    ent = [P_parts[p][6] for p in 1:2]
    G = [finger(def_grad[p]) for p in 1:2]
    strs = [reshape(stress(eos[p], ent[p], def_grad[p]), (3, 3)) for p in 1:2]
    pres[i, :] = [-1/3 * tr_(strs[p]) for p in 1:2]

    # Температуры
    temp[i, :] = [derivative(S -> energy(eos[p], S, G[p]), ent[p]) for p in 1:2]
end

# Создаем директорию для графиков
plot_dir = "relaxation_plots"
mkpath(plot_dir)

# Настройки графиков с высоким DPI и minorticks
plot_settings = (dpi=300, linewidth=2, legend=:topright, grid=true, minorgrid=true, minorticks=5, size=(800, 600))

# График схождения скорости X
plot(times, vel_x[:, 1], label="Фаза 1"; plot_settings...)
plot!(times, vel_x[:, 2], label="Фаза 2", linestyle=:dash)
plot!(title="Схождение скорости X", xlabel="Время", ylabel="Скорость X")
savefig(joinpath(plot_dir, "velocity_x_convergence.png"))
closeall()
println("Сохранен график: velocity_x_convergence.png")

# График схождения скорости Y
plot(times, vel_y[:, 1], label="Фаза 1"; plot_settings...)
plot!(times, vel_y[:, 2], label="Фаза 2", linestyle=:dash)
plot!(title="Схождение скорости Y", xlabel="Время", ylabel="Скорость Y")
savefig(joinpath(plot_dir, "velocity_y_convergence.png"))
closeall()
println("Сохранен график: velocity_y_convergence.png")

# График схождения скорости Z
plot(times, vel_z[:, 1], label="Фаза 1"; plot_settings...)
plot!(times, vel_z[:, 2], label="Фаза 2", linestyle=:dash)
plot!(title="Схождение скорости Z", xlabel="Время", ylabel="Скорость Z")
savefig(joinpath(plot_dir, "velocity_z_convergence.png"))
closeall()
println("Сохранен график: velocity_z_convergence.png")

# График схождения давлений
plot(times, pres[:, 1], label="Фаза 1"; plot_settings...)
plot!(times, pres[:, 2], label="Фаза 2", linestyle=:dash)
plot!(title="Схождение давлений", xlabel="Время", ylabel="Давление")
savefig(joinpath(plot_dir, "pressure_convergence.png"))
closeall()
println("Сохранен график: pressure_convergence.png")

# График схождения температур
plot(times, temp[:, 1], label="Фаза 1"; plot_settings...)
plot!(times, temp[:, 2], label="Фаза 2", linestyle=:dash)
plot!(title="Схождение температур", xlabel="Время", ylabel="Температура")
savefig(joinpath(plot_dir, "temperature_convergence.png"))
closeall()
println("Сохранен график: temperature_convergence.png")

println("\nВсе графики сохранены в директорию: $plot_dir")
