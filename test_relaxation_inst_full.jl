# Подключаем все модули, как в test_relax.jl
include("./SimpleLA.jl")
include("./Strains.jl");
include("./EquationsOfState.jl")
include("./HyperelasticityMPh.jl")
include("./NumFluxes.jl")
include("./RelaxationInst.jl") # Подключаем наш модуль

# Импортируем используемые функции
using .EquationsOfState: entropy, EoS, Barton2009, stress, energy, pressure
using .HyperelasticityMPh: prim2cons_mph, cons2prim_mph, initial_states
using .NumFluxes
using .Strains: finger
using .SimpleLA: tr_
using .RelaxationInst: relaxation_inst, relaxation_vel, relaxation_pres, relaxation_vel_newton

using LinearAlgebra # Для dot, tr, I
using ForwardDiff: derivative # Для вычисления температуры

# --- Подготовка данных ---
println("=== Подготовка данных ===")

# Создаём EoS
eos = (Barton2009(), Barton2009())

# Создаём начальное состояние Q0 (длины 30)
# Ql, Qr = initial_states(eos, 71)
# Q0 = Array{Float64}(undef, 30)
# Q0 = vcat(Ql[1:15], Ql[1:15])
# Q0 = Ql
eos = (Barton2009(), Barton2009())
Ql, Qr = initial_states(eos, 7)
Q0 = Array{Float64}(undef, 30)
Q0[1:15] = Ql[1:15]
Q0[16:30] = Qr[1:15]
dt = 1e-5

# Q0 = [0.48963727371523286, 5.485159330384803, 1.4079647424910786, -0.1322314128744362, 0.042184118933328435, 3.402647885719543, 5.006549722221346, 0.05997698845311275, -0.07099740016438945, -0.006640764462277228, 5.284149265409944, -1.7733417080117044e-6, -0.02373994188127283, 0.009221746550980396, 5.006038602787315, 0.5103627262847672, 4.036691065724143, 2.206356789120675, 0.07574243446968718, 0.20202280596005381, 5.242252551399408, 3.8912153430440783, -0.036356406279523176, -0.09035332031850854, -0.00575591221990463, 4.494168994939205, -2.742044533662284e-5, -0.020251544106153336, 0.08327928079593491, 4.257538462812765]
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

# --- Выполнение релаксации скорости ---
println("\n=== Выполнение релаксации скорости ===")

# Вызов relaxation_vel
Q_after_vel = relaxation_vel(eos, Q0)
# Q_after_vel = relaxation_vel_newton(eos, Q0)

println("Выходной вектор Q_after_vel (после релаксации скорости):")
println(Q_after_vel)

# Преобразуем результат в примитивные переменные для анализа
P_after_vel = cons2prim_mph(eos, Q_after_vel)
P_after_vel_parts = [P_after_vel[i:i+14] for i in 1:15:length(P_after_vel)]

frac_after_vel = [P_after_vel_parts[i][1] for i in 1:2]
true_den_after_vel = [P_after_vel_parts[i][2] for i in 1:2]
den_after_vel = frac_after_vel .* true_den_after_vel
vel_after_vel = [P_after_vel_parts[i][3:5] for i in 1:2]
ent_after_vel = [P_after_vel_parts[i][6] for i in 1:2]
def_grad_after_vel = [P_after_vel_parts[i][7:15] for i in 1:2]
G_after_vel = [finger(def_grad_after_vel[i]) for i in 1:2]
e_int_after_vel = [energy(eos[i], ent_after_vel[i], G_after_vel[i]) for i in 1:2]
strs_after_vel = [reshape(stress(eos[i], ent_after_vel[i], def_grad_after_vel[i]), (3, 3)) for i in 1:2]
pres_after_vel = [- 1/3 * tr_(strs_after_vel[i]) for i in 1:2]

println("\n--- Анализ после релаксации скорости ---")
println("Фракции: ", frac_after_vel)
println("Плотности (rho): ", den_after_vel)
println("Скорости: ", vel_after_vel)
println("Энергии: ", e_int_after_vel)
println("Энтропии: ", ent_after_vel)
println("Давления: ", pres_after_vel)

# --- Выполнение релаксации ---
println("\n=== Выполнение релаксации ===")

# Вызов relaxation_inst
Q_relaxed = relaxation_inst(eos, Q0)

println("Выходной вектор Q_relaxed (после релаксации):")
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

println("\n--- Анализ после релаксации ---")
println("Фракции: ", frac_relaxed)
println("Плотности (rho): ", den_relaxed)
println("Скорости: ", vel_relaxed)
println("Энергии: ", e_int_relaxed)
println("Энтропии: ", ent_relaxed)
println("Давления: ", pres_relaxed)
println("Градиент деформации 1: ", def_grad_relaxed[1])
println("Градиент деформации 2: ", def_grad_relaxed[2])
println("Температуры (через derivative): ", [derivative(S -> energy(eos[i], S, G_relaxed[i]), ent_relaxed[i]) for i in 1:2])

# --- Проверка релаксации ---
println("\n=== Проверка релаксации ===")

println("Разница скоростей до релаксации: ", vel0[1] - vel0[2])
println("Разница скоростей после релаксации скорости: ", vel_after_vel[1] - vel_after_vel[2])
println("Разница скоростей после релаксации: ", vel_relaxed[1] - vel_relaxed[2])

println("Разница давлений до релаксации: ", pres0[1] - pres0[2])
println("Разница давлений после релаксации: ", pres_relaxed[1] - pres_relaxed[2])

println("Разница энтропий до релаксации: ", ent0[1] - ent0[2])
println("Разница энтропий после релаксации: ", ent_relaxed[1] - ent_relaxed[2])

# Проверим, сошёлся ли nlsolve
# (Это не всегда возможно проверить без доступа к объекту sol внутри relaxation_pres)
# Но если функция не падает и возвращает вектор той же длины, это уже хорошо.

println("\nТест завершён.")
