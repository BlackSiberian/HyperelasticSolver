module PlottingToolsTwoPhase

using Printf
using DelimitedFiles
using Plots
using LinearAlgebra
using LaTeXStrings
using ForwardDiff: derivative

export generate_plots_from_simulation_2ph, generate_plots_from_file_2ph

pyplot()

#==========================================================================================
                      НОВАЯ ФУНКЦИЯ-ПОМОЩНИК ДЛЯ ОБРАБОТКИ ДАННЫХ
==========================================================================================#

"""
    _process_phase_data(P, eos, stress_func)

Принимает матрицу примитивных переменных для ОДНОЙ фазы и возвращает все
рассчитанные величины для построения графиков.
"""
function _process_phase_data(P, eos, stress_func, energy_func, finger_func)
    nx = size(P, 1)

    # Извлекаем данные из матрицы примитивных переменных P
    frac = P[:, 1]
    # den = P[:, 2]
    vel = P[:, 3:5]
    ent = P[:, 6]
    def_grad_flat = P[:, 7:15]
    def_grad_mats = [reshape(def_grad_flat[i,:], 3, 3) for i in 1:nx]

    # Вычисляем физические величины
    den = [eos.rho0 / det(def_grad_mats[i]) for i in 1:nx]
    strs = [reshape(stress_func(eos, ent[i], def_grad_flat[i, :]), 3, 3) for i in 1:nx]
    pres = [-1/3 * tr(s) for s in strs]
    dist = [s + p * I(3) for (s, p) in zip(strs, pres)] # Девиатор = Напряжение + p*I

    G = [finger_func(def_grad_flat[i, :]) for i in 1:nx]
    temp = [derivative(S -> energy_func(eos, S, G[i]), ent[i]) for i in 1:nx]

    return frac, den, vel, ent, pres, temp, strs, dist
end


#==========================================================================================
                            ОБЩАЯ ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ
==========================================================================================#

function _create_and_save_plots_2ph(x_coords, data1, data2, plotpath::String;
                                    testcase::Int = 0,
                                    time::Float64 = 0.0,
                                    xlabel = L"Координата $X, м$")

    # Распаковываем кортежи с данными для удобства
    frac1, den1, vel1, ent1, pres1, temp1, strs1, dist1 = data1
    frac2, den2, vel2, ent2, pres2, temp2, strs2, dist2 = data2

    isdir(plotpath) || mkpath(plotpath)
    time_str = @sprintf("%.2e", time)
    units = (den="кг/м^3", vel="м/c", ent="Дж/(кг*К)", pres="Па", strs="Па", temp="К")

    # Конфигурации теперь содержат данные для двух фаз
    plot_configs = [
        # (d1=den1,      d2=den2,      title="Плотность (t=$(time_str) с)",    ylabel=L"$\rho, \, %$(units.den)$", filename="test$(testcase)_density.png"),
        # (d1=vel1[:,1], d2=vel2[:,1], title="Скорость по X (t=$(time_str) с)",        ylabel=L"$u_x, \, %$(units.vel)$",  filename="test$(testcase)_velocity_1.png"),
        # (d1=vel1[:,2], d2=vel2[:,2], title="Скорость по Y (t=$(time_str) с)",        ylabel=L"$u_y, \, %$(units.vel)$",  filename="test$(testcase)_velocity_2.png"),
        # (d1=pres1,     d2=pres2,     title="Давление (t=$(time_str) с)",             ylabel=L"$P, \, %$(units.pres)$", filename="test$(testcase)_pressure.png"),
        # # (d1=temp1,     d2=temp2,     title="Температура (t=$(time_str) с)",             ylabel=L"$P, \, %$(units.temp)$", filename="test$(testcase)_temperature.png"),
        # (d1=[s[1,1] for s in strs1], d2=[s[1,1] for s in strs2], title="Напряжение XX (t=$(time_str) с)", ylabel=L"$\sigma_{xx}, \, %$(units.strs)$", filename="test$(testcase)_stress_xx.png"),
        # # (d1=[d[1,1] for d in dist1], d2=[d[1,1] for d in dist2], title="Девиатор XX (t=$(time_str) с)", ylabel=L"$S_{xx}, \, %$(units.strs)$", filename="test$(testcase)_deviator.png"),
        # (d1=[d[1,2] for d in dist1], d2=[d[1,2] for d in dist2], title="Девиатор XY (t=$(time_str) с)", ylabel=L"$S_{xy}, \, %$(units.strs)$", filename="test$(testcase)_deviator.png"),
        (d1=den1,      d2=den2,      title="Плотность",    ylabel=L"$\rho, \, %$(units.den)$", filename="test$(testcase)_density.png"),
        (d1=vel1[:,1], d2=vel2[:,1], title="Скорость по X",        ylabel=L"$u_x, \, %$(units.vel)$",  filename="test$(testcase)_velocity_1.png"),
        (d1=vel1[:,2], d2=vel2[:,2], title="Скорость по Y",        ylabel=L"$u_y, \, %$(units.vel)$",  filename="test$(testcase)_velocity_2.png"),
        (d1=pres1,     d2=pres2,     title="Давление",             ylabel=L"$P, \, %$(units.pres)$", filename="test$(testcase)_pressure.png"),
        # (d1=temp1,     d2=temp2,     title="Температура (t=$(time_str) с)",             ylabel=L"$P, \, %$(units.temp)$", filename="test$(testcase)_temperature.png"),
        (d1=[s[1,1] for s in strs1], d2=[s[1,1] for s in strs2], title="Напряжение XX", ylabel=L"$\sigma_{xx}, \, %$(units.strs)$", filename="test$(testcase)_stress_xx.png"),
        # (d1=[d[1,1] for d in dist1], d2=[d[1,1] for d in dist2], title="Девиатор XX (t=$(time_str) с)", ylabel=L"$S_{xx}, \, %$(units.strs)$", filename="test$(testcase)_deviator.png"),
        (d1=[d[1,2] for d in dist1], d2=[d[1,2] for d in dist2], title="Девиатор напряжений XY", ylabel=L"$S_{xy}, \, %$(units.strs)$", filename="test$(testcase)_deviator.png"),
    ]

    # den = den1 .* frac1 .+ den2 .* frac2
    # den_config = (d=den, title="Плотность (t=$(time_str) с)",    ylabel=L"$\rho, \, %$(units.den)$", filename="test$(testcase)_density.png")
    # p = plot(x_coords, den_config.d; label="Avg", color=:blue, legend=:best)
    # plot!(p;
    #       title=den_config.title,
    #       ylabel=den_config.ylabel,
    #       xlabel=xlabel,
    #       xlims=(minimum(x_coords), maximum(x_coords)),
    #       minorgrid=true, grid=true,
    # )
    # savefig(p, joinpath(plotpath, den_config.filename))

    analytics = readdlm("analytic_data/Test_1_MN.csv", ','; skipstart=2)

    default(
        fontfamily="serif",
        titlefont="serif",
        guidefont="serif",  # Подписи осей
        tickfont="serif",   # Цифры на осях
        legendfont="serif"
    )

    i = 0
    for config in plot_configs
        x_analytics = filter(x -> x != "", analytics[:, 1+2i])
        y_analytics = filter(x -> x != "", analytics[:, 2+2i])
        p = plot(x_analytics, y_analytics;
                 label="Уилкинс",
                 lw=2, color=:red)
        i += 1
        # plot!(p, x_coords, config.d1; label="Фаза 1", color=:blue, legend=:best)
        # plot!(p, x_coords, config.d2; label="Фаза 2", color=:red)
        plot!(p, x_coords, frac1 .* config.d1 + frac2 .* config.d2;
              label="Гиперупругость",
              lw=2.5, color=:black)

        plot!(p;
            title=config.title,
            ylabel=config.ylabel,
            xlabel=xlabel,
            xlims=(minimum(x_coords), maximum(x_coords)),
            minorgrid=true, grid=true,
            dpi=300,
            legend=false,
        )
        savefig(p, joinpath(plotpath, config.filename))
    end
end

#==========================================================================================
                                ТОЧКИ ВХОДА ДЛЯ ПОЛЬЗОВАТЕЛЯ
==========================================================================================#

function generate_plots_from_simulation_2ph(Q, eos1, eos2, X_len::Float64, time::Float64, testcase::Int, plotpath::String, cons2prim_func, stress_func)
    println("Построение 2-фазных графиков из результатов симуляции...")
    nx = size(Q, 1)
    x_coords = range(0, X_len, length=nx)

    # !!! ВАЖНОЕ ИСПРАВЛЕНИЕ: Q нужно транспонировать для корректной работы cons2prim
    Q_transposed = Q

    # Разделяем Q на две фазы
    Q1 = Q_transposed[1:15, :]  # Берем первые 15 столбцов для всех nx точек
    Q2 = Q_transposed[16:30, :] # Берем вторые 15 столбцов для всех nx точек

    # Обрабатываем фазу 1
    P1_vecs = [cons2prim_func(eos1, Q1[i, :]) for i in 1:nx]
    P1 = hcat(P1_vecs...)
    data1 = _process_phase_data(P1, eos1, stress_func)

    # Обрабатываем фазу 2
    P2_vecs = [cons2prim_func(eos2, Q2[i, :]) for i in 1:nx]
    P2 = hcat(P2_vecs...)
    data2 = _process_phase_data(P2, eos2, stress_func)

    _create_and_save_plots_2ph(x_coords, data1, data2, plotpath; testcase=testcase, time=time, xlabel=L"Координата $X, м$")
end

function generate_plots_from_file_2ph(filepath::String, plotpath::String, eos1, eos2, stress_func, energy_func, finger_func;
                                      testcase=0)
    println("Построение 2-фазных графиков из файла '$filepath'...")

    local time_from_file::Float64
    local P_all::Matrix{Float64}

    open(filepath, "r") do io
        time_line = readline(io)
        time_from_file = parse(Float64, time_line)
        println("... Обнаружено время симуляции: ", time_from_file)
        readline(io) # Пропускаем заголовок
        P_all = readdlm(io, '\t', Float64)
    end

    # Разделяем прочитанные данные на две фазы
    # Предполагаем, что первые 15 столбцов - фаза 1, следующие 15 - фаза 2
    P1 = P_all[:, 1:15]
    P2 = P_all[:, 16:30]

    # Обрабатываем каждую фазу с помощью нашей новой функции
    data1 = _process_phase_data(P1, eos1, stress_func, energy_func, finger_func)
    data2 = _process_phase_data(P2, eos2, stress_func, energy_func, finger_func)

    nx = size(P_all, 1)
    x_coords = range(0, 0.1, length=nx)
    xlabel_file = L"Координата $X, м$"

    _create_and_save_plots_2ph(x_coords, data1, data2, plotpath;
                               testcase=testcase, time=time_from_file, xlabel=xlabel_file)
end

#==========================================================================================
                         БЛОК ДЛЯ АВТОНОМНОГО ЗАПУСКА СКРИПТА
==========================================================================================#
if abspath(PROGRAM_FILE) == @__FILE__
    println("Скрипт запущен автономно. Подгрузка зависимостей...")

    include("./SimpleLA.jl")
    include("./Strains.jl")
    include("./EquationsOfState.jl")
    include("./Hyperelasticity.jl")

    # Создаем два объекта EoS, по одному для каждой фазы
    eos1_default = EquationsOfState.Stiffened(rho0=2780.0, s=1.338, c0=5330.0, cv=9.3e2, mu=27.6e9, G0=2.13, T0=300, S0=0.0)
    # Для примера, второй материал может быть другим
    eos2_default = EquationsOfState.Stiffened(rho0=8930.0, s=1.49,  c0=3970.0, cv=3.9e2, mu=45.0e9, G0=2.00, T0=300, S0=0.0)

    datapath = "./barton_data/"
    plotpath = "./plots_2ph/"
    datafile = "result.csv" # Предполагаем новое имя файла для 2-фазных данных
    filepath = joinpath(datapath, datafile)

    generate_plots_from_file_2ph(
        filepath,
        plotpath,
        eos1_default,
        eos2_default,
        EquationsOfState.stress,
        EquationsOfState.energy,
        Strains.finger;
        testcase=11, # Новый номер теста
    )

    println("2-фазные графики успешно сохранены в директорию: $plotpath")
end

end
