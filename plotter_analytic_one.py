import os
import numpy as np
import matplotlib.pyplot as plt
from math import exp

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './barton_data/'
plotpath = './plots/'
datafile = 'result.csv'
# datafile = 'sol_001600.csv'

# analyticpath = './analytic_data/'


# def read_analytic_data(filename):
#     data = np.loadtxt(analyticpath + filename, delimiter=';')
#     x = data[:, 0]
#     q = data[:, 1]
#     return x, q
#
#
# test = 1
# a_den = []
# a_x_den = []
# a_ent = []
# a_x_ent = []
# a_vel = [[], [], []]
# a_x_vel = [[], [], []]
#
# a_x_den, a_den = read_analytic_data(f'T{test}_den.csv')
# a_x_vel[0], a_vel[0] = read_analytic_data(f'T{test}_vel1.csv')
# a_x_vel[1], a_vel[1] = read_analytic_data(f'T{test}_vel2.csv')
# a_x_vel[2], a_vel[2] = read_analytic_data(f'T{test}_vel3.csv')
# a_x_ent, a_ent = read_analytic_data(f'T{test}_ent.csv')


Q = np.loadtxt(datapath + datafile, delimiter='\t', skiprows=2)

den = Q[:, 0]
vel = Q[:, 1:4]
ent = Q[:, 4]
strs = Q[:, 5:14]
strs = [strs[i, :].reshape(3, 3) for i in range(strs.size // 9)]
pres = -1/3 * sum([Q[:, i] for i in range(4, 13, 4)])
temp = [300 * exp(ent[i] / (9.3 * 1e-4) - 2 * (2.78 / den[i])) for i in range(len(ent))]
print(len(temp))
print(len(ent))

X = np.linspace(0, 1, len(ent))

frac_plt = plt.subplots()
den_plt = plt.subplots()
vel_plt = [plt.subplots() for _ in range(3)]
ent_plt = plt.subplots()
pres_plt = plt.subplots()
temp_plt = plt.subplots()

colors=['C4', 'C3']
titles=['Плотность',
        r'Скорость по координате $X$',
        r'Скорость по координате $Y$',
        r'Скорость по координате $Z$',
        'Энтропия', 'Давление', 'Температура']
ylabels=[r'$\rho, г/см^3$', r'$u_x, км/c$',
         r'$u_y, км/c$', r'$u_z, км/c$',
         r'$\eta, \,\frac{кДж}{г \, К}$',
         r'$P, Па$', r'$\Theta, К$']

# den_plt[1].plot(a_x_den, a_den, label='Аналитика', color='blue')
# ent_plt[1].plot(a_x_ent, a_ent, label='Аналитика', color='blue')
# for i in range(3):
#     vel_plt[i][1].plot(a_x_vel[i], a_vel[i], label='Аналитика', color='blue')

den_plt[1].plot(X, den, label=f'Фаза {0+1}', color=colors[0])
for i in range(3):
    vel_plt[i][1].plot(
        X, vel[:, i], label=f'Фаза {0+1}', color=colors[0])
ent_plt[1].plot(X, ent, label=f'Фаза {0+1}', color=colors[0])
pres_plt[1].plot(X, pres, label=f'Фаза {0+1}', color=colors[0])
temp_plt[1].plot(X, temp, label=f'Фаза {0+1}', color=colors[0])


t=0
for fig, ax in [den_plt, *vel_plt, ent_plt, pres_plt, temp_plt]:
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_title(titles[t])
    # ax.set_xlabel(r'X, см')
    ax.set_ylabel(ylabels[t])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(np.arange(0, 1.01, 0.05), minor=True)
    ax.legend()
    fig.tight_layout()
    t += 1

# vel_plt[1][1].set_yticks(np.arange(-0.06, 0.04, 0.01))
# vel_plt[2][1].set_yticks(np.arange(-0.01, 0.11, 0.01))
# ent_plt[1].set_yticks(np.arange(0.0, 2.5e-4, 2.5e-5))


den_plt[0].savefig(plotpath + 'density.png')
for i in range(3):
    vel_plt[i][0].savefig(plotpath + f'velocity_{i+1}.png')
ent_plt[0].savefig(plotpath + 'entropy.png')
pres_plt[0].savefig(plotpath + 'pressure.png')
temp_plt[0].savefig(plotpath + 'temperature.png')

# frac_plt[0].savefig(plotpath + 'fraction.eps', format="eps")
# den_plt[0].savefig(plotpath + 'density.eps', format="eps")
# for i in range(3):
#     vel_plt[i][0].savefig(plotpath + f'velocity_{i+1}.eps', format="eps")
# ent_plt[0].savefig(plotpath + 'entropy.eps', format="eps")

# for i in range(3):
#     plt.plot(X, frac[1] * vel[1][:, i] + frac[0] * vel[0][:, i])
#     plt.title(f'Weighted velocity {i+1}')
#     plt.grid()
#     plt.xlim(0, 1)
#     plt.savefig(plotpath + f'w_velocity_{i+1}.png')
#     plt.clf()
