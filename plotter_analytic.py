import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './barton_data/'
plotpath = './plots/'
datafile = 'result.csv'
# datafile = 'sol_004700.csv'

analyticpath = './analytic_data/'


def read_analytic_data(filename):
    data = np.loadtxt(analyticpath + filename, delimiter=';')
    x = data[:, 0]
    q = data[:, 1]
    return x, q


test = 1
a_den = []
a_x_den = []
a_ent = []
a_x_ent = []
a_vel = [[], [], []]
a_x_vel = [[], [], []]

a_x_den, a_den = read_analytic_data(f'T{test}_den.csv')
a_x_vel[0], a_vel[0] = read_analytic_data(f'T{test}_vel1.csv')
a_x_vel[1], a_vel[1] = read_analytic_data(f'T{test}_vel2.csv')
a_x_vel[2], a_vel[2] = read_analytic_data(f'T{test}_vel3.csv')
a_x_ent, a_ent = read_analytic_data(f'T{test}_ent.csv')


Q = np.loadtxt(datapath + datafile, delimiter='\t', skiprows=1)

frac = [Q[:, 0], Q[:, 15]]
den = [Q[:, 1], Q[:, 16]]
vel = [Q[:, 2:5], Q[:, 17:20]]
ent = [Q[:, 5], Q[:, 20]]

X = np.linspace(0, 1, len(den[0]))

frac_plt = plt.subplots()
den_plt = plt.subplots()
vel_plt = [plt.subplots() for _ in range(3)]
ent_plt = plt.subplots()

colors = ['C4', 'C3']
titles = ['Объемная доля', 'Истинная плотность',
          r'Скорость по координате $X$',
          r'Скорость по координате $Y$',
          r'Скорость по координате $Z$', 'Энтропия']
ylabels = [r'$\alpha$', r'$\rho, г/см^3$', r'$u_x, км/c$',
           r'$u_y, км/c$', r'$u_z, км/c$',
           r'$\eta, \,\frac{\text{кДж}}{\text{г} \, \text{К}}$']

den_plt[1].plot(a_x_den, a_den, label='Аналитика', color='blue')
ent_plt[1].plot(a_x_ent, a_ent, label='Аналитика', color='blue')
for i in range(3):
    vel_plt[i][1].plot(a_x_vel[i], a_vel[i], label='Аналитика', color='blue')

for p in range(2):
    frac_plt[1].plot(X, frac[p], label=f'Фаза {p+1}', color=colors[p])
    den_plt[1].plot(X, den[p], label=f'Фаза {p+1}', color=colors[p])
    for i in range(3):
        vel_plt[i][1].plot(
            X, vel[p][:, i], label=f'Фаза {p+1}', color=colors[p])
    ent_plt[1].plot(X, ent[p], label=f'Фаза {p+1}', color=colors[p])

t = 0
for fig, ax in [frac_plt, den_plt, *vel_plt, ent_plt]:
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

frac_plt[0].savefig(plotpath + 'fraction.png')
den_plt[0].savefig(plotpath + 'density.png')
for i in range(3):
    vel_plt[i][0].savefig(plotpath + f'velocity_{i+1}.png')
ent_plt[0].savefig(plotpath + 'entropy.png')

# for i in range(3):
#     plt.plot(X, frac[1] * vel[1][:, i] + frac[0] * vel[0][:, i])
#     plt.title(f'Weighted velocity {i+1}')
#     plt.grid()
#     plt.xlim(0, 1)
#     plt.savefig(plotpath + f'w_velocity_{i+1}.png')
#     plt.clf()
