import os
import numpy as np
import matplotlib.pyplot as plt

# Choose path to plots and to files to plot from
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
# datapath = './plot_data/'
datapath = './barton_data/'
plotpath = './plots/'
# files = ['advanced1800conv.csv']
files = ['result.csv']

def add_to_plt(path, color):
    Q = np.loadtxt(path, delimiter='\t', skiprows=1)

    frac = [Q[:, 0], Q[:, 15]]
    den = [Q[:, 1], Q[:, 16]]
    vel = [Q[:, 2:5], Q[:, 17:20]]
    ent = [Q[:, 5], Q[:, 20]]
    strs = [Q[:, 6:15], Q[:, 21:30]]

    X = np.linspace(0, 1, len(den[0]))

    for p in range(2):
        frac_plt[1].plot(X, frac[p], color=colors[color], linestyle=styles[p])
        den_plt[1].plot(X, den[p], color=colors[color], linestyle=styles[p])
        for i in range(3):
            vel_plt[i][1].plot(
                X, vel[p][:, i], color=colors[color], linestyle=styles[p])
        ent_plt[1].plot(X, ent[p], color=colors[color], linestyle=styles[p])
        for j in range(3):
            for i in range(j, 3):
                strs_plt[i + j*3][1].plot(X, strs[p][:, i + j*3], color=colors[color], linestyle=styles[p])

frac_plt = plt.subplots()
den_plt = plt.subplots()
vel_plt = [plt.subplots() for _ in range(3)]
ent_plt = plt.subplots()
strs_plt =[plt.subplots() for _ in range(9)]

colors = ['black']
labels = [r'$\Delta x = 1/5000$, ', r'$\Delta x = 1/10000$, ']
styles = ['-', '--']
# titles = ['Объемная доля', 'Истинная плотность',
#           r'Скорость по координате $X$',
#           r'Скорость по координате $Y$',
#           r'Скорость по координате $Z$', 'Энтропия']
# ylabels = [r'$\alpha$', r'$\rho, г/см^3$', r'$u_x, км/c$',
#            r'$u_y, км/c$', r'$u_z, км/c$',
#            r'$\eta, \,\frac{\text{кДж}}{\text{г} \, \text{К}}$']

c = 0
for file in files:
    add_to_plt(datapath + file, c)
    c += 1

# t = 0
for fig, ax in [frac_plt, den_plt, *vel_plt, ent_plt, *strs_plt]:
    ax.grid()
    ax.set_xlim(0, 1)
    # ax.set_title(titles[t])
    # ax.set_xlabel(r'X, см')
    # ax.set_ylabel(ylabels[t])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticks(np.arange(0, 1.01, 0.05), minor=True)
    # ax.legend()
    fig.tight_layout()
    fig.set_size_inches(8, 6)
    fig.set_dpi(600)
    # t += 1

# frac_plt[0].savefig(plotpath + 'fraction.png')
# den_plt[0].savefig(plotpath + 'density.png')
# for i in range(3):
#     vel_plt[i][0].savefig(plotpath + f'velocity_{i+1}.png')
# ent_plt[0].savefig(plotpath + 'entropy.png')
# for j in range(3):
#     for i in range(j, 3):
#         strs_plt[i+j*3][0].savefig(plotpath + f'stress_{i+1}{j+1}.png')

frac_plt[0].savefig(plotpath + 'fraction.eps', format='eps')
den_plt[0].savefig(plotpath + 'density.eps', format='eps')
for i in range(3):
    vel_plt[i][0].savefig(plotpath + f'velocity_{i+1}.eps', format='eps')
ent_plt[0].savefig(plotpath + 'entropy.eps', format='eps')
for j in range(3):
    for i in range(j, 3):
        strs_plt[i+j*3][0].savefig(plotpath + f'stress_{i+1}{j+1}.eps', format='eps')