import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './plot_data/'
plotpath = './plots/'
lxfdatafile = 'lxf.dat'
hlldatafile = 'hll.dat'

Q = np.loadtxt(datapath + hlldatafile, delimiter='\t')
x_h = Q[:, 0]
den_h = Q[:, 1]
vel_h = Q[:, 2]
press_h = Q[:, 3]
energy_h = Q[:, 4] / Q[:, 1]

Q = np.loadtxt(datapath + lxfdatafile, delimiter='\t')
x_l = Q[:, 0]
den_l = Q[:, 1]
vel_l = Q[:, 2]
press_l = Q[:, 3]
energy_l = Q[:, 4] / Q[:, 1]

def plotter(var_l, var_h, name, dataname):
    data = np.loadtxt(datapath + dataname, delimiter=';')
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(x_l, var_l)
    plt.plot(x_h, var_h)
    plt.legend(['Analytical', 'Lax-Friedriechs', 'HLL'])
    plt.title(name)
    plt.grid()
    plt.xlim(0, 1)
    plt.savefig(plotpath + f'{name}.png')
    plt.clf()

plotter(den_l, den_h, 'Density', 'Sod_Density.csv')
plotter(press_l, press_h, 'Pressure', 'Sod_Pressure.csv')
plotter(vel_l, vel_h, 'Velocity', 'Sod_Velocity.csv')
plotter(energy_l, energy_h, 'Specific energy', 'Sod_Energy.csv')