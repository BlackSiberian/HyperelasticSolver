import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './barton_data/'
plotpath = './plots/'
# datafile = 'plot.dat'
datafile = 'sol_000000.csv'

Q = np.loadtxt(datapath + datafile, delimiter='\t', skiprows=1)

# alpha = []
# den = []
# vel = []
# distortion = []
# entropy = []

den = [Q[:, 1], Q[:, 16]]
vel = [Q[:, 2:5], Q[:, 17:20]]
entropy = [Q[:, 5], Q[:, 20]] 

X = np.linspace(0, 1, len(den[0]))

for p in range(2):
    plt.plot(X, den[p])
    plt.title(f'Density {p+1}')
    plt.grid()
    plt.xlim(0, 1)
    plt.savefig(plotpath + f'density_{p+1}.png')
    plt.clf()

    for i in range(3):
        plt.plot(X, vel[p][:, i])
        plt.title(f'Velocity {i+1} {p+1}')
        plt.grid()
        plt.xlim(0, 1)
        plt.savefig(plotpath + f'velocity_{i+1}_{p+1}.png')
        plt.clf()

    plt.plot(X, entropy[p])
    plt.title(f'Entropy {p+1}')
    plt.grid()
    plt.xlim(0, 1)
    plt.savefig(plotpath + f'entropy_{p+1}.png')
    plt.clf()





