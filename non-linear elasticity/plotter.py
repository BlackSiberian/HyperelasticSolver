import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './data/'
plotpath = './plots/'
datafile = 'plot.dat'

Q = np.loadtxt(datapath + datafile, delimiter='\t')
den = Q[:, 0]
vel = Q[:, 1:4]
stress = np.reshape(Q[:, 4:13], (len(den), 3, 3))
entropy = Q[:, 13]
X = np.linspace(0, 1, len(den))

data = np.loadtxt('./testcase/Test_2_Density.csv', delimiter=';')
plt.plot(data[:, 0], data[:, 1])
plt.plot(X, den)
plt.legend(['Analytical', 'Numerical'])
plt.title('Density')
plt.grid()
plt.xlim(0, 1)
plt.savefig(plotpath + 'density.png')
plt.clf()

for i in range(3):
    data = np.loadtxt(f'./testcase/Test_2_Velocity_{i+1}.csv', delimiter=';')
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(X, vel[:, i])
    plt.legend(['Analytical', 'Numerical'])
    plt.title(f'Velocity_{i+1}')
    plt.grid()
    plt.xlim(0, 1)
    plt.savefig(plotpath + f'velocity_{i+1}.png')
    plt.clf()
    for j in range(i+1):
        data = np.loadtxt(f'./testcase/Test_2_Stress_{j+1}{i+1}.csv', delimiter=';')
        plt.plot(data[:, 0], data[:, 1])
        plt.plot(X, stress[:, i, j])
        plt.legend(['Analytical', 'Numerical'])
        plt.title(f'Stress_{j+1}{i+1}')
        plt.grid()
        plt.xlim(0, 1)
        plt.savefig(plotpath + f'stress_{j+1}{i+1}.png')
        plt.clf()

data = np.loadtxt('./testcase/Test_2_Entropy.csv', delimiter=';')
plt.plot(data[:, 0], data[:, 1])
plt.plot(X, entropy)
plt.legend(['Analytical', 'Numerical'])
plt.title('Entropy')
plt.grid()
plt.xlim(0, 1)
plt.savefig(plotpath + 'entropy.png')
plt.clf()





