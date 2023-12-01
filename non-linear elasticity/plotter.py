import os
import numpy as np
import matplotlib.pyplot as plt

def density(Q: np.array):
    FQ = np.reshape(Q[3:12], (3, 3))
    return (np.linalg.det(FQ) / rho0)**0.5

def cons2Prim(Q: np.array):
    FQ = np.reshape(Q[3:12], (3, 3))
    den = density(Q)
    vel = Q[0:3] / den
    F = FQ / den
    e_int = Q[12] / den - 0.5 * np.dot(vel, vel)
    return den, vel, F, e_int

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

os.chdir(script_dir)
datapath = './data/'

rho0 = 8.93
data_array = np.array([])
datafile1 = 'final.dat'

for datafile in os.listdir(datapath):
    Q = np.loadtxt(datapath + datafile1, delimiter='\t', skiprows=1)
    with open(datapath + datafile, 'r') as file:
        first_line = file.readline()
    T = float(first_line)
    den = np.zeros(len(Q))
    vel = np.zeros((len(Q), 3))
    F = np.zeros((len(Q), 3, 3))
    e_int = np.zeros(len(Q))
    for i in range(len(Q)):
        den[i], vel[i], F[i], e_int[i] = cons2Prim(Q[i])
    X = np.linspace(0, 1, len(Q))

    plt.figure(figsize=(10, 10))

    # Plot density
    plt.subplot(2, 2, 1)
    plt.plot(X, den)
    plt.xlabel('x')
    plt.ylabel('density')

    # Plot velocity - x-component
    plt.subplot(2, 2, 2)
    plt.plot(X, vel[:, 0])
    plt.xlabel('x')
    plt.ylabel('velocity - x-component')

    # Plot velocity - y-component
    plt.subplot(2, 2, 3)
    plt.plot(X, vel[:, 1])
    plt.xlabel('x')
    plt.ylabel('velocity - y-component')

    # Plot velocity - z-component
    plt.subplot(2, 2, 4)
    plt.plot(X, vel[:, 2])
    plt.xlabel('x')
    plt.ylabel('velocity - z-component')


    plt.savefig('plots/' + 'plot_' + datafile1[:-4] + '.png')
    plt.close()


    break


