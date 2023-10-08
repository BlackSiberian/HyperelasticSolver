# Hyberbolic system solver
# Copyright (C) 2023    Ermakov Ilya
# e-mail: ermakoff.ilya@outlook.com

##### Condition functions ######

function LeftBoundaryCondition(t)
    return [pho_l, pho_l * u_l, p_l / (gamma - 1) + 0.5 * pho_l * u_l^2]
end

function RightBoundaryCondition(t)
    return [pho_r, pho_r * u_r, p_r / (gamma - 1) + 0.5 * pho_r * u_r^2]
end

function InitialCondition(i)
    if (i-1) * dx < 0.5
        return [pho_l, pho_l * u_l, p_l / (gamma - 1) + 0.5 * pho_l * u_l^2]
    else
        return [pho_r, pho_r * u_r, p_r / (gamma - 1) + 0.5 * pho_r * u_r^2]
    end
end

##### Calculating functions #####

function f(Q::Array) 
    rho = Q[1]
    u = Q[2] / rho
    etot = Q[3]
    eint = etot - 0.5 * rho * u^2
    p = eint * (gamma - 1)
    return [Q[2], 
            rho * u^2  + p,
            (etot + p) * u]
end

# Returns a value of flow between Q[1] and Q[2]
function LxF(Q::Array, dt) 
    return 0.5 * (f(Q[:, 1]) + f(Q[:, 2])) - 0.5 * dx/dt * (Q[:, 2] - Q[:, 1])
end

# Returns a value of cell on the next time layer
function CalculateCell(Q::Array, F::Function, dt) 
    return Q[:, 2] - dt/dx * (F(Q[:, 2:3], dt) - F(Q[:, 1:2], dt))
end

##### Condition constants #####

pho_l = 1.0 # Left boundary density
u_l = 0 # Left boundary gas velocity (along x)
p_l = 1.0 # Left boundary pressure

pho_r = 0.125 # Right boundary density
u_r = 0 # Right boundary gas velocity (along x)
p_r = 0.1 # Right boundary pressure

##### Main part of programm #####

X = 1.0 # Coordinate boundary [meters]
T = 0.2 # Time boundary [seconds]

nx = 500 # Number of steps on dimension coordinate
gamma = 1.4 # Polytropic index
cu = 0.99 # Courant number

dx = X / nx # Coordinate step

# Creating a 2-dimension calculation field
Q0 = Array{Float64}(undef, 3, nx) 

for i in 1:nx 
    Q0[:, i] = InitialCondition(i)
end

t = 0
while t < T
    # Finding time step
    lambda = abs(Q0[2, 1]/Q0[1, 1])
    for i in 1:nx
        rho = Q0[1, i]
        u = Q0[2, i] / rho
        etot = Q0[3, i]
        eint = etot - 0.5 * rho * u^2
        p = eint * (gamma - 1)
        c = sqrt(gamma * p / rho)
        lambda = max(abs(lambda), abs(u), abs(u + c), abs(u - c))
    end
    global dt = cu * dx / lambda
    ##
    global t += dt
    
    # Creating next time layer
    Q1 = similar(Q0)
    Q1[:, begin] = LeftBoundaryCondition(t)
    Q1[:, end] = RightBoundaryCondition(t)
    for i in 2:nx-1
        Q1[:, i] = CalculateCell(Q0[:, i-1:i+1], LxF, dt)
    end

    global Q0 = copy(Q1)
end

##### Plotting #####

using Plots
    
for j in 1:3   
    x = [dx * i for i in 0:nx-1]
    y = [Q0[j, i] for i in 1:nx]
    plot(x, y)
    savefig(string("Q", j, ".png"))
end