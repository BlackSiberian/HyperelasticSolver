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

# Returns a triplet of eigenvalues
function EigenValue(Q::Array)
    rho = Q[1]
    u = Q[2] / rho
    etot = Q[3]
    eint = etot - 0.5 * rho * u^2
    p = eint * (gamma - 1)
    c = sqrt(gamma * p / rho)
    return [u, u + c, u - c]
end

# Returns the value of the flow in Q
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

# Returns the value of the flow between Q[1] and Q[2] using the Lax-Friedrichs method
function LxF(Q::Array, dt) 
    return 0.5 * (f(Q[:, 1]) + f(Q[:, 2])) - 0.5 * dx/dt * (Q[:, 2] - Q[:, 1])
end

# Returns the value of the flow between Q[1] and Q[2] using the HLL method
function HLL(Q::Array, dt)
    leigenvalues = EigenValue(Q[:, 1])
    reigenvalues = EigenValue(Q[:, 2])
    S_l = minimum([leigenvalues; reigenvalues])
    S_r = maximum([leigenvalues; reigenvalues])
    if S_r < 0
        return f(Q[:, 2])
    elseif S_l > 0
        return f(Q[:, 1])
    else
        Qhll = (f(Q[:, 1]) - f(Q[:, 2]) + Q[:, 2] .* S_r - Q[:, 1] .* S_l) ./ (S_r - S_l)
        Fhll = f(Q[:, 2]) + S_r .* (Qhll - Q[:, 2])
        return Fhll
    end
end

# Returns the value of quantity in the cell to the next time layer
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

nx = 2000 # Number of steps on dimension coordinate
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
    # Finding the time step
    lambda = abs(Q0[2, 1]/Q0[1, 1])
    for i in 1:nx
        eigenvalues = EigenValue(Q0[:, i])
        lambda = max(maximum(abs.(eigenvalues)), lambda)
    end
    global dt = cu * dx / lambda
    
    global t += dt
    
    # Creating the next time layer
    Q1 = similar(Q0)
    Q1[:, begin] = LeftBoundaryCondition(t)
    Q1[:, end] = RightBoundaryCondition(t)
    for i in 2:nx-1
        Q1[:, i] = CalculateCell(Q0[:, i-1:i+1], HLL, dt)
    end

    global Q0 = copy(Q1)
end

##### Plotting #####

using Plots
    
x = [dx * i for i in 0:nx-1]
rho = Q0[1, :]
u = Q0[2, :] ./ rho
etot = Q0[3, :]
eint = etot - 0.5 .* rho .* u.^2
p = eint .* (gamma - 1)

plot(x, rho)
savefig("density.png")
plot(x, u)
savefig("velocity.png")
plot(x, p)
savefig("pressure.png")