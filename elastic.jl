##### Neccessery includes #####

using LinearAlgebra
using Plots

##### Condition functions ######

function LeftBoundaryCondition(t)
    rho = rho0 / det(abs.(F_l))
    Q = Array{Float64}(undef, 13)
    for i in 1:3
        Q[i] = rho * u_l[i]
        for j in 1:3
            Q[3*i + j] = rho * F_l[i, j]
        end
    end
    G = inv(F_l * transpose(F_l))
    I1 = tr(G)
    I3 = det(abs.(G))
    I2 = 0.5 * (tr(G)^2 - tr(G^2))
    B0 = b0^2
    K0 = c0^2 - (4/3)*b0^2
    U = 0.5 * K0 / (alpha^2) * (I3^(0.5*alpha) - 1)^2 + cv * T0 * I3^(0.5*gamma) * (exp(S_l / cv) - 1)
    W = 0.5 * B0 * I3^(0.5*beta)*(I1^2 / 3 - I2)
    eint = U + W
    E = eint + 0.5 * (u_l[1]^2 + u_l[2]^2 + u_l[3]^2)
    Q[13] = rho * E
    return Q
end

function RightBoundaryCondition(t)
    rho = rho0 / det(abs.(F_r))
    Q = Array{Float64}(undef, 13)
    for i in 1:3
        Q[i] = rho * u_r[i]
        for j in 1:3
            Q[3*i + j] = rho * F_r[i, j]
        end
    end
    G = inv(F_r * transpose(F_r))
    I1 = tr(G)
    I3 = det(abs.(G))
    I2 = 0.5 * (tr(G)^2 - tr(G^2))
    B0 = b0^2
    K0 = c0^2 - (4/3)*b0^2
    U = 0.5 * K0 / (alpha^2) * (I3^(0.5*alpha) - 1)^2 + cv * T0 * I3^(0.5*gamma) * (exp(S_r / cv) - 1)
    W = 0.5 * B0 * I3^(0.5*beta)*(I1^2 / 3 - I2)
    eint = U + W
    E = eint + 0.5 * (u_r[1]^2 + u_r[2]^2 + u_r[3]^2)
    Q[13] = rho * E
    return Q
end

function InitialCondition(i)
    if (i-1) * dx < 0.5 * X
        return LeftBoundaryCondition(0)
    else
        return RightBoundaryCondition(0)
    end
end

##### Calculating functions #####

# # Returns a triplet of eigenvalues
# function EigenValue(Q::Array)
#     rho = Q[1]
#     u = Q[2] / rho
#     etot = Q[3]
#     eint = etot - 0.5 * rho * u^2
#     p = eint * (gamma - 1)
#     c = sqrt(gamma * p / rho)
#     return [u, u + c, u - c]
# end

# Returns the value of the flow in Q
function f(Q::Array) 
    FQ = [Q[4] Q[5] Q[6];
          Q[7] Q[8] Q[9];
          Q[10] Q[11] Q[12]]
    rho = sqrt(det(abs.(FQ))/rho0)
    F = FQ ./ rho
    G = inv(F * transpose(F))
    K0 = c0^2 - (4/3)*b0^2
    B0 = b0^2
    I1 = tr(G)
    I3 = det(abs.(G))
    I2 = 0.5 * (tr(G)^2 - tr(G^2))
    e1 = B0 * I1 * I3^(0.5*beta) / 3
    e2 = - 0.5 * B0 * I3^(0.5*beta)
    e3 = 0.5 * K0 / alpha * (I3^(0.5*alpha) - 1) + 0.25 * beta * B0 * (I1^2 / 3 - I2) * I3^(0.5*beta-1)
    e3 += 0.5 * gamma * (Q[13] / rho - 0.5 * (Q[1]^2 + Q[2]^2 + Q[3]^2) / (rho^2) - 0.5 * B0 * I3^(0.5*beta) * (I1^2 / 3 - I2) - 0.5 * K0 / (alpha^2) * (I3^(0.5*alpha) - 1)^2) / I3
    sigma = -2 * rho .* G * (e1 .* I + e2 * I1 .* I - e2 .* G + e3 * I3 .* inv(G))

    f = similar(Q)
    for i in 1:3
        f[i] = Q[i]^2 / rho - sigma[1, i]
        f[i+3] = 0
        f[i+6] = (Q[i] * Q[1] - Q[i+3] * Q[2]) / rho
        f[i+9] = (Q[i+9] * Q[1] - Q[i+3] * Q[3]) / rho
    end
    f[13] = (Q[13] * Q[1] - Q[1] * sigma[1, 1] - Q[2] * sigma[1, 2] - Q[3] * sigma[1, 3]) / rho
    return f
end

# Returns the value of the flow between Q[1] and Q[2] using the Lax-Friedrichs method
function LxF(Q::Array, dt) 
    return 0.5 * (f(Q[:, 1]) + f(Q[:, 2])) - 0.5 * dx/dt * (Q[:, 2] - Q[:, 1])
end

# Returns the value of quantity in the cell to the next time layer
function CalculateCell(Q::Array, F::Function, dt) 
    return Q[:, 2] - dt/dx * (F(Q[:, 2:3], dt) - F(Q[:, 1:2], dt))
end

##### Condition constants #####

# u_l = [0, 0.5, 1] # [km/s]
# F_l = [0.98 0   0;
#        0.02 1   0.1; 
#         0   0   1]
# S_l = 1e-3 # [kJ/(g*K)]

# u_r = [0, 0, 0] # [km/s]
# F_r = [1    0   0
#        0    1   0.1
#        0    0   1]
# S_r = 0 # [kJ/(g*K)]

u_l = [2, 0, 0.1] # [km/s]
F_l = [1        0   0;
       -0.01  0.95  0.02; 
       -0.015   0   0.9]
S_l = 0 # [kJ/(g*K)]

u_r = [0, -0.03, -0.01] # [km/s]
F_r = [1        0      0
       0.015    0.95   0
       -0.01    0     0.9]
S_r = 0 # [kJ/(g*K)]

rho0 = 8.93 # Initial density [g/cm^3]
c0 = 4.6 # Speed of sound [km/s]
cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
T0 = 300 # Initial temperature [K]
b0 = 2.1 # Speed of the shear wave [km/s]
alpha = 1.0 # Non-linear
beta = 3.0  # characteristic
gamma = 2.0 # constants

##### Main part of programm #####

X = 1.0 # Coordinate boundary [m]
T = 0.6e-3 # Time boundary [s]

nx = 500 # Number of steps on dimension coordinate
# gamma = 1.4 # Polytropic index
cu = 0.6 # Courant number

dx = X / nx # Coordinate step

# Creating a 2-dimension calculation field

Q0 = Array{Float64}(undef, 13, nx)

for i in 1:nx 
    Q0[:, i] = InitialCondition(i)
end

t = 0
while t < T
    # Finding the time step
    lambda = 0
    for i in 1:nx
        # eigenvalues = EigenValue(Q0[:, i])
        # lambda = max(maximum(abs.(eigenvalues)), lambda)
        Q = Q0[:, i]
        FQ = [Q[4]  Q[5]  Q[6];
              Q[7]  Q[8]  Q[9];
              Q[10] Q[11] Q[12]]
        rho = sqrt(det(abs.(FQ))/rho0)
        lambda = max(abs(Q0[1, i]) / rho + 5000, lambda) # TODO replace 5000 with max eigenvalue
    end
    global dt = cu * dx / lambda
    
    global t += dt
    
    # Creating the next time layer
    Q1 = similar(Q0)
    Q1[:, begin] = LeftBoundaryCondition(t)
    Q1[:, end] = RightBoundaryCondition(t)
    for i in 2:nx-1
        Q1[:, i] = CalculateCell(Q0[:, i-1:i+1], LxF, dt)
    end

    global Q0 = copy(Q1)
    println("t = ", t)
end

##### Plotting #####

# TODO rewrite Plotting

x = [dx * i for i in 0:nx-1]
rho = Array{Float64}(undef, nx)
u1 = Array{Float64}(undef, nx)
u2 = Array{Float64}(undef, nx)
u3 = Array{Float64}(undef, nx)
for i in 1:nx
    Q = Q0[:, i]
    FQ = [Q[4] Q[5] Q[6];
          Q[7] Q[8] Q[9];
          Q[10] Q[11] Q[12]]
    rho[i] = sqrt(det(abs.(FQ))/rho0)
    u1[i] = Q[1] / rho[i]
    u2[i] = Q[2] / rho[i]
    u3[i] = Q[3] / rho[i]
end

plot(x, rho)
savefig("density.png")
plot(x, u1)
savefig("velocity_x.png")
plot(x, u2)
savefig("velocity_y.png")
plot(x, u3)
savefig("velocity_z.png")