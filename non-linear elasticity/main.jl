using LinearAlgebra
using Plots
using Printf

# Returns the symmetric Finger's tensor 
Finger(F) = inv(F * transpose(F))

# Returns the density from conservative variables
# @param Q is the conservative variables vector
function Density(Q::Array)
    FQ = [Q[4]  Q[5]  Q[6];     # The part of the vector that
          Q[7]  Q[8]  Q[9];     # corresponds to the deformation gradient
          Q[10] Q[11] Q[12]]
    return sqrt(det(FQ) / rho0)
end

# Returns the triplet of symmetric tensor invariants
# @param F is the elastic deformation gradient tensor
function Invariants(G::Array)
    I1 = tr(G)
    I2 = 0.5 * (tr(G)^2 - tr(G^2))
    I3 = det(G)
    return [I1, I2, I3]
end

# Returns the value of the entropy
# @param e_int is the internal energy
# @param i is invariants
function Entropy(e_int, i::Array)
    B0 = b0^2
    K0 = c0^2 - (4/3)*b0^2
    S = e_int - 0.5 * B0 * i[3]^(0.5*beta) * (i[1]^2 / 3 - i[2]) - 0.5 * K0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2
    S = (S / (cv * T0 * i[3]^(0.5*gamma)) + 1)
    return log(S) * cv
end

# Returns the value of the internal energy
# @param S is entropy
# @param i is invariants
function EoS(S, i::Array)
    B0 = b0^2
    K0 = c0^2 - (4/3)*b0^2
    U = 0.5 * K0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2 + cv * T0 * i[3]^(0.5*gamma) * (exp(S / cv) - 1)
    W = 0.5 * B0 * i[3]^(0.5*beta)*(i[1]^2 / 3 - i[2])
    e_int = U + W
    return e_int
end

# Returns the triplet of derivatives of the internal energy to the invariants
# @param e_int is the internal energy
# @param i is invariants
function dEoS(e_int, i::Array)
    B0 = b0^2               # Squared speed of the shear wave
    K0 = c0^2 - (4/3)*b0^2  # Square bulk speed of sound
    
    # TODO: Probably implement automatic differentiation
    e1 = B0 * i[1] * i[3]^(0.5*beta) / 3.0
    e2 = - 0.5 * B0 * i[3]^(0.5*beta)
    e3 = 0.5 * K0 / alpha * (i[3]^(0.5*alpha) - 1) * i[3]^(0.5*alpha-1) + 0.25 * beta * B0 * (i[1]^2 / 3 - i[2]) * i[3]^(0.5*beta-1)
    e3 += 0.5 * gamma * (e_int - 0.5 * B0 * i[3]^(0.5*beta) * (i[1]^2 / 3 - i[2]) - 0.5 * K0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2) / i[3]

    return [e1, e2, e3]
end

dI1dG(G, I1, I2, I3) = I            # Returns the derivative of I with respect to G
dI2dG(G, I1, I2, I3) = I1 .* I - G  # @param G is the symmetric tensor
dI3dG(G, I1, I2, I3) = I3 .* inv(G) # @param I1, I2, I3 are the invariants

# Returns the stress tensor
function Stress(den, e_int, F::Array)
    G = Finger(F) 
    I1, I2, I3 = Invariants(G)
    G = Finger(F)

    e1, e2, e3 = dEoS(e_int, [I1, I2, I3])

    return -2.0 * den .* G * (e1 .* dI1dG(G, I1, I2, I3) + e2 .* dI2dG(G, I1, I2, I3) + e3 .* dI3dG(G, I1, I2, I3))
end

# Translates conservative variables to primitive variables
# @param Q is the conservative variables vector
function Cons2Prim(Q::Array)
    FQ = [ Q[4]  Q[5]  Q[6];                              
           Q[7]  Q[8]  Q[9];
           Q[10] Q[11] Q[12]] 
    den = Density(Q)
    vel = Q[1:3] ./ den
    F = FQ ./ den
    E = Q[13] / den
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    e_int = E - e_kin
    return [den, vel, F, e_int]
end

# TODO: Prim2Cons is equivalent to BoundaryCondition
# Translates primitive variables to conservative variables
# @param vel is the velocity vector
# @param F is the deformation gradient tensor
# @param S is the entropy
function Prim2Cons(vel::Array, F::Array, S)
    Q = Array{Float64}(undef, 13)
    den = rho0 / det(F)
    Q[1:3] = den .* vel
    for i in 1:3
        for j in 1:3
            Q[3*i + j] = den * F[i, j]
        end
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    G = Finger(F) 
    i = Invariants(G)
    E = EoS(S, i) + e_kin
    Q[13] = den * E
    return Q
end

# Sets the boundary conditions to the solution
# @param vel is the velocity vector
# @param F is the deformation gradient tensor
# @param S is the entropy
# function BoundaryCondition(vel::Array, F::Array, S)
#     Q = Array{Float64}(undef, 13) # Conservative variables vector
#     den = rho0 / det(F)
#     for i in 1:3
#         Q[i] = den * vel[i]
#         for j in 1:3
#             Q[3*i + j] = den * F[i, j]
#         end
#     end
#     e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
#     G = Finger(F) 
#     i = Invariants(G)
#     E = EoS(S, i) + e_kin
#     Q[13] = den * E
#     return Q
# end

# Returns the value of the physical flux in Q
# @param Q is the conservative variables vector
function Flux(Q::Array) 
    den, vel, F, e_int = Cons2Prim(Q)
    sigma = Stress(den, e_int, F)

    flux = similar(Q)
    for i in 1:3
        flux[i] = den * vel[1] * vel[i] - sigma[1, i]
        flux[i+3] = 0
        flux[i+6] = den * (F[2, i] * vel[1] - F[1, i] * vel[2])
        flux[i+9] = den * (F[3, i] * vel[1] - F[1, i] * vel[3])
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    E = e_int + e_kin
    flux[13] = den * vel[1] * E - vel[1] * sigma[1, 1] - vel[2] * sigma[1, 2] - vel[3] * sigma[1, 3]
    return flux
end

# Returns the value of the numerical flux between cells using the Lax-Friedrichs method
function LxF(Q_l::Array, Q_r::Array, lambda) 
    return 0.5 * (Flux(Q_l) + Flux(Q_r)) - 0.5 * lambda * (Q_r - Q_l)
end

# Returns the value of quantity in the cell to the next time layer
# @param Q is the quantity in the left, central and right cells
# @param F is the numerical flux method
# @param lambda is the coordinate step to the time step
function UpdateCell(Q::Array, F::Function, lambda)
    Q_l, Q, Q_r = Q[:, 1], Q[:, 2], Q[:, 3]
    F_l = F(Q_l, Q, lambda)
    F_r = F(Q, Q_r, lambda)
    return Q - 1.0 / lambda * (F_r - F_l)
end

# Sets initial conditions to the solution
function InitialCondition(testcase::Int)
    if testcase == 1
        u_l = [0.0, 0.5, 1.0]       # velocity on the left boundary [km/s]
        F_l = [ 0.98  0.0   0.0;    # elastic deformation gradient tensor
                0.02  1.0   0.1;    # on the left boundary
                0.0   0.0   1.0]
        S_l = 1e-3                  # Entropy on the left boundary [kJ/(g*K)]
        
        u_r = [0.0, 0.0, 0.0]       # velocity on the right boundary [km/s]
        F_r = [ 1.0    0.0   0.0;   # elastic deformation gradient tensor
                0.0    1.0   0.1;   # on the right boundary
                0.0    0.0   1.0]
        S_r = 0                     # Entropy on the right boundary [kJ/(g*K)]
    elseif testcase == 2
        u_l = [2.0, 0.0, 0.1] # [km/s]
        F_l = [ 1.0     0.0   0.0 ;
                -0.01   0.95  0.02; 
                -0.015  0.0   0.9 ]
        S_l = 0.0 # [kJ/(g*K)]
        
        u_r = [0.0, -0.03, -0.01] # [km/s]
        F_r = [ 1.0     0.0     0.0;
                0.015   0.95    0.0;
                -0.01   0.0     0.9]
        S_r = 0.0 # [kJ/(g*K)]
    elseif testcase == 3
        u_l = [1.0, 0.0, 0.0] # [km/s]
        F_l = [0.5      -0.5*3^0.5      0.0;
               0.5*3^0.5    0.5         0.0;
               0.0          0.0         1.0]
        S_l = 0.0 # [kJ/(g*K)]
        
        u_r = [1.0, 0.0, 0.0] # [km/s]
        F_r = [0.5       -0.5*3^0.5     0.0;
               0.5*3^0.5    0.5         0.0;
               0.0          0.0         1.0]
        S_r = 0.0 # [kJ/(g*K)]
    else 
        u_l = u_r = zeros(3)
        F_l = F_r = Array{Float64}(I, 3, 3)
        S_l = S_r = 0.0
    end

    Q = Array{Float64}(undef, 13, nx)
    for i in 1:nx
        if (i-1) * dx < 0.5 * X
            # Q[:, i] = BoundaryCondition(u_l, F_l, S_l)
            Q[:, i] = Prim2Cons(u_l, F_l, S_l)
        else
            # Q[:, i] = BoundaryCondition(u_r, F_r, S_r)
            Q[:, i] = Prim2Cons(u_r, F_r, S_r)
        end
    end
    return Q
end

# Reads the data file
function ReadData(filename::String)
    data = readlines(filename)
    t = parse(Float64, data[1])
    nx = length(data) - 1
    Q = Array{Float64}(undef, 13, nx)
    for (line, i) in zip(data[2:end], 1:nx)
        Q[:, i] = parse.(Float64, split(line))
    end
    return t, Q, nx
end

# Initilization of the problem parameters and constants from the config file
function Initialize()
    global nstep = 0             # Initilization of step counter
    global t = 0.0               # Initilization of time
    data = readlines(joinpath(@__DIR__, "init.config")) # Read the config file
    filter!(line -> line != "" && line[1] != '[', data) # Remove empty lines and comments
    # Read the constants
    global rho0 = parse(Float64, split(data[1])[3])
    global c0 = parse(Float64, split(data[2])[3])
    global b0 = parse(Float64, split(data[3])[3])
    global cv = parse(Float64, split(data[4])[3])
    global T0 = parse(Float64, split(data[5])[3])
    global alpha = parse(Float64, split(data[6])[3])
    global beta = parse(Float64, split(data[7])[3])
    global gamma = parse(Float64, split(data[8])[3])
    # Read the parameters
    testcase = parse(Int, split(data[9])[3])
    global log_freq = parse(Int, split(data[10])[3])
    global X = parse(Float64, split(data[11])[3])
    global T = parse(Float64, split(data[12])[3])
    global cu = parse(Float64, split(data[13])[3])
    global nx = parse(Int, split(data[14])[3])

    global dx = X / nx # Coordinate step
    # Creating a 2-dimension solution array
    global Q0 = InitialCondition(testcase) # Apply initial conditions to the solution array
end

Initialize()

cd(@__DIR__)                     # Go to the directory where this file is
ispath("data") || mkpath("data") # Make the data folder if it does not exist
cd("data")                       # Go to the data folder
if ("final.dat" in readdir() || readdir() == [])
    foreach(rm, readdir())       # Remove all files in the folder
else 
    # Find the last file
    last_file = sort(readdir()[1:end-1], by = x -> parse(Int, split(x, ".")[1]))[end]
    global nstep = parse(Int, split(last_file, ".")[1]) # Initilization of step counter
    global t, Q0, nx = ReadData(last_file)       # Read the last file
end 

while t < T
    # Computing the time step using CFL condition
    lambda = 0
    for i in 1:nx
        Q = Q0[:, i]
        local den = Density(Q)
        lambda = max(abs(Q0[1, i]) / den + 5.0, lambda) # TODO: replace 5 with max eigenvalue
    end
    
    global dt = cu * dx / lambda    
    global t += dt                  # Updating the time
    global nstep += 1               # Updating the step counter
    
    # Computing the next time layer
    Q1 = similar(Q0)                
    Q1[:, begin] = Q0[:, begin]
    Q1[:, end] = Q0[:, end]
    for i in 2:nx-1
        Q1[:, i] = UpdateCell(Q0[:, i-1:i+1], LxF, dx/dt)
    end

    global Q0 = copy(Q1)

    if nstep % log_freq == 0
        # Saving the solution array to a file
        local file = open("$(nstep).dat", "w")
        write(file, "$t\n")
        for i in 1:nx
            write(file, join(Q0[:, i], "\t"), "\n")
        end
        close(file)
    end

    # TODO: min rho , max rho, ... (для того, чтобы сразу отсекать неправильные решения) 
    @printf("nstep = %d,\t t = %.6f / %.6f,\t dt = %.6f\n", nstep, t, T, dt)
end

# Saving the final solution array to a file 
file = open("final.dat", "w")
write(file, "$t\t")
for i in 1:nx
    write(file, join(Q0[:, i], "\t"), "\n")
end
close(file)

##### Plotting #####
cd(@__DIR__)
ispath("jl_plots") || mkpath("jl_plots")

x = [dx * i for i in 0:nx-1]
den = Array{Float64}(undef, nx)
entropy = Array{Float64}(undef, nx)
vel = Array{Float64, 2}(undef, 3, nx)
stress = Array{Float64, 3}(undef, 3, 3, nx)
for i in 1:nx
    Q = Q0[:, i]
    den[i], vel[:, i], F, e_int = Cons2Prim(Q)
    local sigma = Stress(den[i], e_int, F)
    for j in 1:3
        for k in 1:3
            stress[j, k, i] = sigma[k, j]
        end
    end
    entropy[i] = Entropy(e_int, Invariants(Finger(F)))
end

plot(x, den)
savefig("jl_plots/density.png")
for i in 1:3
    plot(x, vel[i, :])
    savefig("jl_plots/velocity_$(i).png")
    for j in i:3
        plot(x, stress[i, j, :])
        savefig("jl_plots/stress_$(i)$(j).png")
    end
end
plot(x, entropy)
savefig("jl_plots/entropy.png")

println("Done!")

# MUSTA-type upwind fluxes for non-linear elasticity tests