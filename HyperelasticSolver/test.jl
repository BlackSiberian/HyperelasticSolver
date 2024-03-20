function old_prim2cons(vel::Array{<:Any,1}, F::Array{<:Any,2}, S)
    Q = Array{Float64}(undef, 13)
    den = rho0 / det(F)
    
    Q[1:3] = den .* vel
    for i in 1:3
        for j in 1:3
            Q[3*i + j] = den * F[i, j]
        end
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    G = old_finger(F) 
    # println(transpose(reshape([G...], 3, 3)))
    i = old_invariants(G)
    E = old_energy(S, i) + e_kin
    Q[13] = den * E
    return Q
end

function prim2cons(P::Array{<:Any,1})
    Q = similar(P)
    frac = P[1]
    true_den = P[2]
    den = frac * true_den
    vel = P[3:5]
    entropy = P[6]
    def_grad = P[7:15]
    
    # G = finger(inv(reshape(distortion, (3, 3))))
    # G = finger(distortion)
    G = finger(def_grad)
    # println(reshape(G, 3, 3))
    e_int = energy(entropy, G)
    e_kin = sum(vel.^2) / 2
    e_total = e_int + e_kin

    Q[1] = frac
    Q[2] = den
    Q[3:5] = den * vel
    Q[6] = den * e_total
    Q[7:15] = den * def_grad

    return Q
end
function density(Q::Array{<:Any,1})
    # FQ = [Q[4]  Q[5]  Q[6];     # The part of the vector that
    #       Q[7]  Q[8]  Q[9];     # corresponds to the deformation gradient
    #       Q[10] Q[11] Q[12]]
    rho0 = 8.93
    FQ = [ Q[1]  Q[2]  Q[3];     # The part of the vector that
           Q[4]  Q[5]  Q[6];     # corresponds to the deformation gradient
           Q[7]  Q[8]  Q[9] ]    
    return sqrt(det(FQ) / rho0)
end

function old_cons2prim(Q::Array{<:Any,1})
    FQ = [ Q[4]  Q[5]  Q[6];                              
           Q[7]  Q[8]  Q[9];
           Q[10] Q[11] Q[12]] 
    den = density(Q[4:12])
    vel = Q[1:3] ./ den
    F = FQ ./ den
    E = Q[13] / den
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    e_int = E - e_kin
    return [den, vel, F, e_int]
end

function old_flux(Q::Array{<:Any,1}) 
    den, vel, F, e_int = old_cons2prim(Q)
    sigma = old_stress(den, e_int, F)

    flux = similar(Q)
    for i in 1:3
        flux[i] = den * vel[1] * vel[i] - sigma[1, i]
        flux[i+3] = den * (F[1, i] * vel[1] - F[1, i] * vel[1])
        flux[i+6] = den * (F[2, i] * vel[1] - F[1, i] * vel[2])
        flux[i+9] = den * (F[3, i] * vel[1] - F[1, i] * vel[3])
    end
    e_kin = 0.5 * (vel[1]^2 + vel[2]^2 + vel[3]^2)
    E = e_int + e_kin
    flux[13] = den * vel[1] * E #- vel[1] * sigma[1, 1] - vel[2] * sigma[1, 2] - vel[3] * sigma[1, 3]
    return flux
end

function flux(Q::Array{<:Any,1})
    frac = Q[1]
    den = Q[2]
    true_den = den / frac
    vel = Q[3:5] / den
    e_total = Q[6] / den
    e_kin = sum(vel.^2) / 2
    e_int = e_total - e_kin
    def_grad = Q[7:15] / den
    
    # strs = stress(eos, true_den, e_int, distortion)
    strs = stress(true_den, e_int, def_grad)

    flux = similar(Q)

    flux[1] = 0
    flux[2] = den * vel[1]
    flux[3:5] = den * vel * vel[1] - strs[1:3]
    flux[6] = den * vel[1] * e_total #- sum(vel .* strs[1:3])
    flux[7:15] = den * (def_grad * vel[1] - reshape(def_grad[1:3] * transpose(vel), 9))
    
    return flux
end

### Strains ###

old_finger(f::Array{<:Any,2})::Array{<:Any,2} = inv(f * transpose(f))

function old_finger(f::Array{<:Any,1})::Array{<:Any,2}
    f2 = [ f[1] f[2] f[3];
           f[4] f[5] f[6];
           f[7] f[8] f[9] ]
    return inv(f2 * transpose(f2))
end

function old_invariants(g::Array{<:Any,2})
    i1 = tr(g)
    i2 = 0.5 * (tr(g)^2 - tr(g^2))
    i3 = det(g)
    return [i1, i2, i3]
end

function old_invariants(g::Array{<:Any,1})
    g2 = [ g[1] g[2] g[3];                    # This isn't the natural
           g[4] g[5] g[6];                    # order in Julia!
           g[7] g[7] g[9] ]
    i1 = tr(g2)
    i2 = 0.5 * (tr(g2)^2 - tr(g2^2))
    i3 = det(g2)
    return [i1, i2, i3]
end

di1dg(G, i1, i2, i3) = I            # Returns the derivative of I with respect to G
di2dg(G, i1, i2, i3) = i1 .* I - G  # @param G is the symmetric tensor
di3dg(G, i1, i2, I3) = I3 .* inv(G) # @param I1, I2, I3 are the invariants


function finger(a::Array{<:Any,1})::Array{<:Any,1}
    # a = inv(reshape(a, (3, 3)))
    # g = a * transpose(a)

    a = reshape(a, (3, 3))
    g = inv(a * transpose(a))
    
    g = reshape(g, length(g))
    return g
end

function invariants(g::Array{<:Any,1})::Array{<:Any,1}
    g = reshape(g, (3, 3))
    i1 = tr(g)
    i2 = 0.5 * (tr(g)^2 - tr(g^2))
    i3 = det(g)
    return [i1, i2, i3]
end

### Stress
function denergy(e_int, i::Array{<:Any,1})
    rho0 = 8.93 # Initial density [g/cm^3]
    c0 = 4.6    # Speed of sound [km/s]
    cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
    t0 = 300    # Initial temperature [K]
    b0 = 2.1    # Speed of the shear wave [km/s]
    alpha = 1.0 # Non-linear
    beta = 3.0  # characteristic
    gamma = 2.0 # constants
    
    # Secondaty parameters
    b0sq = b0^2              # Formely B0
    k0 = c0^2 - (4/3)*b0^2
    
    # TODO: Probably implement automatic differentiation
    e1 = b0sq * i[1] * i[3]^(0.5*beta) / 3.0
    e2 = - 0.5 * b0sq * i[3]^(0.5*beta)
    e3 = 0.5 * k0 / alpha * (i[3]^(0.5*alpha) - 1) * i[3]^(0.5*alpha-1) + 0.25 * beta * b0sq * (i[1]^2 / 3 - i[2]) * i[3]^(0.5*beta-1)
    e3 += 0.5 * gamma * (e_int - 0.5 * b0sq * i[3]^(0.5*beta) * (i[1]^2 / 3 - i[2]) - 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2) / i[3]

    return [e1, e2, e3]
end

function old_stress(den, e_int, F::Array{<:Any,2})
    G = old_finger(F) 
    I1, I2, I3 = old_invariants(G)
    G = old_finger(F)

    e1, e2, e3 = denergy(e_int, [I1, I2, I3])

    return -2.0 * den .* G * (e1 .* di1dg(G, I1, I2, I3) + e2 .* di2dg(G, I1, I2, I3) + e3 .* di3dg(G, I1, I2, I3))
end

using ForwardDiff: gradient

function entropy(e_int, G::Array{<:Any,1})
    rho0 = 8.93 # Initial density [g/cm^3]
    c0 = 4.6    # Speed of sound [km/s]
    cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
    t0 = 300    # Initial temperature [K]
    b0 = 2.1    # Speed of the shear wave [km/s]
    alpha = 1.0 # Non-linear
    beta = 3.0  # characteristic
    gamma = 2.0 # constants
    
    # Secondaty parameters
    b0sq = b0^2              # Formely B0
    k0 = c0^2 - (4/3)*b0^2

    i = invariants(G)

    S = e_int - 0.5 * b0sq * i[3]^(0.5*beta) * (i[1]^2 / 3 - i[2]) - 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2
    S = (S / (cv * t0 * i[3]^(0.5*gamma)) + 1)
    return log(S) * cv
end

function stress(den, e_int, F::Array{<:Any,1})::Array{<:Any,1}
    G = finger(F)
    S = entropy(e_int, G)
    # e(G::Array) = energy(eos, entropy(eos, e_int, G), G)
    e(G::Array) = energy(S, G)

    dedG = gradient(e, G)
    
    G = reshape(G, (3, 3))
    dedG = reshape(dedG, (3, 3))

    stress = - 2 * den .* G * dedG
    return reshape(stress, length(stress)) 
end

### Energy ###

function old_energy(S, i::Array{<:Any,1})
    rho0 = 8.93 # Initial density [g/cm^3]
    c0 = 4.6    # Speed of sound [km/s]
    cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
    t0 = 300    # Initial temperature [K]
    b0 = 2.1    # Speed of the shear wave [km/s]
    alpha = 1.0 # Non-linear
    beta = 3.0  # characteristic
    gamma = 2.0 # constants
    
    # Secondaty parameters
    b0sq = b0^2              # Formely B0
    k0 = c0^2 - (4/3)*b0^2
    
    U = ( 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1.0)^2 
          + cv * t0 * i[3]^(0.5*gamma) * (exp(S / cv) - 1.0)
          )
    W = 0.5 * b0sq * i[3]^(0.5*beta)*(i[1]^2 / 3.0 - i[2])
    e_int = U + W
    return e_int
end

function energy(S, G::Array{<:Any,1})
    rho0 = 8.93 # Initial density [g/cm^3]
    c0 = 4.6    # Speed of sound [km/s]
    cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
    t0 = 300    # Initial temperature [K]
    b0 = 2.1    # Speed of the shear wave [km/s]
    alpha = 1.0 # Non-linear
    beta = 3.0  # characteristic
    gamma = 2.0 # constants
    
    # Secondaty parameters
    b0sq = b0^2              # Formely B0
    k0 = c0^2 - (4/3)*b0^2

    i = invariants(G)

    U = ( 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1.0)^2 
          + cv * t0 * i[3]^(0.5*gamma) * (exp(S / cv) - 1.0)
          )
    
    W = 0.5 * b0sq * i[3]^(0.5*beta)*(i[1]^2 / 3.0 - i[2])
    e_int = U + W
    return e_int
end

using LinearAlgebra

### Testing ###

# rho0 = 8.93
# u = [0.0, 0.5, 1.0]       
# F = [ 0.98  0.0   0.0;    
#       0.02  1.0   0.1;    
#       0.0   0.0   1.0]
# S = 1e-3                  

rho0 = 8.93
u = [2.0, 0.0, 0.1]
F_o = [1.0 0.0 0.0;
-0.01 0.95 0.02;
-0.015 0.0 0.0]
# S = 0.0

# u = [0.0, -0.03, -0.01]
# F_o = [1.0 0.0 0.0;
# 0.015 0.95 0;
# -0.01 0.0 0.9]
F = transpose(F_o)
S = 0.0

alpha = 1.0
rho = rho0 / det(F)

Q = prim2cons([alpha, rho, u..., S, F...])
Q_old = old_prim2cons(u, F_o, S)

println("Comparing Q before flux:\n New: ", Q[3:15], "\n Old: ", Q_old)
# println("---")
# println(Q[3:5])
# println(Q_old[1:3])
# println("---")
# println(Q[6])
# println(Q_old[end])
# println("---")
# println(Q[7:15])
# println(Q_old[4:12])
# println("Done!")

Fl = flux(Q)
Fl_old = old_flux(Q_old)

println("Comparing fluxes:")
println("--- Velocity")
println(Fl[3:5])
println(Fl_old[1:3])
println("--- Energy")
println(Fl[6])
println(Fl_old[end])
println("--- Deformation gradient")
println(Fl[7:15])
println(Fl_old[4:12])
println("Done!")
