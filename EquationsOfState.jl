#
# EquationsOfState.jl
#
# Equation of state definitions and functions.

module EquationsOfState

using LinearAlgebra: det, inv, tr
using SpecialFunctions: expinti
using ForwardDiff: derivative, gradient, jacobian

using ..Strains

export energy, entropy, stress, acoustic, density, temperature
export EoS, Barton2009, Stiffened, Hayes
export density_GRP


# ##############################################################################
# Параметризация уравнения состояния производится типом.
# В более общем случае --- callable structs или HolyTraits
# See:
#    https://www.ahsmart.com/pub/holy-traits-design-patterns-and-best-practice-book
#    https://discourse.julialang.org/t/function-factories-or-callable-structs/52987
#    https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects
#
# ##############################################################################

# Abstract EoS type
"""
    EoS

Abstract type for equations of state.
"""
abstract type EoS end

# All EoS types has to provide the following methods:
"""
    energy(eos::EoS, S::Real, G::AbstractVector{<:Real})

Compute specific internal energy.

# Arguments
- `eos::EoS`: equation of state
- `S::Real`: entropy
- `G::AbstractVector{<:Real}`: flattened strain tensor
"""
energy(eos::EoS, S::Real, G::AbstractVector{<:Real}) = error("energy() isn't implemented for EoS: ", typeof(eos))

"""
    entropy(eos::EoS, e_int::Real, G::AbstractVector{<:Real})

Compute entropy from internal energy and strain.

# Arguments
- `eos::EoS`: equation of state
- `e_int::Real`: specific internal energy
- `G::AbstractVector{<:Real}`: flattened strain tensor
"""
entropy(eos::EoS, e_int::Real, G::AbstractVector{<:Real}) = error("entropy() isn't implemented for EoS: ", typeof(eos))

"""
    stress(eos::EoS, ent::Real, F::AbstractVector{<:Real}) -> AbstractVector{<:Real}

Compute stress tensor.

# Arguments
- `eos::EoS`: equation of state
- `ent::Real`: entropy
- `F::AbstractVector{<:Real}`: flattened deformation gradient tensor
"""
stress(eos::EoS, ent::Real, F::AbstractVector{<:Real}) = error("stress() isn't implemented for EoS: ", typeof(eos))

"""
    acoustic(eos::EoS, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real}) -> AbstractMatrix{<:Real}

Compute acoustic tensor.

# Arguments
- `eos::EoS`: equation of state
- `ent::Real`: entropy
- `F::AbstractVector{<:Real}`: flattened deformation gradient tensor
- `n::AbstractVector{<:Real}`: normal vector
"""
acoustic(eos::EoS, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = error("acoustic() isn't implemented for EoS: ", typeof(eos))

"""
    density(eos::EoS, F::AbstractVector{<:Real})

Compute density.

# Arguments
- `eos::EoS`: equation of state
- `F::AbstractVector{<:Real}`: flattened deformation gradient tensor
"""
density(eos::EoS, F::AbstractVector{<:Real}) = error("density() isn't implemented for EoS: ", typeof(eos))

"""
    temperature(eos::EoS, ent::Real, F::AbstractVector{<:Real})

Compute temperature.

# Arguments
- `eos::EoS`: equation of state
- `ent::Real`: entropy
- `F::AbstractVector{<:Real}`: flattened deformation gradient tensor
"""
temperature(eos::EoS, ent::Real, F::AbstractVector{<:Real}) = error("temperature() isn't implemented for EoS: ", typeof(eos))

# ##############################################################################
# Finger (or deformation gradient) strain tensor
# ##############################################################################

function stress_grdef(eos::EoS, ent::Real, F::AbstractVector{<:Real})
    G = finger(F)

    dedG = gradient(G -> energy(eos, ent, G), G)

    G = reshape(G, (3, 3))
    dedG = reshape(dedG, (3, 3))

    stress = -2 * density_grdef(eos, F) .* G * dedG
    return reshape(stress, length(stress))
end

function acoustic_grdef(eos::EoS, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real})
    acoustic = zeros(3, 3)
    dTdF = reshape(jacobian(F -> stress_grdef(eos, ent, F), F), (3, 3, 3, 3))
    A = (1 / density_grdef(eos, F)) .* dTdF
    F = reshape(F, (3, 3))

    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l = 1:3
                    for m = 1:3
                        acoustic[i, j] += A[m, i, j, l] * F[k, l] * n[m] * n[k]
                    end
                end
            end
        end
    end
    # acoustic = dropdims(sum(A .* reshape(n * (F' * n)', (3,1,1,3)), dims=(1,4)), dims=(1,4))

    return acoustic
end

density_grdef(eos::EoS, F::AbstractVector{<:Real}) = eos.rho0 / det(reshape(F, 3, 3))

density_finger(eos::EoS, G::AbstractVector{<:Real}) = eos.rho0 * sqrt(det(reshape(G, 3, 3)))

temperature_finger(eos::EoS, ent::Real, G::AbstractVector{<:Real}) = derivative(S -> energy(eos, S, G), ent)


# ##############################################################################
# Model specific functions
# ##############################################################################

"""
    density_GRP(eos::EoS, FQ::AbstractVector{<:Real})

Compute density from conservative variables for GRP model.

# Arguments
- eos::EoS: equation of state
- FQ::AbstractVector{<:Real}): part of conservative variables vector related to strain tensor
"""
density_GRP(eos::EoS, FQ::AbstractVector{<:Real}) = sqrt(det(reshape(FQ, (3, 3)) / eos.rho0)

# ##############################################################################
# Barton2009
# ##############################################################################

"""
    Barton2009 <: EoS

Barton2009 equation of state, a hyperelastic material model based on the paper by Barton et al. (2009).

# Constructor
    Barton2009(; rho0=8.93, c0=4.6, cv=3.9e-4, t0=300, b0=2.1, alpha=1, beta=3, gamma=2)

# Arguments
- `rho0`: Initial density [g/cm³]
- `c0`: Speed of sound [km/s]
- `cv`: Heat capacity [kJ/(g·K)]
- `t0`: Initial temperature [K]
- `b0`: Speed of the shear wave [km/s]
- `alpha`: Non-linear parameter for volumetric energy term
- `beta`: Non-linear parameter for shear energy term
- `gamma`: Non-linear parameter for thermal energy term

See also: [`energy`](@ref), [`stress`](@ref), [`entropy`](@ref)
"""
@kwdef struct Barton2009 <: EoS
    rho0::Real = 8.93
    c0::Real = 4.6
    cv::Real = 3.9e-4
    t0::Real = 300
    b0::Real = 2.1
    alpha::Real = 1
    beta::Real = 3
    gamma::Real = 2

    # Derived parameters
    b0sq::Real = b0^2
    k0::Real = c0^2 - (4 / 3) * b0^2
end # struct Barton2009 <: EoS

function energy(eos::Barton2009, S::Real, G::AbstractVector{<:Real})
    i = invariants(G)

    U = 0.5 * eos.k0 / (eos.alpha^2) * (i[3]^(0.5 * eos.alpha) - 1.0)^2 +
        eos.cv * eos.t0 * i[3]^(0.5 * eos.gamma) * (exp(S / eos.cv) - 1.0)

    W = 0.5 * eos.b0sq * i[3]^(0.5 * eos.beta) * (i[1]^2 / 3.0 - i[2])

    return U + W
end

function entropy(eos::Barton2009, e_int::Real, G::AbstractVector{<:Real})
    i = invariants(G)

    S = e_int -
        0.5 * eos.b0sq * i[3]^(0.5 * eos.beta) * (i[1]^2 / 3 - i[2]) -
        0.5 * eos.k0 / (eos.alpha^2) * (i[3]^(0.5 * eos.alpha) - 1)^2
    S = S / (eos.cv * eos.t0 * i[3]^(0.5 * eos.gamma)) + 1

    # NOTE: Using entropy fix
    S = S < 1e-13 ? 1e-13 : S

    return log(S) * eos.cv
end

stress(eos::Barton2009, ent::Real, F::AbstractVector{<:Real}) = stress_grdef(eos, ent, F)
acoustic(eos::Barton2009, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = acoustic_grdef(eos, ent, F, n)
density(eos::Barton2009, F::AbstractVector{<:Real}) = density_grdef(eos, F)
temperature(eos::Barton2009, ent::Real, G::AbstractVector{<:Real}) = temperature_finger(eos, ent, G)

# ##############################################################################
# Stiffened
# ##############################################################################

"""
    Stiffened <: EoS

Stiffened equation of state, commonly used for materials like water or other liquids under high pressure conditions.

# Constructor
    Stiffened(; rho0=2.78, s=1.338, c0=5.33, cv=9.3e-4, mu=27.6, T0=300, G0=2, S0=1e-3)

# Arguments
- `rho0`: Initial density [g/cm³]
- `s`: Slope of the linear Us-Up relation
- `c0`: Speed of sound in the material [km/s]
- `cv`: Heat capacity [kJ/(g·K)]
- `mu`: Shear elastic modulus
- `T0`: Reference temperature [K]
- `G0`: Gruneisen parameter
- `S0`: Reference entropy

See also: [`energy`](@ref), [`stress`](@ref), [`entropy`](@ref)
"""
@kwdef struct Stiffened <: EoS
    rho0::Real = 2.78
    s::Real = 1.338
    c0::Real = 5.33
    cv::Real = 9.3e-4
    mu::Real = 27.6
    T0::Real = 300
    G0::Real = 2
    S0::Real = 1e-3
end # struct Stiffened <: EoS

function energy(eos::Stiffened, S::Real, G::AbstractVector{<:Real})
    i = invariants(G)
    i[2] = tr(reshape(G, (3, 3))^2)

    rho = density_finger(eos, G)
    nu = eos.rho0 / rho
    cs = sqrt(eos.mu / rho)

    e_ref = 1/2 * eos.c0^2 * (1 - nu)^2 / (1 - eos.s * (1 - nu))^2
    T_ref = 1 / (2eos.cv * eos.s^4) * (
        eos.s * (-eos.c0^2 * (eos.G0 - 3eos.s) + 2eos.cv * eos.s^3 * eos.T0) * exp(eos.G0 * (1 - nu)) +
        (eos.c0^2 * eos.s * ((eos.G0 - 4 * eos.s) * eos.s * nu + eos.G0 - (3 + eos.G0) * eos.s + 4eos.s^2)) /
        (eos.s * (nu - 1) + 1)^2 +
        eos.c0^2 * exp(eos.G0 * (1 - (1 / eos.s + nu))) * (eos.G0^2 - 4eos.G0 * eos.s + 2eos.s^2) * (
            expinti(eos.G0 / eos.s) - expinti(eos.G0 * (1 / eos.s + nu - 1))
        )
    )
    T = eos.T0 * exp((S - eos.S0) / eos.cv - eos.G0 * (nu - 1))

    e_int = e_ref + eos.cv * (T - T_ref)
    e_int += cs^2 / 4 * (i[2] - 1/3 * i[1]^2)

    return e_int
end


function entropy(eos::Stiffened, e_int::Real, G::AbstractVector{<:Real})
    i = invariants(G)
    i[2] = tr(reshape(G, (3, 3))^2)

    rho = density_finger(eos, G)
    nu = eos.rho0 / rho
    cs = sqrt(eos.mu / rho)

    e_ref = 1/2 * eos.c0^2 * (1 - nu)^2 / (1 - eos.s * (1 - nu))^2
    T_ref = 1 / (2eos.cv * eos.s^4) * (
        eos.s * (-eos.c0^2 * (eos.G0 - 3eos.s) + 2eos.cv * eos.s^3 * eos.T0) * exp(eos.G0 * (1 - nu)) +
        (eos.c0^2 * eos.s * ((eos.G0 - 4eos.s) * eos.s * nu + eos.G0 - (3 + eos.G0) * eos.s + 4 * eos.s^2)) /
        (s * (nu - 1) + 1)^2 +
        eos.c0^2 * exp(eos.G0 * (1 - 1 / eos.s - nu)) * (eos.G0^2 - 4eos.G0 * eos.s + 2s^2) * (
            expinti(eos.G0 / eos.s) - expinti(eos.G0 * (1 / eos.s + nu - 1))
        )
    )

    e_int -= cs^2 / 4 * (i[2] - 1/3 * i[1]^2)

    T = (T_ref + (e_int - e_ref) / eos.cv) / eos.T0

    # NOTE: Using entropy fix
    T = T < 1e-13 ? 1e-13 : T

    ent = eos.S0 + eos.cv * (log(T) + eos.G0 * (nu - 1))

    return ent
end

stress(eos::Stiffened, ent::Real, F::AbstractVector{<:Real}) = stress_grdef(eos, ent, F)
acoustic(eos::Stiffened, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = acoustic_grdef(eos, ent, F, n)
density(eos::Stiffened, F::AbstractVector{<:Real}) = density_grdef(eos, F)
temperature(eos::Stiffened, ent::Real, G::AbstractVector{<:Real}) = temperature_finger(eos, ent, G)

# ##############################################################################
# Hayes
# ##############################################################################

"""
    Hayes <: EoS

Hayes equation of state, a model commonly used for certain types of materials under extreme conditions.

# Constructor
    Hayes(; rho0=880.0, s=2.17, c0=1570, cv=1900, mu=2.2e6, T0=300, G0=0.7, S0=1e-6, P0=1e5, e0=1e-6)

# Arguments
- `rho0`: Initial density [g/cm³]
- `s`: Parameter related to the linear Us-Up relation
- `c0`: Speed of sound in the material [km/s]
- `cv`: Heat capacity [kJ/(g·K)]
- `mu`: Shear elastic modulus
- `T0`: Reference temperature [K]
- `G0`: Gruneisen parameter
- `S0`: Reference entropy
- `P0`: Reference pressure
- `e0`: Reference energy

See also: [`energy`](@ref), [`stress`](@ref), [`entropy`](@ref)
"""
@kwdef struct Hayes <: EoS
    rho0::Real = 880.0
    s::Real = 2.17
    c0::Real = 1570
    cv::Real = 1900
    mu::Real = 2.2e6
    T0::Real = 300
    G0::Real = 0.7
    S0::Real = 1e-6
    P0::Real = 1e5
    e0::Real = 1e-6
end # struct Hayes <: EoS

function energy(eos::Hayes, S::Real, G::AbstractVector{<:Real})
    i = invariants(G)

    rho = density_finger(eos, G)
    V = 1 / rho
    V0 = 1 / eos.rho0
    eta = V / V0
    expS = exp((S - eos.S0)/eos.cv - eos.G0 * (eta - 1))

    K0 = eos.rho0 * (eos.c0^2 - eos.G0^2 * eos.cv * eos.T0)
    N = (4eos.s - 1) + (4eos.s - eos.G0) * (eos.G0^2 * eos.cv * eos.T0 / (V0 * K0))

    U = eos.e0 + (S - eos.S0) * eos.T0 * expS +
        eos.cv * eos.T0 * (eos.G0 * (1 - eta) + 1) * (expS - 1) -
        eos.cv * eos.T0 * expS * ((S - eos.S0) / eos.cv - eos.G0 * (eta - 1)) +
        K0 * V0 * (eta^-(N-1) - (N - 1) * (1 - eta) - 1) / ((N - 1) * N) +
        eos.P0 * (V0 - V)

    i[2] = -2i[2] + i[1]^2
    W = eos.mu / 4eos.rho0 * (i[2] / cbrt(i[3])^2 - 2i[1] / cbrt(i[3]) + 3)
    return U + W
end

function entropy(eos::Hayes, e_int::Real, G::AbstractVector{<:Real})
    i = invariants(G)
    i[2] = -2i[2] + i[1]^2

    W = eos.mu / 4eos.rho0 * (i[2] / cbrt(i[3])^2 - 2i[1]/cbrt(i[3]) + 3)

    rho = density_finger(eos, G)
    V = 1 / rho
    V0 = 1 / eos.rho0
    eta = V / V0
    K0 = eos.rho0 * (eos.c0^2 - eos.G0^2 * eos.cv * eos.T0)
    N = (4eos.s - 1) + (4eos.s - eos.G0) * (eos.G0^2 * eos.cv * eos.T0 / (V0 * K0))

    U = e_int - W

    P = eos.P0 + K0 / N * (
            (eta^-N - 1)
            - eos.G0 / (N-1) * (eta^-(N-1) - 1)
            + eos.G0 * (1 - eta)
        ) +
        eos.G0 / V0 * (
            U - eos.e0
            + eos.cv * eos.T0 * eos.G0 * (1 - eta)
            - eos.P0*(V0 - V)
        )

    T = eos.T0 + V0 / (eos.cv * eos.G0) * (P - eos.P0 - K0 / N * (eta^-N - 1))
    T = T / eos.T0

    # NOTE: Using entropy fix
    T = T < 1e-13 ? 1e-13 : T

    return eos.cv * log(T) + eos.S0 + eos.cv * (V - V0) * eos.G0 / V0
end

stress(eos::Hayes, ent::Real, F::AbstractVector{<:Real}) = stress_grdef(eos, ent, F)
acoustic(eos::Hayes, ent::Real, F::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = acoustic_grdef(eos, ent, F, n)
density(eos::Hayes, F::AbstractVector{<:Real}) = density_grdef(eos, F)
temperature(eos::Hayes, ent::Real, G::AbstractVector{<:Real}) = temperature_finger(eos, ent, G)

# ##############################################################################
# Since there are only few equation of of states and material (~10) supposed
# to be used, ---  define the corresponding EoS functions here just once.

# eos_barton2009 = Barton2009()
# energy(S,i)                  = energy(eos_barton2009,S,i)
# entropy(e_int, i)            = entropy(eos_barton2009,e_int, i)
# denergy(e_int, i)            = denergy(eos_barton2009, e_int, i)
# density(Q::Array)            = density(eos_barton2009, Q)
# stress(den, e_int, F::Array) = stress(eos_barton2009, den, e_int, F::Array)


# Для других материалов --- инициализируем тип другим набором констант,
# нужно дописать конструктор --- как в типе, только со списком аргументов.
#     eos_barton_2009_fe = Barton2009(...)
#
#     energy(S,i)                  = energy(eos_barton2009_fe,S,i)
#     entropy(e_int, i)            = entropy(eos_barton2009_fe,e_int, i)
#     denergy(e_int, i)            = denergy(eos_barton2009_fe, e_int, i)
#     density(Q::Array)            = density(eos_barton2009_fe, Q)
#     stress(den, e_int, F::Array) = stress(eos_barton2009_fe, den, e_int, F::Array)
#
#
#     Чтобы не возникало путаницы (какому материалу соответствует
#     конкретный Barton2009) --- добавить в тип поле с названием материала.
#
#     Это плохое решение --- например, нельзя выбрать все варианты
#     УрС, какие есть для меди, например, --- но пока так.
#     Более правильно параметризовать еще одним типом для материала.

# Для многофазной задачи --- сразу писать через частичное вычисление
# в массив УрС для фаз.
#

end # module EoS
# EOF
