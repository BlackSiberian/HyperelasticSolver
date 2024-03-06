#
# EquationsOfState.jl
#
# Equation of state definitions and functions.

module EquationsOfState

using LinearAlgebra: det, inv
using ForwardDiff: gradient
using ..Strains: finger, invariants#, di1dg, di2dg, di3dg

export energy, entropy, stress, Barton2009, EoS


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
abstract type EoS end

# All EoS types has to provide the following methods:
energy(eos::T, S, G::Array{<:Any,1})      where {T <: EoS} = error("energy() isn't implemented for EoS: ", typeof(eos))
entropy(eos::T, e_int, G::Array{<:Any,1}) where {T <: EoS} = error("entropy() isn't implemented for EoS: ", typeof(eos))
stress(eos::T, e_int, F::Array{<:Any,1})  where {T <: EoS} = error("stress() isn't implemented for EoS: ", typeof(eos))
density(eos::T, Q::Array{<:Any,1})        where {T <: EoS} = error("density() isn't implemented for EoS: ", typeof(eos))
    
# ##############################################################################
# Barton2009
# ##############################################################################

"""
    Barton2009 EoS.
    See paper for parameters description.
"""
struct Barton2009 <: EoS
    # Primary parameters
    rho0    # Initial density [g/cm^3]
    c0      # Speed of sound [km/s]
    cv      # Heat capacity [kJ/(g*K)]
    t0      # Initial temperature [K]
    b0      # Speed of the shear wave [km/s]
    alpha   # Non-linear
    beta    # characteristic
    gamma   # constants

    # Secondary parameters
    b0sq                      # Formerly B0
    k0                        #

    # Default constructor
    # TODO: Implement specific constructors, see
    #       https://discourse.julialang.org/t/automatic-keyword-argument-constructor/36573
    #       to define only keyword arguments
    function Barton2009()
        # Primary parameters
        rho0 = 8.93 # Initial density [g/cm^3]
        c0 = 4.6    # Speed of sound [km/s]
        cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
        t0 = 300    # Initial temperature [K]
        b0 = 2.1    # Speed of the shear wave [km/s]
        alpha = 1.0 # Non-linear
        beta = 3.0  # characteristic
        gamma = 2.0 # constants
        
        # Secondary parameters
        b0sq = b0^2              # Formerly B0
        k0 = c0^2 - (4/3)*b0^2
        
        return new(rho0, c0, cv, t0, b0, alpha, beta, gamma, b0sq, k0)
    end
end # struct Barton2009 <: EoS

"""
    energy(eos::Barton2009, S, G::Array{<:Any,1})

Returns the value of the internal energy for Barton2009.

# Arguments
- `eos::Barton2009`: the equation of state using in the model  
- `S`: an entropy
- `G`: a Finger tensor
"""
function energy(eos::Barton2009, S, G::Array{<:Any,1})
    b0sq  = eos.b0sq 
    k0    = eos.k0
    alpha = eos.alpha
    beta  = eos.beta
    gamma = eos.gamma
    cv    = eos.cv
    t0    = eos.t0

    i = invariants(G)

    U = ( 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1.0)^2 
          + cv * t0 * i[3]^(0.5*gamma) * (exp(S / cv) - 1.0)
          )
    
    W = 0.5 * b0sq * i[3]^(0.5*beta)*(i[1]^2 / 3.0 - i[2])
    e_int = U + W
    return e_int
end

"""
    entropy(eos::Barton2009, e_int, G::Array{<:Any,1})

Returns the value of the internal energy for Barton2009.

# Arguments
- `eos::Barton2009`: the equation of state using in the model  
- `e_int`: an internal energy
- `G`: a Finger tensor
"""
function entropy(eos::Barton2009, e_int, G::Array{<:Any,1})
    b0sq  = eos.b0sq 
    k0    = eos.k0
    alpha = eos.alpha
    beta  = eos.beta
    gamma = eos.gamma
    cv    = eos.cv
    t0    = eos.t0

    i = invariants(G)

    S = e_int - 0.5 * b0sq * i[3]^(0.5*beta) * (i[1]^2 / 3 - i[2]) - 0.5 * k0 / (alpha^2) * (i[3]^(0.5*alpha) - 1)^2
    S = (S / (cv * t0 * i[3]^(0.5*gamma)) + 1)
    return log(S) * cv
end

"""
    Returns stress tensor.
    TODO: Remove computations if invariants from here, 
          pass only precomputed invariants.
    TODO: Pass only F, since den can be extracted form EoS type
"""
function stress(eos::Barton2009, den, e_int, F::Array{<:Any,1})::Array{<:Any,1}
    G = finger(F)
    S = entropy(eos, e_int, G)
    # e(G::Array) = energy(eos, entropy(eos, e_int, G), G)
e(G::Array) = energy(eos, S, G)

    dedG = gradient(e, G)
    
    G = reshape(G, (3, 3))
    dedG = reshape(dedG, (3, 3))

    stress = - 2 * den .* G * dedG
    return reshape(stress, length(stress)) 
end


# Здесь Q --- одномерный массив.
"""
    Returns density computed from conservative variables for GRP model.
    Actual input is \$\rho\tn{F}\$.
    TODO: Make Finer type and the function 
          to accept only Finger tenors and not others!
"""
function density(eos::Barton2009, Q::Array{<:Any,1})
    rho0 = eos.rho0

    FQ = reshape(Q[1:9], (3, 3))
    return sqrt(det(FQ) / rho0)
end


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

"""
    Hank2016 EoS
    See paper for parameters description.
"""
struct Hank2016 <: EoS 
    # Primary parameters
    rho0        # Initial density [g/cm^3]
    mu          # Shear modulus [Pa]
    gamma       # Characteristic
    pres_inf    # constants
    a           # EoS parameter

    function Hank2016()
        rho0 = 2.7 # Initial density [g/cm^3]
        mu = 26e9  # Shear modulus [Pa]
        gamma = 3.4 
        pres_inf = 21.5e9
        a = 0.5        
        return new(rho0, mu, gamma, pres_inf, a)
    end
end # struct Hank2016 <: EoS

"""
    Returns the value of the internal energy
    @param eos::Hank2016 is EoS parameter type 
    @param S is entropy
    @param i is invariants
"""
function energy(eos::Hank2016, den, pres, G::Array{<:Any,2})
    rho0 = eos.rho0
    mu = eos.mu
    gamma = eos.gamma
    a = eos.a
    pres_inf = eos.pres_inf

    i = invariants(G)

    j = [i[1] / i[3]^(1/3), (i[1]^2 - 2 * i[2]) / i[3]^(2/3)]

    e_el = mu / (4 * rho0) * ((1 - 2 * a) / 3 * j[1]^2 + a * j[2] + 3 * (a - 1)) 
    e_h = (pres + gamma * pres_inf) / (den * (gamma - 1)) 
    e_int = e_el + e_h
    return e_int
end

function pressure(eos::Hank2016, den, e_int, i::Array{<:Any,1})
    rho0 = eos.rho0
    mu = eos.mu
    gamma = eos.gamma
    a = eos.a
    pres_inf = eos.pres_inf

    j = [i[1] / i[3]^(1/3), (i[1]^2 - 2 * i[2]) / i[3]^(2/3)]

    e_el = mu / (4 * rho0) * ((1 - 2 * a) / 3 * j[1]^2 + a * j[2] + 3 * (a - 1)) 
    e_h = e_int - e_el
    pres = e_h * (gamma - 1) * den - gamma * pres_inf
    return pres
end

function stress(eos::Hank2016, den, pressure, distortion::Array{<:Any, 1})::Array{<:Any, 1}
    G = finger(inv(reshape(distortion, (3, 3))))
    
    e(G::Array) = energy(eos, den, pressure, G)
    dedG = reshape(ForwardDiff.gradient(e, G), (3, 3))
    stress = -2.0 * den .* G * dedG
    return reshape(stress, length(stress)) 
end

eos_hank2016 = Hank2016()

export Hank2016, eos_hank2016, pressure

end # module EoS
# EOF