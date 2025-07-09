#
# EquationsOfState.jl
#
# Equation of state definitions and functions.

module EquationsOfState

using LinearAlgebra: det, inv, tr
using SpecialFunctions: expinti
using ForwardDiff: derivative, gradient, jacobian
using ..Strains: finger, invariants#, di1dg, di2dg, di3dg

export energy, entropy, stress, Barton2009, Stiffened, EoS, acoustic


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
"""
    energy(eos::T, S, G::Array{<:Any,1}) where {T <: EoS}

Computes the value of the internal energy for `eos` equation of state

- `S` : an entropy
- `G` : a Finger's tensor
"""
energy(eos::T, S, G::Array{<:Any,1}) where {T<:EoS} = error("energy() isn't implemented for EoS: ", typeof(eos))

"""
    entropy(eos::eos, e_int, G::Array{<:Any,1}) where {T <: EoS}

Computes the value of the internal energy for `eos` equation of state
 
- `e_int` : an internal energy
- `G` : a Finger's tensor
"""
entropy(eos::T, e_int, G::Array{<:Any,1}) where {T<:EoS} = error("entropy() isn't implemented for EoS: ", typeof(eos))

"""
    stress(eos::T, e_int, F::Array{<:Any,1}) where {T <: EoS}
    
Computes stress tensor for `eos` equation of state.

- `den` : a density
- `e_int` : an internal energy
- `F` : a gradient deformations tensor
"""
stress(eos::T, den, e_int, F::Array{<:Any,1}) where {T<:EoS} = error("stress() isn't implemented for EoS: ", typeof(eos))

# Deprecated function
density(eos::T, Q::Array{<:Any,1}) where {T<:EoS} = error("density() isn't implemented for EoS: ", typeof(eos))

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
  function Barton2009(; _rho0=8.93, _c0=4.6, _cv=3.9e-4, _t0=300, _b0=2.1, _alpha=1, _beta=3, _gamma=2)
    # Primary parameters
    # rho0 = 8.93 # Initial density [g/cm^3]
    # c0 = 4.6    # Speed of sound [km/s]
    # cv = 3.9e-4 # Heat capacity [kJ/(g*K)]
    # t0 = 300    # Initial temperature [K]
    # b0 = 2.1    # Speed of the shear wave [km/s]
    # alpha = 1.0 # Non-linear
    # beta = 3.0  # characteristic
    # gamma = 2.0 # constants

    rho0 = _rho0
    c0 = _c0
    cv = _cv
    t0 = _t0
    b0 = _b0
    alpha = _alpha
    beta = _beta
    gamma = _gamma

    # Secondary parameters
    b0sq = b0^2              # Formerly B0
    k0 = c0^2 - (4 / 3) * b0^2

    return new(rho0, c0, cv, t0, b0, alpha, beta, gamma, b0sq, k0)
  end
end # struct Barton2009 <: EoS

function energy(eos::Barton2009, S, G::Array{<:Any,1})
  b0sq = eos.b0sq
  k0 = eos.k0
  alpha = eos.alpha
  beta = eos.beta
  gamma = eos.gamma
  cv = eos.cv
  t0 = eos.t0

  i = invariants(G)

  U = (0.5 * k0 / (alpha^2) * (i[3]^(0.5 * alpha) - 1.0)^2
       +
       cv * t0 * i[3]^(0.5 * gamma) * (exp(S / cv) - 1.0)
  )

  W = 0.5 * b0sq * i[3]^(0.5 * beta) * (i[1]^2 / 3.0 - i[2])
  e_int = U + W
  return e_int
end

function entropy(eos::Barton2009, e_int, G::Array{<:Any,1})
  b0sq = eos.b0sq
  k0 = eos.k0
  alpha = eos.alpha
  beta = eos.beta
  gamma = eos.gamma
  cv = eos.cv
  t0 = eos.t0

  i = invariants(G)

  S = e_int - 0.5 * b0sq * i[3]^(0.5 * beta) * (i[1]^2 / 3 - i[2]) - 0.5 * k0 / (alpha^2) * (i[3]^(0.5 * alpha) - 1)^2
  S = (S / (cv * t0 * i[3]^(0.5 * gamma)) + 1)
  if S < 1e-6
    S = 1e-6
  end
  return log(S) * cv
end

function stress(eos::T, ent, F::Array{<:Any,1})::Array{<:Any,1} where {T<:EoS}
  den = eos.rho0 / det(reshape(F, (3, 3)))
  G = finger(F)

  dedG = gradient(G -> energy(eos, ent, G), G)

  G = reshape(G, (3, 3))
  dedG = reshape(dedG, (3, 3))

  stress = -2 * den .* G * dedG
  return reshape(stress, length(stress))
end

function acoustic(eos::T, ent, F::Array{<:Any,1}, n::Array{<:Any,1})::Array{<:Any,2} where {T<:EoS}
  acoustic = zeros(3, 3)
  dTdF = reshape(jacobian(F -> stress(eos, ent, F), F), (3, 3, 3, 3))
  F = reshape(F, (3, 3))
  den = eos.rho0 / det(F)
  A = (1 / den) .* dTdF

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

function temperature(eos::T, ent, F::Array{<:Any,1}) where {T<:EoS}
  G = finger(F)
  return derivative(S -> energy(eos, S, G), ent)
end

# Deprecated function
# Здесь Q --- одномерный массив.
"""
    Returns density computed from conservative variables for GRP model.
    Actual input is ``\\rho \\tn{F}``.
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


# ##############################################################################
# Stiffened
# ##############################################################################

"""
    Stiffened EoS
"""
struct Stiffened <: EoS
  # Primary parameters
  rho0    # Initial density [g/cm^3]
  s
  c0      # Speed of sound [km/s]
  cv      # Heat capacity [kJ/(g*K)]
  mu      # Shear elastic modulus
  T0      # Initial temperature
  G0      # Mie-Gruneisen parameter
  S0      # Initial entropy

  function Stiffened(; rho0=2.78, s=1.338, c0=5.33, cv=9.3e-4, mu=27.6, T0=300, G0=2, S0=1e-3)
    return new(rho0, s, c0, cv, mu, T0, G0, S0)
  end
end # struct Stiffened <: EoS

function energy(eos::Stiffened, S, G::Array{<:Any,1})
  rho0 = eos.rho0
  s = eos.s
  mu = eos.mu
  c0 = eos.c0
  cv = eos.cv
  T0 = eos.T0
  G0 = eos.G0
  S0 = eos.S0

  i = invariants(G)
  i[2] = tr(reshape(G, (3, 3))^2)

  rho = rho0 * sqrt(i[3])
  nu = rho0 / rho
  cs = sqrt(mu / rho)

  e_ref = 1 / 2 * c0^2 * (1 - nu)^2 / (1 - s * (1 - nu))^2
  T_ref = 1 / (2 * cv * s^4) * (
    s * (-c0^2 * (G0 - 3 * s) + 2 * cv * s^3 * T0) * exp(G0 * (1 - nu))
    + (c0^2 * s * ((G0 - 4 * s) * s * nu + G0 - (3 + G0) * s + 4 * s^2))
      /
      (s * (nu - 1) + 1)^2
    + c0^2 * exp(G0 * (1 - (1 / s + nu))) * (G0^2 - 4 * G0 * s + 2s^2) * (expinti(G0 / s) - expinti(G0 * (-1 + 1 / s + nu)))
  )
  T = T0 * exp((S - S0) / cv - G0 * (nu - 1))

  e_int = e_ref + cv * (T - T_ref)
  e_int += cs^2 / 4 * (i[2] - 1 / 3 * i[1]^2)

  return e_int
end

function pressure(eos::Stiffened, den, e_int, i::Array{<:Any,1})
  rho0 = eos.rho0
  c0 = eos.c0
  cv = eos.cv
  T0 = eos.T0
  G0 = eos.G0
  S0 = eos.S0

  i = invariants(G)

  rho = rho0 * sqrt(i[3])
  nu = rho0 / rho
  e_ref = 1 / 2 * c0^2 * (1 - nu)^2 / (1 - s * (1 - nu))^2
  T_ref = 1 / (2 * cv * s^4) * (
    s * (-c0^2 * (G0 - 3 * s) + 2 * cv * s^3 * T0) * exp(G0 * (1 - nu))
    + (c0^2 * s * ((G0 - 4 * s) * s * nu + G0 - (3 + G0) * s + 4 * s^2))
      /
      (s * (nu - 1) + 1)^2
    + c0^2 * exp(G0 * (1 - (1 / s + nu))) * (G0^2 - 4 * G0 * s + 2s^2) * (expinti(G0 / s) - expinti(G0 * (-1 + 1 / s + nu)))
  )



  return pres
end

function entropy(eos::Stiffened, e_int, G::Array{<:Any,1})
  rho0 = eos.rho0
  mu = eos.mu
  s = eos.s
  c0 = eos.c0
  cv = eos.cv
  T0 = eos.T0
  G0 = eos.G0
  S0 = eos.S0

  i = invariants(G)
  i[2] = tr(reshape(G, (3, 3))^2)

  rho = rho0 * sqrt(i[3])
  nu = rho0 / rho
  cs = sqrt(mu / rho)

  e_ref = 1 / 2 * c0^2 * (1 - nu)^2 / (1 - s * (1 - nu))^2
  T_ref = 1 / (2 * cv * s^4) * (
    s * (-c0^2 * (G0 - 3 * s) + 2 * cv * s^3 * T0) * exp(G0 * (1 - nu))
    + (c0^2 * s * ((G0 - 4 * s) * s * nu + G0 - (3 + G0) * s + 4 * s^2))
      /
      (s * (nu - 1) + 1)^2
    + c0^2 * exp(G0 * (1 - (1 / s + nu))) * (G0^2 - 4 * G0 * s + 2s^2) * (expinti(G0 / s) - expinti(G0 * (-1 + 1 / s + nu)))
  )

  e_int -= cs^2 / 4 * (i[2] - 1 / 3 * i[1]^2)
  ent = S0 + cv * (log((T_ref + (e_int - e_ref) / cv) / T0) + G0 * (nu - 1))

  return ent
end


end # module EoS
# EOF
