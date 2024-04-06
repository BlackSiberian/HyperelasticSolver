# File structure

- main.jl

Main driver. Includes other files and is used to set properties of
the calculation.


- NumFluxes.jl

Module with different numerical flux functions.


- EquationsOfState.jl

Module with functions of energy, entropy and stress behavior for different 
equations of state.


- Strains.jl

Module with functions of relations between strains, distortions, etc. Also
contains invariants.


- Hyperelasticity.jl

Single-phase hyperelasticity model. Module with functions of fluxes, conservative
to primitive variables and vice versa.


- HyperelasticityMPh.jl

Multiphase hyperelasticity model. Module with functions of fluxes, conservative
to primitive variables and vice versa.
