module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet, Tank2D
import CFBoxes
using CFBoxes: Box2D, dims, AdvectionScheme, BoxBoundaryConditions, discretize, check_tracers
using CFDiffusionSchemes: ThermoFluxScheme, ViscosityScheme
import CFTimeSchemes

export FC2D

struct NewtonSolve
    niter::Int          # number of Newton-Raphson iterations
    flip_solve::Bool    # direction of LU solver passes: false => bottom-up then top-down ; true => top-down then bottom-up
    update_W::Bool      # update W during Newton iteration (true), or only at the end (false)
    verbose::Bool
end
function NewtonSolve(; niter=5, flip_solve=false, update_W=false, verbose=false, other...)
    return NewtonSolve(niter, flip_solve, update_W, verbose)
end

# fully compressible model for a rotating planet
struct FCE{F, Manager, Coord, Domain, Fluid, TwoDimScalar<:AbstractArray{F}}
    mgr::Manager
    vcoord::Coord
    planet::ShallowTradPlanet{F}
    domain::Domain
    gas::Fluid
    fcov::TwoDimScalar # covariant Coriolis factor = f(lat)*radius^2
    # bottom boundary condition: p = ps - rhob*(Phi-Phis)
    Phis::TwoDimScalar # target surface geopotential
    rhob::F # larger rhob makes smaller Phi-Phis but stiffer system
    # options for Newton iteration used to solve HEVI problem
    newton::NewtonSolve
end
function FCE(model::HPE{F}, gravity::F, rhob, newton) where F
    (; mgr, vcoord, planet, domain, gas, fcov, Phis) = model
    (; radius, Omega) = planet
    planet = ShallowTradPlanet(radius, Omega, gravity)
    return FCE(mgr, vcoord, planet, domain, gas, fcov, Phis, rhob, newton)
end

# fully compressible model for a 2D box geometry
struct FC2D{F, Manager, Domain<:Box2D, Boundary<:BoxBoundaryConditions, Fluid, Advection<:AdvectionScheme, ThermoFlux<:ThermoFluxScheme{F}}
    mgr::Manager
    domain::Domain
    space::Tank2D{F}
    boundary::Boundary
    fluid::Fluid
    advection::Advection
    thermoflux::ThermoFlux
    viscosity::ViscosityScheme{F}
    dx::F
    dz::F
    inv_dx::F
    inv_dz::F
end
function FC2D((; mgr, domain, space, fluid, advection_scheme, thermoflux_scheme, viscosity_scheme, boundary_conditions))
    Mx, Mz = CFBoxes.dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    check_tracers(boundary_conditions, 2)       # tracers (consvar, q)
    boundary = discretize(boundary_conditions, domain, (dx, dz))
    return FC2D(mgr, domain, space, boundary, fluid, advection_scheme, thermoflux_scheme, viscosity_scheme, dx, dz, 1/dx, 1/dz)
end

# implemented in Dynamics
"""
    slow, fast, scratch = FCE_tendencies!(slow, fast, scratch, model, layer, state, tau)
"""
function FCE_tendencies! end

# specify tendencies
"""
    slow, fast, scratch = tendencies!(slow, fast, scratch, model, state, t, tau)
"""
tendencies!(slow, fast, scratch, model::FCE, state, _, tau) = FCE_tendencies!(slow, fast, scratch, model, model.domain.layer, state, tau)

"""
    dstate, scratch = tendencies!(dstate, scratch, model, state, t)
"""
tendencies!(dstate, scratch, model::FC2D, state, t) = BoxDynamics.FC2D_tendencies!(dstate, scratch, model, state, t)
CFTimeSchemes.tendencies!(dstate, scratch, model::FC2D, state, t) = BoxDynamics.FC2D_tendencies!(dstate, scratch, model, state, t)

# vertical remap
"""
    tmp = vertical_remap!(state, model::FCE, void) # first call
    vertical_remap!(state, model::FCE, tmp)        # subsequent calls

Apply vertical remap to `state`. After this remap, `state` 
conforms to the vertical coordinate defined by `model.vcoord`.

If scratch space `tmp` is `::Void`, then `new_tmp` is a (nested)
named tuple of arrays that can be used as `tmp` in a future call.
"""
vertical_remap!(state, model::FCE, tmp) = vertical_remap_FCE!(state, model, model.domain.layer, tmp)
function vertical_remap_FCE! end # implemented in `remap_spectral.jl`

# specify initialization
"""
    (; m, u, w) = initialize(model::FC2D; density, consvar, q, velocity = nothing)
    ρ = balanced_density(model::FC2D; consvar, q, ptop)
    (u, w) = divergence_free_velocity(model::FC2D, ρ, ψ)

Initial state of the FC2D model, in three steps: the rest density, a discretely divergence-free perturbation for
it, then the state; see `BoxInitialize`.
"""
initialize(model::FC2D; kwargs...) = BoxInitialize.initialize(model; kwargs...)
balanced_density(model::FC2D; kwargs...) = BoxInitialize.balanced_density(model; kwargs...)
divergence_free_velocity(model::FC2D, ρ, ψ) = BoxInitialize.divergence_free_velocity(model, ρ, ψ)

# specify main time integration loop
"""
    state = loop(model::FC2D, state0, time_scheme, params; outputs = nothing)
Main time integration loop for the FC2D model; returns the final state.
- `model`: FC2D model instance
- `state0`: initial state (see `initialize`)
- `time_scheme`: time integration scheme
- `params`: (; Nslice, slice_size, dt_max, cfl): `Nslice` time slices of `slice_size` [s], each split into equal
  steps of at most `dt_max` [s] and `cfl` (≤ 1 for stability) times the stability limit of `time_scheme`, see
  `cfl_condition`
- `outputs`: callable `outputs(model, state, dstate, scratch, book, t)`, called at t = 0 and at the end of each
  time slice, with the diagnostics `book` and the time `t` [s] (e.g. to record diagnostics), then closed with
  `close(outputs)` when the run ends, even on error; `nothing` records nothing
"""
loop(model::FC2D, state0, time_scheme, params; kwargs...) = Loops.loop(model, state0, time_scheme, params; kwargs...)

# pointwise energies of FC2D
"""
    ke, pe, te = energies!((ke, pe, te), state, model::FC2D)

Kinetic, potential + internal, and total energy per unit volume at cell centres; see `BoxEnergetics.energies!`.
"""
energies!(energies, state, model::FC2D) = BoxEnergetics.energies!(energies, state, model)

# specify diagnostics for each model
diagnostics(::FCE) = Diagnostics.diagnostics_FCE()
diagnostics(model::FC2D) = Diagnostics.diagnostics_FC2D(model)
internal_diagnostics(::FC2D) = Diagnostics.internal_diagnostics_FC2D()
diagnostic_info(::FC2D) = Diagnostics.diagnostic_info_FC2D()
describe_diagnostics(model::FC2D) = foreach(pairs(diagnostic_info(model))) do (name, (; long_name, units))
    println(rpad(name, 34), rpad(long_name, 52), units)
end

# include("julia/lazy_broadcast.jl")

include("julia/vertical_dynamics.jl")
include("julia/horizontal_energies.jl")
include("julia/dynamics.jl")
include("julia/voronoi_dynamics.jl")
include("julia/remap_collocated.jl")
include("julia/remap_spectral.jl")
include("julia/NH_state.jl")
include("julia/diagnostics.jl")

include("julia/box_dynamics.jl")
include("julia/box_initialize.jl")
include("julia/box_energetics.jl")
include("julia/loops.jl")

end # module CFCompressible
