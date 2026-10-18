module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet, Tank2D
import CFBoxes
using CFBoxes: Box2D, dims, SpatialScheme, check_halo, VortexScheme, BoxBoundaryConditions, discretize, check_tracers
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

"""
    model = FC2D((; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, thermoflux_scheme, viscosity_scheme,
                    boundary_conditions))

Fully compressible binary fluid in a 2D box (x, z): prognostic m = (ρ, ρ consvar, ρ q) at cell centers and the
velocity (u, w) on faces. `spatial_scheme` (e.g. `Centred2()`) sets the interpolations and differences of the C-grid,
`vortex_scheme` the vortex force.
"""
struct FC2D{F, Manager, Domain<:Box2D, Boundary<:BoxBoundaryConditions, Fluid, Spatial<:SpatialScheme, Vortex<:VortexScheme, ThermoFlux<:ThermoFluxScheme{F}}
    "loop manager running the kernels (e.g. `MultiThread(VectorizedCPU())`), or `nothing`"
    mgr::Manager
    "the box (grid sizes, halo widths, topology)"
    domain::Domain
    "the physical box: lengths Lx, Lz [m] and gravity g [m s⁻²]"
    space::Tank2D{F}
    "wall conditions, discretized on the grid"
    boundary::Boundary
    "equation of state of the binary fluid"
    fluid::Fluid
    "interpolations and differences of the C-grid (`operators`)"
    spatial_scheme::Spatial
    "discretization of the vortex force ω×u"
    vortex_scheme::Vortex
    "diffusion of consvar and q (coefficients k_T, k_q [m² s⁻¹])"
    thermoflux::ThermoFlux
    "viscosity (dynamic viscosity μ [Pa s])"
    viscosity::ViscosityScheme{F}
    "grid spacing along x [m]"
    dx::F
    "grid spacing along z [m]"
    dz::F
    "1/dx [m⁻¹]"
    inv_dx::F
    "1/dz [m⁻¹]"
    inv_dz::F
end
function FC2D(params)
    (; mgr, domain, space, fluid, spatial_scheme, vortex_scheme, thermoflux_scheme, viscosity_scheme, boundary_conditions) = params
    check_halo(domain, spatial_scheme)
    Mx, Mz = CFBoxes.dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    check_tracers(boundary_conditions, 2)       # tracers (consvar, q)
    boundary = discretize(boundary_conditions, domain, (dx, dz))
    return FC2D(mgr, domain, space, boundary, fluid, spatial_scheme, vortex_scheme, thermoflux_scheme, viscosity_scheme, dx, dz, 1/dx, 1/dz)
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

# the FC2D methods of tendencies! are in `box_dynamics.jl`

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
    state = loop(state, model::FC2D, time_scheme, params; outputs = nothing)

Main time integration loop for the FC2D model; advances `state` in place and returns it.
- `model`: FC2D model instance
- `state`: initial state (see `initialize`), overwritten by the final state; copy it first to keep it
- `time_scheme`: time integration scheme
- `params`: (; Nslice, slice_size, dt_max, cfl): `Nslice` time slices of `slice_size` [s], each split into equal
  steps of at most `dt_max` [s] and `cfl` (≤ 1 for stability) times the stability limit of `time_scheme`, see
  `cfl_condition`
- `outputs`: callable `outputs(model, state, dstate, scratch, book, t)`, called at t = 0 and at the end of each
  time slice, with the diagnostics `book` and the time `t` [s] (e.g. to record diagnostics), then closed with
  `close(outputs)` when the run ends, even on error; `nothing` records nothing
"""
loop(state, model::FC2D, time_scheme, params; kwargs...) = Loops.loop(state, model, time_scheme, params; kwargs...)

# specify diagnostics for each model
diagnostics(::FCE) = Diagnostics.diagnostics_FCE()
# the FC2D methods of these are in `diagnostics.jl`
function internal_diagnostics end
function diagnostic_info end
function describe_diagnostics end

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
include("julia/loops.jl")

end # module CFCompressible
