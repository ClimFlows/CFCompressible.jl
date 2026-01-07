module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet, Tank2D
using CFBoxes: Box2D, dims, AdvectionScheme, BoundaryConditions2D
using CFDiffusionSchemes: HeatFluxScheme, ViscosityScheme

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
struct FC2D{F, Manager, Fluid}
    mgr::Manager
    domain::Box2D
    space::Tank2D{F}
    boundary::BoundaryConditions2D
    fluid::Fluid
    advection::AdvectionScheme
    heatflux::HeatFluxScheme{F}
    viscosity::ViscosityScheme{F}
    dx::F
    dz::F
    inv_dx::F
    inv_dz::F
end
function FC2D((; mgr, domain, space, fluid, advection_scheme, heatflux_scheme, viscosity_scheme, boundary_conditions))
    Mx, Mz = CFBoxes.dims(domain)
    dx, dz = space.Lx / Mx, space.Lz / Mz
    return FC2D(mgr, domain, space, boundary_conditions, fluid, advection_scheme, heatflux_scheme, viscosity_scheme, dx, dz, 1/dx, 1/dz)
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

# specify initialization
"""
    (; m, u, w) = initialize(model::FC2D, params)
Initialize the state for the FC2D model.
"""
initialize(model::FC2D, params) = BoxInitialize.initialize(model, params)

# specify main time integration loop
"""
    loop(model, time_scheme, params; write_func = write_func, write_obj = nothing, plot_func = plot_func, plot_obj = nothing)
Main time integration loop for the FC2D model.
- `write_func`: function to write data at each time slice
- `write_obj`: object to pass to write_func (e.g., file handle) 
- `plot_func`: function to plot data at each time slice
- `plot_obj`: object to pass to plot_func (e.g., plot handle)
"""
loop(model::FC2D, time_scheme, params; kwargs...) = Loops.loop(model, time_scheme, params; kwargs...)

# specify diagnostics for each model
diagnostics(::FCE) = Diagnostics.diagnostics_FCE()
diagnostics(::FC2D) = Diagnostics.diagnostics_FC2D()

# include("julia/lazy_broadcast.jl")

include("julia/vertical_dynamics.jl")
include("julia/horizontal_energies.jl")
include("julia/dynamics.jl")
include("julia/voronoi_dynamics.jl")
include("julia/NH_state.jl")
include("julia/diagnostics.jl")

include("julia/box_dynamics.jl")
include("julia/loops.jl")

end # module CFCompressible
