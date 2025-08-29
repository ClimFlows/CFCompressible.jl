module CFCompressible

using CFHydrostatics: HPE
using CFPlanets: ShallowTradPlanet

struct NewtonSolve
    niter::Int # number of Newton-Raphson iterations
    flip_solve::Bool # direction of LU solver passes: false => bottom-up then top-down ; true => top-down then bottom-up
    update_W::Bool # update W during Newton iteration (true), or only at the end (false)
    verbose::Bool
end

function NewtonSolve(; niter=5, flip_solve=false, update_W=false, verbose=false, other...)
    return NewtonSolve(niter, flip_solve, update_W, verbose)
end

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

# implemented in Dynamics
"""
slow, fast, scratch = FCE_tendencies!(slow, fast, scratch, model, layer, state, tau)
"""
function FCE_tendencies!
end

"""
slow, fast, scratch = tendencies!(slow, fast, scratch, model, state, t, tau)
"""
tendencies!(slow, fast, scratch, model::FCE, state, _, tau) = FCE_tendencies!(slow, fast, scratch, model, model.domain.layer, state, tau)

include("julia/zero_arrays.jl")
include("julia/lazy_broadcast.jl")

include("julia/vertical_dynamics.jl")
include("julia/horizontal_energies.jl")
include("julia/dynamics.jl")
include("julia/voronoi_dynamics.jl")
include("julia/NH_state.jl")
include("julia/diagnostics.jl")

diagnostics(::FCE) = Diagnostics.diagnostics()

end # module CFCompressible
