module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using CFDomains: VoronoiSphere
using SHTnsSpheres:
                    SHTnsSphere,
                    analysis_scalar!,
                    synthesis_scalar!,
                    analysis_vector!,
                    synthesis_vector!,
                    synthesis_spheroidal!,
                    divergence!,
                    curl!
using ManagedLoops: @with, @vec

using ..CFCompressible.Dynamics: FCE_tendencies!


#======================= FCE Diagnostics =======================#

# Return a CookBook of diagnostic functions for the FCE model.
function diagnostics_FCE()
    return CookBook(;
                    # used for dispatch
                    sphere,
                    # independent from vertical coordinate, native grid
                    temperature_i,
                    # independent from vertical coordinate
                    uv, ulon, ulat,
                    temperature,
                    specific_volume,
                    pressure,
                    surface_pressure,
                    hydrostatic_pressure,
                    NH_pressure,
                    sound_speed,
                    # depend on vertical coordinate
                    masses,
                    conservative_variable,
                    Phi_dot,
                    slow_fast_scratch, slow, fast, scratch,
                    slow_mass_air,
                    )
end

#=================== independent from vertical coordinate ==============#

sphere(model) = model.domain.layer

# native -> lonlat

temperature(to_lonlat, temperature_i) = to_lonlat(temperature_i)

# same as HPE

slow_mass_air(model, slow) = synthesis_scalar!(void, slow.mass_air_spec, model.domain.layer)

function sound_speed(model, pressure, temperature_i)
    return model.gas(:p, :T).sound_speed.(pressure, temperature_i)
end

function temperature_i(model, pressure, conservative_variable)
     return model.gas(:p, :consvar).temperature.(pressure, conservative_variable)
end

conservative_variable(masses) = @. masses.consvar / masses.air

ulon(uv) = uv.ulon
ulat(uv) = -uv.colat

# FCE-specicific

function uv(model, scratch)
    (; Uxk, Uyk) = scratch.slow_mass.fluxes
    m = scratch.common.mk
    (; radius) = model.planet
    return (ucolat=(@. radius*Uxk/m), ulon=(@. radius*Uyk/m))
end

function specific_volume(sphere::SHTnsSphere, model, scratch)
    (; gravity, radius) = model.planet
    Jac = radius^2/gravity
    Phi, m = scratch.common.Phil, scratch.common.mk
    dPhi = Phi[:,:,2:end]-Phi[:,:,1:end-1]
    return @. Jac*dPhi/m
end

function specific_volume(sphere::VoronoiSphere, model, scratch)
    (; gravity, radius) = model.planet
    Jac = radius^2/gravity
    Phi, m = scratch.common.Phil, scratch.common.mk
    dPhi = Phi[:,2:end]-Phi[:,1:end-1]
    return permutedims((@. Jac*dPhi/m), (2,1))
end

surface_pressure(scratch) = scratch.common.ps

function pressure(model, specific_volume, conservative_variable)
    return model.gas(:v, :consvar).pressure.(specific_volume, conservative_variable)
end

function hydrostatic_pressure(model, masses)
    mass = masses.air
    p = similar(mass)
    ptop, gravity = model.vcoord.ptop, model.planet.gravity # avoids capturing `model`
    @with model.mgr let (irange, jrange) = (axes(p, 1), axes(p, 2))
        nz = size(p, 3)
        for j in jrange
            @vec for i in irange
                p[i, j, nz] = ptop + mass[i, j, nz] / 2
                for k in nz:-1:2
                    p[i, j, k - 1] = p[i, j, k] + (mass[i, j, k] + mass[i, j, k - 1]) * (gravity/2)
                end
            end
        end
    end
    return p
end

NH_pressure(pressure, hydrostatic_pressure) = pressure - hydrostatic_pressure

slow_fast_scratch(model, state) = FCE_tendencies!(void, void, void, model, model.domain.layer, state, 0)
slow(slow_fast_scratch) = slow_fast_scratch[1]
fast(slow_fast_scratch) = slow_fast_scratch[2]
scratch(slow_fast_scratch) = slow_fast_scratch[3]

Phi_dot(scratch) = scratch.fast_spat.dPhil

#======================= depend on vertical coordinate ======================#

# same as HPE

function masses(sphere, model, state)
    fac, sph = model.planet.radius^-2, model.domain.layer
    return (air=synthesis_scalar!(void, fac * state.mass_air_spec, sph),
            consvar=synthesis_scalar!(void, fac * state.mass_consvar_spec, sph))
end

function masses(sphere::VoronoiSphere, model, state)
    fac = model.planet.radius^-2
    return (air=fac*state.mass_air, consvar=fac*state.mass_consvar)
end

# ======================= FC2D Diagnostics =======================#


# Return a CookBook of diagnostic functions for the FC2D model.
diagnostics_FC2D() = CookBook(;
    r           = state -> state.m[:, :, 1],
    s           = state -> state.m[:, :, 2] ./ state.m[:, :, 1],
    q           = state -> state.m[:, :, 3] ./ state.m[:, :, 1],
    u           = state -> state.u,
    w           = state -> state.w,
    p           = scratch -> scratch.thermo.p,
    T           = scratch -> scratch.thermo.T,
    vorticity   = scratch -> scratch.derivatives.ω,
    divu        = scratch -> scratch.derivatives.divu,
    conjvar     = scratch -> scratch.thermo.conjvar,
    chempot     = scratch -> scratch.thermo.chempot,
    rs          = state -> state.m[:, :, 2],
    rq          = state -> state.m[:, :, 3],
    ωw          = scratch -> scratch.advection.ωw,
    ωu          = scratch -> scratch.advection.ωu, 
    Jcons_x     = scratch -> scratch.fluxes.Jcons_x,
    Jcons_z     = scratch -> scratch.fluxes.Jcons_z,
    Jq_x        = scratch -> scratch.fluxes.Jq_x,
    Jq_z        = scratch -> scratch.fluxes.Jq_z,
    ρε          = scratch -> scratch.viscous.ρε,
    σ_cons      = scratch -> scratch.irreversible.σ_cons,
    div_Jcons   = (scratch, model) -> div_diag(scratch.irreversible.Jcons_x, scratch.irreversible.Jcons_z, scratch, model),
    div_Jq      = (scratch, model) -> div_diag(scratch.irreversible.Jq_x, scratch.irreversible.Jq_z, scratch, model),
    div_ru      = (scratch, model) -> div_diag(scratch.advection.mu[:,:,1], scratch.advection.mw[:,:,1], scratch, model),
    div_rsu     = (scratch, model) -> div_diag(scratch.advection.mu[:,:,2], scratch.advection.mw[:,:,1], scratch, model),
    div_rqu     = (scratch, model) -> div_diag(scratch.advection.mu[:,:,3], scratch.advection.mw[:,:,2], scratch, model),
    ∂u_x        = scratch -> scratch.derivatives.∂u_x,
    ∂w_z        = scratch -> scratch.derivatives.∂w_z,
    ∂u_z        = scratch -> scratch.derivatives.∂u_z,
    ∂w_x        = scratch -> scratch.derivatives.∂w_x,
    dr          = dstate -> dstate.dm[:, :, 1],
    drs         = dstate -> dstate.dm[:, :, 2],
    drq         = dstate -> dstate.dm[:, :, 3],
    du          = dstate -> dstate.du,
    dw          = dstate -> dstate.dw,
    B           = scratch -> scratch.thermo.B,
    ke          = scratch -> scratch.advection.K,
    ru          = scratch -> scratch.advection.mu[:,:,1],
    rw          = scratch -> scratch.advection.mw[:,:,1],
    su          = scratch -> scratch.advection.mu[:,:,2],
    sw          = scratch -> scratch.advection.mw[:,:,2],
    qu          = scratch -> scratch.advection.mu[:,:,3],
    qw          = scratch -> scratch.advection.mw[:,:,3],
)

function div_diag(Ax, Az, scratch, model)
    (; domain, inv_dx, inv_dz) = model.environment
    div_A = similar(scratch.derivatives.∂u_x)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            div_A[i, j] = inv_dx * dif_x(Ax, i, j) + inv_dz * dif_z(Az, i, j)
        end       
    end
    periodize!(model, div_A)
    return div_A
end

end # module
