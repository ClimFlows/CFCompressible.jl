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
using CFBoxes: xrange_interior, zrange_interior, xdim, zdim, dif_x, dif_z, mirror_halo!

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
                p[i, j, nz] = ptop + mass[i, j, nz] *(gravity/2)
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
#
# Every diagnostic is the full array, halo included, with the halo defined (copied from the bounded state and the
# tendency scratch, or filled by the boundary conditions), so that post-processing can difference it at the
# boundaries; strip the halo for plotting (e.g. `interior(domain, field)` for cell fields). Names are shared with the
# AN2D diagnostics (CFAnelastics); `diagnostic_info_FC2D` gives the long name and units of each.

# CookBook of the FC2D diagnostics, opened with `open(book; state, dstate, scratch, model)`
diagnostics_FC2D() = CookBook(;
    # state
    conservative_variable            = scratch -> scratch.thermo.consvar,
    composition                      = scratch -> scratch.thermo.q,
    u                                = state -> state.u,
    w                                = state -> state.w,
    density                          = state -> state.m[:, :, 1],
    ρconsvar                         = state -> state.m[:, :, 2],
    ρq                               = state -> state.m[:, :, 3],
    # thermodynamics
    specific_entropy                 = (model, pressure, conservative_variable, composition) -> model.fluid(:p, :consvar, :q).specific_entropy.(pressure, conservative_variable, composition),
    potential_temperature            = (model, pressure, conservative_variable, composition) -> model.fluid(:p, :consvar, :q).potential_temperature.(pressure, conservative_variable, composition),
    pressure                         = scratch -> scratch.thermo.p,
    temperature                      = scratch -> scratch.thermo.T,
    conjugate_variable               = scratch -> scratch.thermo.conjvar,
    modified_chemical_potential      = scratch -> scratch.thermo.chempot,
    forcing_potential                = scratch -> scratch.momentum.B,
    # kinematics
    vorticity                        = scratch -> scratch.derivatives.∂u_z .- scratch.derivatives.∂w_x,
    divergence                       = scratch -> scratch.derivatives.∂u_x .+ scratch.derivatives.∂w_z,
    mass_divergence                  = (scratch, model) -> divergence_field(model, view(scratch.advection.mu, :, :, 1), view(scratch.advection.mw, :, :, 1)),
    kinetic_energy                   = scratch -> scratch.advection.K,
    ∂u_x                             = scratch -> scratch.derivatives.∂u_x,
    ∂w_z                             = scratch -> scratch.derivatives.∂w_z,
    ∂u_z                             = scratch -> scratch.derivatives.∂u_z,
    ∂w_x                             = scratch -> scratch.derivatives.∂w_x,
    # irreversible processes
    Jcons_x                          = scratch -> scratch.irreversible.Jcons_x,
    Jcons_z                          = scratch -> scratch.irreversible.Jcons_z,
    Jq_x                             = scratch -> scratch.irreversible.Jq_x,
    Jq_z                             = scratch -> scratch.irreversible.Jq_z,
    viscous_dissipation              = scratch -> scratch.irreversible.ρε,
    conservative_variable_production = scratch -> scratch.irreversible.σ_cons,
    div_Jcons                        = (scratch, model) -> divergence_field(model, scratch.irreversible.Jcons_x, scratch.irreversible.Jcons_z),
    div_Jq                           = (scratch, model) -> divergence_field(model, scratch.irreversible.Jq_x, scratch.irreversible.Jq_z),
    # tendencies
    dρ                               = dstate -> dstate.m[:, :, 1],
    dρconsvar                        = dstate -> dstate.m[:, :, 2],
    dρq                              = dstate -> dstate.m[:, :, 3],
    du                               = dstate -> dstate.u,
    dw                               = dstate -> dstate.w,
)

# CookBook of intermediates of the FC2D tendencies, for debugging: the vorticity terms ωw, ωu of the advection, the
# advective fluxes mu, mw (third index: ρ, ρ consvar, ρ q), and the gradients of the kinetic energy ∂K_x, ∂K_z
internal_diagnostics_FC2D() = CookBook(;
    ωw   = scratch -> scratch.advection.ωw,
    ωu   = scratch -> scratch.advection.ωu,
    mu   = scratch -> scratch.advection.mu,
    mw   = scratch -> scratch.advection.mw,
    ∂K_x = scratch -> scratch.advection.∂K_x,
    ∂K_z = scratch -> scratch.advection.∂K_z,
)

# long name and units of each FC2D diagnostic
diagnostic_info_FC2D() = (;
    conservative_variable            = (long_name = "conservative variable",                                              units = "consvar"),
    composition                      = (long_name = "composition (solute mass fraction)",                                 units = "kg/kg"),
    u                                = (long_name = "horizontal velocity",                                                units = "m s⁻¹"),
    w                                = (long_name = "vertical velocity",                                                  units = "m s⁻¹"),
    density                          = (long_name = "density",                                                            units = "kg m⁻³"),
    ρconsvar                         = (long_name = "density × conservative variable",                                    units = "kg m⁻³ × consvar"),
    ρq                               = (long_name = "density × composition",                                              units = "kg m⁻³"),
    specific_entropy                 = (long_name = "specific entropy",                                                   units = "J kg⁻¹ K⁻¹"),
    potential_temperature            = (long_name = "potential temperature",                                              units = "K"),
    pressure                         = (long_name = "pressure",                                                           units = "Pa"),
    temperature                      = (long_name = "temperature",                                                        units = "K"),
    conjugate_variable               = (long_name = "conjugate variable of consvar, ∂h/∂consvar",                         units = "J kg⁻¹ / consvar"),
    modified_chemical_potential      = (long_name = "modified chemical potential, ∂h/∂q",                                 units = "J kg⁻¹"),
    forcing_potential                = (long_name = "potential of the momentum forcing, h - conjvar consvar - chempot q", units = "J kg⁻¹"),
    vorticity                        = (long_name = "vorticity ∂u/∂z - ∂w/∂x",                                            units = "s⁻¹"),
    divergence                       = (long_name = "velocity divergence ∂u/∂x + ∂w/∂z",                                  units = "s⁻¹"),
    mass_divergence                  = (long_name = "mass divergence ∇·(ρu)",                                             units = "kg m⁻³ s⁻¹"),
    kinetic_energy                   = (long_name = "kinetic energy per unit mass",                                       units = "J kg⁻¹"),
    ∂u_x                             = (long_name = "∂u/∂x",                                                              units = "s⁻¹"),
    ∂w_z                             = (long_name = "∂w/∂z",                                                              units = "s⁻¹"),
    ∂u_z                             = (long_name = "∂u/∂z",                                                              units = "s⁻¹"),
    ∂w_x                             = (long_name = "∂w/∂x",                                                              units = "s⁻¹"),
    Jcons_x                          = (long_name = "diffusive flux of consvar along x",                                  units = "kg m⁻² s⁻¹ × consvar"),
    Jcons_z                          = (long_name = "diffusive flux of consvar along z",                                  units = "kg m⁻² s⁻¹ × consvar"),
    Jq_x                             = (long_name = "diffusive flux of composition along x",                              units = "kg m⁻² s⁻¹"),
    Jq_z                             = (long_name = "diffusive flux of composition along z",                              units = "kg m⁻² s⁻¹"),
    viscous_dissipation              = (long_name = "viscous dissipation",                                                units = "W m⁻³"),
    conservative_variable_production = (long_name = "irreversible production of consvar",                                 units = "kg m⁻³ s⁻¹ × consvar"),
    div_Jcons                        = (long_name = "divergence of the diffusive flux of consvar",                        units = "kg m⁻³ s⁻¹ × consvar"),
    div_Jq                           = (long_name = "divergence of the diffusive flux of composition",                    units = "kg m⁻³ s⁻¹"),
    dρ                               = (long_name = "tendency of ρ",                                                      units = "kg m⁻³ s⁻¹"),
    dρconsvar                        = (long_name = "tendency of ρ consvar",                                              units = "kg m⁻³ s⁻¹ × consvar"),
    dρq                              = (long_name = "tendency of ρ q",                                                    units = "kg m⁻³ s⁻¹"),
    du                               = (long_name = "tendency of u",                                                      units = "m s⁻²"),
    dw                               = (long_name = "tendency of w",                                                      units = "m s⁻²"),
)

# divergence of a flux (Ax, Az) on the faces, at cell centres: δ_x Ax / Δx + δ_z Az / Δz on the interior cells, with
# the halo filled by mirroring (periodic copy in periodic directions)
function divergence_field((; domain, inv_dx, inv_dz), Ax, Az)
    div_A = zeros(eltype(Ax), xdim(domain), zdim(domain))
    for i in xrange_interior(domain), j in zrange_interior(domain)
        div_A[i, j] = inv_dx * dif_x(Ax, i, j) + inv_dz * dif_z(Az, i, j)
    end
    return mirror_halo!(domain, div_A)
end

end # module
