module Diagnostics

using MutatingOrNot: void, Void
using CookBooks
using SHTnsSpheres:
                    analysis_scalar!,
                    synthesis_scalar!,
                    analysis_vector!,
                    synthesis_vector!,
                    synthesis_spheroidal!,
                    divergence!,
                    curl!
using ManagedLoops: @with, @vec

using ..CFCompressible.Dynamics: FCE_tendencies!

function diagnostics()
    return CookBook(;
                    # independent from vertical coordinate
                    uv, ulon, ulat,
                    specific_volume,
                    pressure,
                    surface_pressure,
                    hydrostatic_pressure,
                    NH_pressure,
                    temperature,
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

# same as HPE

slow_mass_air(model, slow) = synthesis_scalar!(void, slow.mass_air_spec, model.domain.layer)

function sound_speed(model, pressure, temperature)
    return model.gas(:p, :T).sound_speed.(pressure, temperature)
end

function temperature(model, pressure, conservative_variable)
     return model.gas(:p, :consvar).temperature.(pressure, conservative_variable)
end

conservative_variable(masses) = @. masses.consvar / masses.air

ulon(uv) = uv.ulon
ulat(uv) = -uv.colat

# FCE-specicific

#=
function uv(model, state) # FIXME: HPE version
    (; ucolat, ulon) = synthesis_vector!(void, map(copy, state.uv_spec), model.domain.layer)
    invrad = model.planet.radius^-1
    return (ucolat=invrad * ucolat, ulon=invrad * ulon)
end
=#

function uv(model, scratch)
    (; Uxk, Uyk) = scratch.slow_mass.fluxes
    m = scratch.common.mk
    (; radius) = model.planet
    return (ucolat=(@. radius*Uxk/m), ulon=(@. radius*Uyk/m))
end

function specific_volume(model, scratch)
    (; gravity, radius) = model.planet
    Jac = radius^2/gravity
    Phi, m = scratch.common.Phil, scratch.common.mk
    dPhi = Phi[:,:,2:end]-Phi[:,:,1:end-1]
    return @. Jac*dPhi/m
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

slow_fast_scratch(model, state) = FCE_tendencies!(void, void, void, model, model.domain.layer, state, 0.0)
slow(slow_fast_scratch) = slow_fast_scratch[1]
fast(slow_fast_scratch) = slow_fast_scratch[2]
scratch(slow_fast_scratch) = slow_fast_scratch[3]

Phi_dot(scratch) = scratch.fast_spat.dPhil

#======================= depend on vertical coordinate ======================#

# same as HPE

function masses(model, state)
    fac, sph = model.planet.radius^-2, model.domain.layer
    return (air=synthesis_scalar!(void, fac * state.mass_air_spec, sph),
            consvar=synthesis_scalar!(void, fac * state.mass_consvar_spec, sph))
end

end # module
