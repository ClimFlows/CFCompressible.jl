module NH_state

using CFHydrostatics
using ..CFCompressible: FCE

using SHTnsSpheres: SHTnsSphere, analysis_scalar!, analysis_vector!, erase, void
using CFDomains: VoronoiSphere, Stencils
using ManagedLoops: @with, @vec, @unroll

"""
    diags_HPE = CFHydrostatics.diagnostics(model_HPE)
    model_FCE = FCE(model_HPE, gravity)
    state_FCE = diagnose(model_FCE, diags_HPE, state_HPE)

    Diagnose fully-compressible degrees of freedom from `state_HPE`, containing hydrostatic degrees of freedom.
"""
diagnose(model::FCE, diags, state) = diagnose_FCE(model, model.domain.layer, diags, state)

function diagnose_FCE(model, sph::SHTnsSphere, diags, state)
    (; mgr, planet) = model
    (; radius, gravity) = planet
    rad2, invrad, gm2 = radius^2, inv(radius), gravity^-2

    # NB : in HPE, `masses` are multiplied by gravity but not in FCE => divide by gravity
    session = open(diags; model, state)
    mass = (radius^2/gravity) * session.masses.air # 0-form => 2-form
    ux, uy = (invrad * u for u in session.uv) # physical => contravariant
    Phi_x, Phi_y = session.gradPhi_cov
    dPhi = session.dgeopotential
    W = similar(dPhi)

    @with mgr let (irange, jrange) = (axes(W, 1), axes(W, 2))
        krange = axes(ux, 3)
        nz = length(krange)
        for j in jrange
            @vec for i in irange
                u, v = ux[i, j, 1], uy[i, j, 1]
                Phi_dot = dPhi[i, j, 1] + (u * Phi_x[i, j, 1] + v * Phi_y[i, j, 1])
                W[i, j, 1] = gm2 * Phi_dot * mass[i, j, 1] / 2
            end
            for l in 2:nz
                @vec for i in irange
                    u = ux[i, j, l - 1] + ux[i, j, l]
                    v = uy[i, j, l - 1] + uy[i, j, l]
                    Phi_dot = dPhi[i, j, l] + (u * Phi_x[i, j, l] + v * Phi_y[i, j, l]) / 2
                    W[i, j, l] = gm2 * Phi_dot * (mass[i, j, l - 1] + mass[i, j, l]) / 2
                end
            end
            @vec for i in irange
                u, v = ux[i, j, nz], uy[i, j, nz]
                Phi_dot = dPhi[i, j, nz + 1] +
                          (u * Phi_x[i, j, nz + 1] + v * Phi_y[i, j, nz + 1])
                W[i, j, nz + 1] = gm2 * Phi_dot * mass[i, j, nz] / 2
            end

            # horizontal NH momentum (covariant)
            for (ui, Phi_i) in ((ux, Phi_x), (uy, Phi_y))
                @vec for i in irange
                    ui[i, j, 1] = rad2 * ui[i, j, 1] +
                                  (Phi_i[i, j, 1] * 2W[i, j, 1] +
                                   Phi_i[i, j, 2] * W[i, j, 2]) /
                                  (2 * mass[i, j, 1])
                end
                for k in 2:(nz - 1)
                    @vec for i in irange
                        ui[i, j, k] = rad2 * ui[i, j, k] +
                                      (Phi_i[i, j, k] * W[i, j, k] +
                                       Phi_i[i, j, k + 1] * W[i, j, k + 1]) /
                                      (2 * mass[i, j, k])
                    end
                end
                @vec for i in irange
                    ui[i, j, nz] = rad2 * ui[i, j, nz] +
                                   (Phi_i[i, j, nz] * W[i, j, nz] +
                                    Phi_i[i, j, nz + 1] * 2W[i, j, nz + 1]) /
                                   (2 * mass[i, j, nz])
                end
            end
        end # j in jrange
    end # @with

    (; mass_air_spec, mass_consvar_spec) = state
    uv_spec = analysis_vector!(void, erase((ucolat=ux, ulon=uy)), sph)
    W_spec = analysis_scalar!(void, erase(W), sph)
    Phi_spec = analysis_scalar!(void, erase(session.geopotential), sph)
    return (; mass_air_spec=mass_air_spec/gravity, mass_consvar_spec=mass_consvar_spec/gravity, uv_spec, Phi_spec, W_spec)
end

function diagnose_FCE(model, sph::VoronoiSphere, diags, state)
    (; gravity) = model.planet
    # NB : in HPE, `masses` are multiplied by gravity but not in FCE => divide by gravity
    session = open(diags; model, state)
    mass_air = state.mass_air / gravity
    mass_consvar = state.mass_consvar / gravity
    Phi = session.geopotential_i
    ucov = copy(state.ucov)
    W = zero(Phi) # FIXME: use hydrostatic velocity
    return (; mass_air, mass_consvar, ucov, Phi, W)
end

end # module
