module RemapSpectral

import SHTnsSpheres: SHTnsSpheres, SHTnsSphere
using CFDomains: data_layout
using MutatingOrNot: similar!
using CFHydrostatics.RemapHPE: flatten

import ..CFCompressible
using ..RemapCollocated: cov_to_horiz!, NH_pressure!, remap_FCE!, NH_geopotential!, horiz_to_cov!

function CFCompressible.vertical_remap_FCE!(state, model, sph::SHTnsSphere, tmp)
    # steps:
    #   1 - spectral => spatial
    #   2 - covariant momentum, mass => horizontal momentum, weight
    #   3 - geopot => p_NH
    #   4 - remap
    #   5 - p_NH => Phi, ∇Phi
    #   6 - horizontal momentum, weight => covariant momentum, mass
    #   7 - spatial => spectral

    layout = data_layout(model.domain)
    @inline flat(x) = flatten(x, layout)  # (nx, ny, nz) => (nx * ny, nz)
    @inline flat(x::Union{Tuple,NamedTuple}) = map(flat, x)
    @inline flat(x...) = map(flat,x)

    #   1 - spectral => spatial
    mass_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.mass, state.mass_air_spec, sph)
    massq_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.massq, state.mass_consvar_spec, sph)
    (ux, uy) = uv_spat = SHTnsSpheres.synthesis_vector!(tmp.spat.uv, state.uv_spec, sph)
    W_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.W, state.W_spec, sph)
    Phi_spat = SHTnsSpheres.synthesis_scalar!(tmp.spat.Phi, state.Phi_spec, sph)
    gradPhi_cov = SHTnsSpheres.synthesis_spheroidal!(tmp.spat.gradPhi_cov, state.Phi_spec, sph)

    q_spat = similar!(tmp.spat.q, massq_spat)
    p_hydro = similar!(tmp.p_hydro, massq_spat)
    p_NH = similar!(tmp.p_NH, massq_spat)

    mgr, metric = model.mgr, model.planet.gravity*model.planet.radius^-2
    #   2 - covariant momentum, mass => horizontal momentum, weight
    cov_to_horiz!(uv_spat, mass_spat, q_spat, mgr, metric, gradPhi_cov, W_spat, massq_spat)
    #   3 - geopot => p_NH
    NH_pressure!(p_hydro, p_NH, model.mgr, model.gas, model.vcoord.ptop, mass_spat, q_spat, Phi_spat)

    now = (
        mass = mass_spat, # per unit area, includes gravity
        W = W_spat,       # per steradian
        q = q_spat,
        ux, uy,
        p_NH              # non-hydrostatic pressure
    )

    #   4 - remap NB : inputs (nx, ny, nz) => outputs (nx * ny, nz)
    new, scratch_remapped =
        remap_FCE!(tmp.new, tmp.remapped, model.mgr, model.vcoord, layout, now)

    #   5 - p_NH => Phi, ∇Phi
    NH_geopotential!(flat(Phi_spat, p_hydro), mgr, model.gas, model.vcoord.ptop, new.mass, new.p_NH, new.q)
    SHTnsSpheres.analysis_scalar!(state.Phi_spec, Phi_spat, sph)
    SHTnsSpheres.synthesis_spheroidal!(gradPhi_cov, state.Phi_spec, sph)

    #   6 - horizontal momentum, weight => covariant momentum, mass
    outputs = map(flat, (mass_spat, massq_spat, W_spat, ux, uy))
    horiz_to_cov!(outputs, mgr, inv(metric), new.mass, new.ux, new.uy, new.q, new.W, flat(gradPhi_cov))

    #   7 - spatial => spectral
    SHTnsSpheres.analysis_scalar!(state.mass_air_spec, mass_spat, sph)
    SHTnsSpheres.analysis_scalar!(state.mass_consvar_spec, massq_spat, sph)
    SHTnsSpheres.analysis_scalar!(state.W_spec, W_spat, sph)
    SHTnsSpheres.analysis_vector!(state.uv_spec, uv_spat, sph)

    spat = (; mass=mass_spat, massq=massq_spat, q=q_spat, uv=uv_spat, Phi=Phi_spat, W=W_spat, gradPhi_cov)
    return (; spat, p_hydro, p_NH, new, remapped=scratch_remapped) # == tmp
end

end # module

