module RemapCollocated

using CFTransport: remap_fluxes!
using CFDomains: mass_coordinate
using CFHydrostatics.RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!, flatten

using ..CFCompressible.Dynamics: scalar_fields!

function remap_spectral!(new_state, tmp, model, state)
    # transforms prognostic fields into remap-ready fields
    fields = scalar_fields!(tmp.fields, model, model.domain.layer)
    (vx, vy) = v = synthesis_vector!(tmp.v, state.uv_spec, sph)
    (gx, gy) = gradPhi = synthesis_spheroidal!(tmp.gradPhi, state.Phi_spec, sph)
    ux, uy = eulerian_momentum!(tmp.ux, tmp.uy, mgr, vx, vy, gx, gy, fields.Wl, fields.ml)
    NHvol, _ = NH_volume!(tmp.NHvol, fields.ps, model, fields.mk, fields.sk, fields.Phil)

    # mass is in kg while hybrid coefficients are in Pa => Jacobian to convert Pa into kg
    (; mgr, planet, vcoord) = model
    Jac = planet.radius^2/planet.gravity 
    schemes = (; scalar=vanleer, momentum=vanleer)
    now = (; mass=fields.mk, q=fields.sk, NHvol, ux, uy)
    new = remap!(tmp.new, tmp.remap, mgr, vcoord, Jac, layout, schemes, now)
    # NH volume => geopotential

    tmp = (; fields, v, gradPhi, Ux, Uy, vol)
    return new_state, tmp
end

function eulerian_momentum!(Ux_, Uy_, mgr, vx, vy, gx, gy, Wl, ml)
    # covariant momentum (vx,vy,_) w.r.t. (x,y,η)
    # => covariant momentum (ux,uy,w) w.r.t. (x,y,Φ)
    ux = similar!(Ux_, vx)
    uy = similar!(Uy_, vy)

    Nz = size(vx, 3)
    @with mgr let (irange, jrange) = (axes(ml,1), axes(ml, 2))
        #=@inbounds=# for j in jrange, k in 1:Nz
            #=@vec=# for i in irange
                wl_d = Wl[i,j,k]/ml[i,j,k]
                wl_u = Wl[i,j,k+1]/ml[i,j,k+1]
                # U = a⁻² m (v - W/m ∇Φ), sUx
                ux[i,j,k] = vx[i,j,k]-(wl_d*gx[i,j,k]+wl_u*gx[i,j,k+1])/2
                uy[i,j,k] = vy[i,j,k]-(wl_d*gy[i,j,k]+wl_u*gy[i,j,k+1])/2
            end
        end
    end
    return ux, uy
end

function NH_volume!(NHvol_, ps_, model, mk, sk, Phil)
    (; mgr, planet, gas) = model
    ptop, Jac = vcoord.ptop, planet.radius^2/planet.gravity

    NHvol = similar!(NHvol_, mk)
    ps = similar!(ps_, mk, size(mk,1), size(mk,2))

    @with mgr  let (irange, jrange) = (axes(mk,1), axes(mk, 2))
        Nz = size(mk, 3)
        #=@inbounds=# for i in irange, j in jrange   
            ps[i,j] = ptop
        end
        #=@inbounds=# for j in jrange, k in Nz:-1:1
            #=@vec=# for i in irange
                dp = inv_Jac*mk[i,j,k]
                pmid = ps[i,j] + dp/2
                v = gas(:p, :consvar).volume(pmid, sk[i,j,k])
                NHvol[i,j,k] = Jac*(Phil[i,j,k+1]-Phil[i,j,k])/mk[i,j,k] - v
                ps[i,j] += dp
            end
        end
    end
end

function remap!(new, tmp, mgr, vcoord, Jac, layout, schemes, now)
    (; mass, q, NHvol, ux, uy) = map( x->flatten(x, layout), now)
    scheme_q = schemes.scalar(:scalar, layout)
    scheme_u = schemes.momentum(:scalar, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, Jac)
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), #==# tmp.flux, tmp.new_mass, #==# mass)
    # vertical transport
    new_q = remap_scalar!(mgr, scheme_q, new.q, #==# tmp.fluxq, tmp.slope, #==# q, mass, flux)
    new_NHvol = remap_scalar!(mgr, scheme_q, new.NHvol, #==# tmp.fluxq, tmp.slope, #==# NHvol, mass, flux)
    new_ux = remap_scalar!(mgr, scheme_u, new.ux, #==# tmp.fluxq, tmp.slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, scheme_u, new.uy, #==# tmp.fluxq, tmp.slope, #==# uy, mass, flux)
    new = (; mass=new_mass, q=new_q, NHvol=new_NHvol, ux=new_ux, uy=new_uy)
end

end # module
