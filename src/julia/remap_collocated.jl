module RemapCollocated

using CFHydrostatics.RemapHPE: vanleer, remap_density!, remap_scalar!, update_mass!, flatten
using CFDomains: mass_coordinate, data_layout
using CFTransport: remap_fluxes!, mass_flux_dual!
using MutatingOrNot: similar!
using ManagedLoops: @with, @vec

# these functionas are called from RemapSpectral
# here we put everything that does not require SHTnsSpheres

function cov_to_horiz!(uv, mass, q, mgr, metric, gradPhi, W, massq)
    (; ucolat, ulon) = uv
    Phi_colat, Phi_lon = gradPhi
    @with mgr let (irange, jrange) = (axes(ucolat, 1), axes(ucolat, 2))
        krange = axes(ucolat, 3)
        # horizontal NH momentum (covariant)
        for (ui, Phi_i) in ((ucolat, Phi_colat), (ulon, Phi_lon))
            for j in jrange, k in krange
                @vec for i in irange
                    ui[i, j, k] -= (Phi_i[i, j, k] * W[i, j, k] +
                                   Phi_i[i, j, k + 1] * W[i, j, k + 1]) /
                                   (2 * mass[i, j, k])
                end # i
            end # j,k
        end # colat, lon
        # compute q, apply metric factor to mass
        for j in jrange, k in krange
            @vec for i in irange
                q[i, j, k] = massq[i, j, k]/mass[i, j ,k]
                mass[i, j, k] *= metric
            end
        end
    end # @with
    return nothing
end

function NH_pressure!(p_hydro, p_NH, mgr, gas, ptop, mass, consvar, Phi)
    pressure = gas(:v, :consvar).pressure 
    @with mgr let (irange, jrange) = (axes(p_NH, 1), axes(p_NH, 2))
        nz = size(p_NH, 3)
        for j in jrange
            # mass is per unit area, includes gravity => same unit as pressure, as in HPE
            let k=nz
                @vec for i in irange
                    vol = (Phi[i,j,k+1]-Phi[i,j,k])/mass[i,j,k]
                    p_hydro[i, j, nz] = ptop + mass[i, j, nz] / 2
                    p_NH[i,j,k] = pressure(vol, consvar[i,j,k]) - p_hydro[i,j,k]
                end
            end
            for k in nz-1:-1:1
                @vec for i in irange
                    vol = (Phi[i,j,k+1]-Phi[i,j,k])/(mass[i,j,k])                   
                    p_hydro[i, j, k] = p_hydro[i, j, k+1] + (mass[i, j, k] + mass[i, j, k+1])/2
                    p_NH[i,j,k] = pressure(vol, consvar[i,j,k]) - p_hydro[i,j,k]
                end
            end
        end
    end
    return nothing
end

function remap_FCE!(new, tmp, mgr, vcoord, layout, now, schemes=(scalar=vanleer, momentum=vanleer))
    (; mass, W, q, ux, uy, p_NH) = map( x->flatten(x, layout), now)
    momentum = schemes.momentum(:scalar, layout)
    scalar = schemes.scalar(:scalar, layout)
    density = schemes.scalar(:density, layout)
    # mass fluxes and new mass
    mcoord = mass_coordinate(vcoord, one(eltype(mass)))
    flux, new_mass = remap_fluxes!(mgr, mcoord, flatten(layout), tmp.flux, tmp.new_mass, #==# mass)
    # scalars
    fluxq = similar!(tmp.fluxq, flux)
    slope = similar!(tmp.slope, q)
    new_q = remap_scalar!(mgr, scalar, new.q, #==# fluxq, slope, #==# q, mass, flux)
    new_p_NH = remap_scalar!(mgr, scalar, new.p_NH, #==# fluxq, slope, #==# p_NH, mass, flux)
    new_ux = remap_scalar!(mgr, momentum, new.ux, #==# fluxq, slope, #==# ux, mass, flux)
    new_uy = remap_scalar!(mgr, momentum, new.uy, #==# fluxq, slope, #==# uy, mass, flux)
    # densities
    mass_dual, flux_dual = mass_flux_dual!(tmp.mass_dual, tmp.flux_dual, mgr, flatten(layout), mass, flux)
    w = similar!(tmp.w, W)
    slopeW = similar!(tmp.slopeW, W)
    fluxW = similar!(tmp.fluxW, flux_dual)
    # new_massq = remap_density!(mgr, scheme_mq, new.massq, #==# fluxq, slope, q, #==# massq, mass, flux)
    new_W = remap_density!(mgr, density, new.W, #==# fluxW, slopeW, w, #==# W, mass_dual, flux_dual)
    new_mass = update_mass!(mgr, new.mass, #==# new_mass)
    # return
    tmp = (; flux, new_mass, fluxq, slope, mass_dual, flux_dual, fluxW, slopeW, w)
    return (mass=new_mass, W=new_W, q=new_q, ux=new_ux, uy=new_uy, p_NH=new_p_NH), tmp
end

# p_NH => geopot
function NH_geopotential!((Phi, p_hydro), mgr, gas, ptop, mass, p_NH, consvar)
    volume = gas(:p, :consvar).specific_volume
    nz = size(p_NH, 2)
    @with mgr let irange=axes(p_NH, 1)
        # mass is per unit area, includes gravity => same unit as pressure, as in HPE
        @vec for i in irange
            p_hydro[i, nz] = ptop + mass[i, nz] / 2
        end
        for k in nz-1:-1:1
            @vec for i in irange
                p_hydro[i, k] = p_hydro[i, k+1] + (mass[i, k] + mass[i, k+1])/2
            end
        end
        # Phi[:,1] is already set
        for k in axes(mass,2)
            @vec for i in irange
                vol = volume(p_hydro[i,k] + p_NH[i,k], consvar[i,k])
                Phi[i,k+1] = Phi[i,k] + mass[i,k]*vol
            end
        end
    end
    return nothing
end

# horizontal momentum, weight => covariant momentum, mass
function horiz_to_cov!((mass, massq, W, ux, uy), mgr, metric, new_mass, new_ux, new_uy, new_q, new_W, (Phi_x, Phi_y))
    @with mgr let (irange, krange) = axes(W)
        for i in irange, k in krange
            W[i,k] = new_W[i,k]
        end
    end
    @with mgr let (irange, krange) = axes(mass)
        for k in krange
            for i in irange 
                mass[i,k] = metric*new_mass[i,k]
                massq[i,k] = mass[i,k] * new_q[i,k]
            end 
            for (ui, Phi_i) in ((ux, Phi_x), (uy, Phi_y))
                for i in irange
                    ui[i, k] += (Phi_i[i, k] * W[i, k] +
                                Phi_i[i, k + 1] * W[i, k + 1]) /
                               (2 * mass[i, k])
                end
            end # (ux,uy)
        end # k
    end # let
end

#=

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
=#

end # module
