module VoronoiDynamics

using CFPlanets: lonlat_from_cov
using CFDomains: Stencils, VoronoiSphere, shell, VHLayout, transpose!

using MutatingOrNot: MutatingOrNot, void, Void, similar!
using ManagedLoops: @with, @vec, @unroll

using ..CFCompressible: FCE
import ..CFCompressible: FCE_tendencies!
using ..CFCompressible.VerticalDynamics: VerticalEnergy, batched_bwd_Euler!, ref_bwd_Euler!

using ..ZeroArrays: ZeroArray

#= Units
[m] = kg
[w] = s             w = g⁻²̇Φ
[W] = kg⋅s             
[p] = kg⋅m⁻¹⋅s⁻²
[ρ] = kg⋅m⁻³
[Jac] = m⋅s²         Jac = a²/g
[Jp] = kg
=#

#=
Computation of tendencies is split into the following steps:
1- Evaluate spatial inputs of HEVI solver: Phiₗ, Wₗ, mₖ, mₗ, sₖ
2- HEVI solver => new spatial values of W, Phi
3- fast tendencies for W, Phi
4- fast tendencies fur u,v
5- new spectral values for u,v
6- slow spectral tendencies for masses (mass budgets) and W, Phi (advection)
7- slow spectral tendencies for u,v (curl form)

Each step has its own additional scratch space for intermediate fields.
In addition, there is shared scratch space for (Phiₗ, Wₗ, mₖ, Sₖ, mₗ, sₖ)
=#

const State = NamedTuple{(:mass_air, :mass_consvar, :ucov, :Phi, :W)}

model_state(mass_air, mass_consvar, ucov, Phi, W) = (; mass_air, mass_consvar, ucov, Phi, W)

function FCE_tendencies!(slow, fast, scratch, model, sph::VoronoiSphere, state::State, tau)
    # layout:
    #   (k,ij), better for horizontal stencils: state.X
    #   (ij,k), better for implicit step:       common.X
    common = spatial_fields!(scratch.common, model, state)
    @info "common" map(size, common)

    Phil_new, Wl_new, tridiag = batched_bwd_Euler!(model, common.ps,
                                                   (common.mk, common.ml, common.Sk,
                                                    common.Phil, common.Wl), tau)
    @info "batched_bwd_Euler!" size(Phil_new) size(Wl_new) map(size, tridiag)

    fast_dPhil, fast_dWl, fast_HV = fast_tendencies_PhiW!(fast.Phil, fast.Wl,
                                                          scratch.fast_HV, model,
                                                          common, Phil_new,
                                                          Wl_new)

    #= let # debug
        dw = fast_spat.dWl./common.ml
        @info "tendencies! with tau = $tau" extrema(common.ps) extrema(common.Phil) extrema(dk(common.Phil))
        # @info "fast" extrema(fast_spat.dPhil) extrema(dw).*model.planet.gravity^2 
    end =#

    # from now on we need to compute horizontal operators
    # this is best done in layout [k,ij]
    # => transpose fields
    fast_VH = fields_VH!(scratch.fast_VH, model.mgr, common, fast_HV)

    fast_duvk, fast_uv = fast_tendencies_ucov!(fast.uvk, scratch.fast_uv, model, state.ucov,
                                               fast_VH.sk, fast_VH.dHdm, fast_VH.dHdS)
    zero_mass = ZeroArray(state.mass_air_spec)
    fast = model_state(zero_mass, zero_mass, fast_duvk, fast_dPhil, fast_dWl) # air, consvar, uv, Phi, W

    new_uvk = (@. scratch.new_uvk = state.uvk + tau*fast_duvk)
    new_state = (; uvk=new_uvk, Phil=Phil_new, Wl=Wl_new) # masses are unchanged

    # step 6
    (dmass_air_spec, dmass_consvar_spec, dW_spec, dPhi_spec), slow_mass = mass_budgets!(slow,
                                                                                        scratch.slow_mass,
                                                                                        model,
                                                                                        sph,
                                                                                        new_state,
                                                                                        common)
    # step 7
    duv_spec, slow_curl_form = curl_form!(slow.uv_spec, scratch.slow_curl_form, model.fcov,
                                          sph, new_state, slow_mass.fluxes, common.mk)

    #= let # debug
        fluxes = slow_mass.fluxes
        # @info "slow" extrema(fluxes.dPhi)
        fliplat(x) = reverse(x; dims=1)
        Linf(x) = maximum(abs,x)
        sym(x, op) = Linf(op(x,fliplat(x)))/Linf(x)
        (; uv, grad_Phi, fluxes) = slow_mass
        @info "symmetry" sym(model.Phis,-) sym(fluxes.B, -) sym(fluxes.dPhi, -) sym(fluxes.Uyk, -) sym(fluxes.wUy,-) sym(fluxes.sUy, -) sym(fast_uv.fy, -) sym(uv.ulon, -) sym(grad_Phi.ulon, -) sym(common.mk, -)  sym(common.ml, -) sym(common.Wl, -)
        @info "antisymmetry" sym(fluxes.Uxk, +) sym(fluxes.wUx, +) sym(fluxes.sUx, +) sym(fast_uv.fx, +) sym(uv.ucolat, +) sym(grad_Phi.ucolat, +)
    end =#

    # Done
    slow = model_state(dmass_air_spec, dmass_consvar_spec, duv_spec, dPhi_spec, dW_spec) # air, consvar, uv, Phi, W
    scratch = (; common, fast_spat, fast_uv, slow_mass, slow_curl_form, Phil_new, Wl_new,
               spheroidal, toroidal)
    return slow, fast, scratch
end

function fields_VH!(scratch, mgr, common, fast_spat)
    sk = transpose!(scratch.sk, mgr, common.sk)  # FIXME: rather state.mass_consvar ./ state.mass_air
    dHdm = transpose!(scratch.dHdm, mgr, fast_spat.dHdm)
    dHdS = transpose!(scratch.dHdS, mgr, fast_spat.dHdS)
    return (; sk, dHdm, dHdS)
end

function spatial_fields!(scratch, model, state)
    (; vcoord, planet, mgr) = model
    ptop, inv_Jac = vcoord.ptop, planet.gravity/planet.radius^2

    # state fields are in (k, ij) layout
    mk = transpose!(scratch.mk, mgr, state.mass_air)
    Sk = transpose!(scratch.Sk, mgr, state.mass_consvar)
    Phil = transpose!(scratch.Phil, mgr, state.Phi)
    Wl = transpose!(scratch.Wl, mgr, state.W)

    # now work in (ij, k) layout
    ps = similar!(scratch.ps, @view mk[:, 1])
    sk = similar!(scratch.sk, Sk)
    ml = similar!(scratch.ml, Phil)

    @with model.mgr let ijrange = axes(mk, 1)
        Nz = size(mk, 3)
        for l in 1:(Nz + 1)
            for ij in ijrange
                if l == 1
                    mm = mk[ij, 1] / 2
                elseif l == Nz + 1
                    mm = mk[ij, Nz] / 2
                else
                    mm = (mk[ij, l - 1] + mk[ij, l]) / 2
                end
                ml[ij, l] = mm
            end
        end
        @vec for ij in ijrange
            ps[ij] = typeof(ps[ij])(ptop)
        end
        for k in 1:Nz
            @vec for ij in ijrange
                ps[ij] += inv_Jac*mk[ij, k]
                sk[ij, k] = Sk[ij, k] / mk[ij, k]
            end
        end
    end # @with

    return (; mk, Sk, Phil, Wl, sk, ml, ps)
end#============= fast tendencies ================#

zero!(x) = @. x=0

function fast_tendencies_PhiW!(dPhil, dWl, scratch, model, common, Phil, Wl)
    # layout is [ij,k]
    (; Phis, rhob) = model # bottom boundary condition p = ps - rhob*(Phi-Phis)
    (; vcoord, planet, gas) = model
    (; mk, sk, ml, ps) = common

    dWl = similar!(scratch.dWl, Wl) # = -dHdPhi
    dPhil = similar!(scratch.dPhil, Phil) # =+dHdW
    dHdm = similar!(scratch.dHdm, mk)
    dHdS = similar!(scratch.dHdS, sk)

    ptop, grav2, Jac = vcoord.ptop, planet.gravity^2, planet.radius^2/planet.gravity

    foreach(zero!, (dWl, dHdm, dHdS))

    # FIXME: rewrite to avoid writing in several passes
    @with model.mgr let ijrange = axes(mk, 1)
        Nz = size(mk, 3)
        for l in 1:(Nz + 1)
            @vec for ij in ijrange
                wm = Wl[ij, l] / ml[ij, l]
                dPhil[ij, l] = grav2 * wm
                l > 1 && (dHdm[ij, l - 1] -= grav2 * wm^2/4)
                l <= Nz && (dHdm[ij, l] -= grav2 * wm^2/4)
            end
        end
        for k in 1:Nz
            @vec for ij in ijrange
                # potential
                dHdm[ij, k] += (Phil[ij, k+1] + Phil[ij, k]) / 2
                dHdm[ij, k] -= Phil[ij, 1]-Phis[ij] # contribution due to elastic bottom BC
                dWl[ij, k] -= mk[ij, k] / 2
                dWl[ij, k+1] -= mk[ij, k] / 2
                # internal
                s = sk[ij, k]
                vol = Jac * (Phil[ij, k+1] - Phil[ij, k]) / mk[ij, k]
                p = gas(:v, :consvar).pressure(vol, s)
                Jp = Jac * p
                dWl[ij, k] -= Jp
                dWl[ij, k+1] += Jp
                h, _, exner = gas(:p, :consvar).exner_functions(p, s)
                dHdm[ij, k] += h - s * exner
                dHdS[ij, k] += exner
            end
        end
        # boundary
        @vec for ij in ijrange
            dWl[ij, Nz+1] -= Jac * ptop
            dWl[ij, 1] -= Jac * (rhob * (Phil[ij, 1] - Phis[ij]) - ps[ij])
        end
    end # @with

    return dPhil, Wl, (; dHdm, dHdS)
end

function fast_tendencies_ucov!(ducov_, scratch, model, ucov, consvar, B, exner)
    # layout is [k, ij]
    ducov = similar!(ducov_, ucov)
    vsphere = model.domain.layer

    #= @with model.mgr, =#
          let (krange, ijrange) = axes(ducov)
              #=@inbounds=# for ij in ijrange
                  grad = Stencils.gradient(vsphere, ij) # covariant gradient
                  avg = Stencils.average_ie(vsphere, ij) # centered average from cells to edges
                  #=@vec=# for k in krange
                      ducov[k, ij] = - muladd(avg(consvar, k), grad(exner, k),
                                              grad(B, k))
                  end
              end
          end
    return ducov
end

# ============= slow tendencies ================ #

# vertical averaging requires (ij,k)
# horizontal stencils (gradient) require (k, ij)

function mass_budgets!(dstate, scratch, model, sph, new_state, common)
    (; mgr, planet), (; laplace) = model, sph      # parameters
    (; mk, ml, sk), (; uv_spec, Phil, Wl) = common, new_state   # inputs

    gradPhi_le = similar!(scratch.gradPhi_le, new_state.ucov) # FIXME: wrong vertical size
    gradcov!(gradPhi_le, model, Phi_li)

    fluxes = NH_fluxes!(scratch.fluxes, mk, sk, uv_ke, gradPhi_le, Wl, ml, mgr, planet)

    # air mass budget
    flux_spec = analysis_vector!(scratch.flux_spec, vector_spat(fluxes.Uxk, fluxes.Uyk),
                                 sph)
    dmass_air_spec = @. dstate.mass_air_spec = -flux_spec.spheroidal * laplace
    # consvar mass budget
    flux_spec = analysis_vector!(scratch.flux_spec,
                                 erase(vector_spat(fluxes.sUx, fluxes.sUy)), sph)
    dmass_consvar_spec = @. dstate.mass_consvar_spec = -flux_spec.spheroidal * laplace
    # W budget
    Wflux_spec = analysis_vector!(scratch.Wflux_spec,
                                  erase(vector_spat(fluxes.wUx, fluxes.wUy)), sph)
    dW_spec = @. dstate.W_spec = -Wflux_spec.spheroidal * laplace
    # Phi tendency
    dPhi_spec = analysis_scalar!(dstate.Phi_spec, erase(fluxes.dPhi), sph)

    return (dmass_air_spec, dmass_consvar_spec, dW_spec, dPhi_spec),
           (; Wl, uv, grad_Phi, fluxes, flux_spec, Wflux_spec, Phi_spec)
end

function gradcov!(gradcov_ke, model, Phi_ki)
    vsphere = model.domain.layer
    @with model.mgr,
          let (krange, ijrange) = axes(gradcov_ke)
              for ij in ijrange
                  grad = Stencils.gradient(vsphere, ij) # covariant gradient
                  @vec for k in krange
                      gradcov_ke[k, ij] = grad(Phi_ki, k)
                  end
              end
          end
    return gradcov_ke
end

function NH_fluxes!(scratch, mk, sk, vx, vy, gx, gy, Wl, ml, mgr, planet)
    # covariant momentum (vx,vy) => contravariant mass flux (Ux,Uy)
    B = similar!(scratch.B, mk)
    Uxk = similar!(scratch.Uxk, vx)
    Uyk = similar!(scratch.Uyk, vy)
    sUx = similar!(scratch.sUx, vx)
    sUy = similar!(scratch.sUy, vy)
    wUx = similar!(scratch.wUx, Wl)
    wUy = similar!(scratch.wUy, Wl)
    dPhi = similar!(scratch.dPhi, Wl)

    factor = planet.radius^-2
    Nz = size(mk, 3)
    @with mgr let (ijrange, krange) = (axes(mk, 1), axes(mk, 2))
        # full levels
        @inbounds for k in 1:Nz
            @vec for ij in ijrange
                wl_d = Wl[ij, k]/ml[ij, k]
                wl_u = Wl[ij, k+1]/ml[ij, k+1]
                # U = a⁻² m (v - W/m ∇Φ), sUx
                U_ek[ij, k] = factor*mk[ij, k]*(vx[ij,
                                                   k]-(wl_d*gx[ij, k]+wl_u*gx[ij, k+1])/2)
                sU_ek[ij, k] = sk[ij, k] * Uxk[ij, k]
            end
        end
        # interfaces
        @inbounds for j in jrange, l in 1:(Nz + 1)
            @vec for ij in ijrange
                if l==1
                    Uxl = Uxk[ij, l]/2
                    Uyl = Uyk[ij, l]/2
                elseif l==Nz+1
                    Uxl = Uxk[ij, l-1]/2
                    Uyl = Uyk[ij, l-1]/2
                else
                    Uxl = (Uxk[ij, l]+Uxk[ij, l-1])/2
                    Uyl = (Uyk[ij, l]+Uyk[ij, l-1])/2
                end
                wl = Wl[ij, l]/ml[ij, l]
                wUx[ij, l], wUy[ij, l] = Uxl * wl, Uyl * wl # wU → ∂ₜW = -∇⋅(wU)
                dPhi[ij, l] = -(Uxl*gx[ij, l]+Uyl*gy[ij, l])/ml[ij, l] # ∂ₜΦ = -u⋅∇Φ
            end
        end
        # full levels again: Bernoulli function dH/dm
        @inbounds for j in jrange, k in 1:Nz
            @vec for ij in ijrange
                X_d = dPhi[ij, k]*(Wl[ij, k]/ml[ij, k])  # (W/m) (-u⋅∇Φ) → m²⋅s⁻²
                X_u = dPhi[ij, k+1]*(Wl[ij, k+1]/ml[ij, k+1])
                K = (Uxk[ij, k]^2+Uyk[ij, k]^2)/(2*factor*mk[ij, k]^2) # a^2 u⋅u/2
                B[ij, k] = K - (X_d+X_u)/2  # u⋅u/2 + (u⋅∇Φ)W/m
            end
        end
    end # @with
    return (; Uxk, Uyk, sUx, sUy, wUx, wUy, dPhi, B)
end

function curl_form!(duv_spec, scratch, fcov, sph, state, fluxes, mk)
    (; Uxk, Uyk, B) = fluxes
    (; laplace) = sph
    zeta_spec = @. scratch.zeta_spec = -laplace * state.uv_spec.toroidal # curl
    zeta = synthesis_scalar!(scratch.zeta, zeta_spec, sph)
    qfx = @. scratch.qfx = Uyk * (zeta + fcov)/mk
    qfy = @. scratch.qfy = -Uxk * (zeta + fcov)/mk
    qflux_spec = analysis_vector!(scratch.qflux_spec, erase(vector_spat(qfx, qfy)), sph)

    B_spec = analysis_scalar!(scratch.B_spec, erase(B), sph)
    duv_spec = vector_spec((@. duv_spec.spheroidal = qflux_spec.spheroidal - B_spec),
                           (@. duv_spec.toroidal = qflux_spec.toroidal))
    return duv_spec, (; zeta_spec, zeta, qfx, qfy, qflux_spec, B_spec)
end#======================== utilities ====================#

vector_spec(spheroidal, toroidal) = (; spheroidal, toroidal)
vector_spat(ucolat, ulon) = (; ucolat, ulon)

end # module
