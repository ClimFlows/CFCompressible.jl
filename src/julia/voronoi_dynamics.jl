 module VoronoiDynamics

using CFPlanets: lonlat_from_cov
using CFDomains: Stencils, VoronoiSphere, shell, VHLayout, transpose!

using MutatingOrNot: MutatingOrNot, void, Void, similar!
using ManagedLoops: @with, @vec, @unroll

using ..CFCompressible: FCE
import ..CFCompressible: FCE_tendencies!
using ..CFCompressible.VerticalDynamics: VerticalEnergy, batched_bwd_Euler!, ref_bwd_Euler!

using ..ZeroArrays: ZeroArray

struct Lazy; end
@inline Base.materialize!(::Lazy, rhs::Broadcast.Broadcasted) = BCArray(rhs)

"""
    z = @.. x*y

    Constructs the lazy array-like `z` such that 
        z[i, ...] == x[i, ...]*z[i, ...]
"""
macro (>)(expr)
    lazy = Lazy()
    return esc(:( @. $lazy = $expr ))
end

struct BCArray{T,N,B} <: AbstractArray{T,N}
    bc::B
end
# @adapt_structure BCArray

function BCArray(rhs::B) where {N, B<:Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{N}}}
    T = Base.Broadcast.combine_eltypes(rhs.f, rhs.args)
    return BCArray{T,N,B}(rhs)
end

@inline Base.axes(x::BCArray) = axes(x.bc)
@inline Base.size(x::BCArray) = size(x.bc)
@inline Base.eltype(::BCArray{T}) where T = T
@inline Base.similar(x::BCArray{T}) where T = similar(x.bc, T, size(x.bc))

Base.@propagate_inbounds Base.getindex(x::BCArray, i...) = getindex(x.bc, i...)

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
    common = spatial_fields!(scratch.common, model, state) # mk, Sk, Phil, ps, sk, ml

    new_Phil, new_Wl, tridiag = batched_bwd_Euler!(model, common.ps,
                                                   (common.mk, common.ml, common.Sk,
                                                    common.Phil, common.Wl), tau)
    dPhil, dWl, fast_HV = fast_tendencies_PhiW!(scratch.dPhil, scratch.dWl,
                                                          scratch.fast_HV, model,
                                                          common, new_Phil,
                                                          new_Wl)
    Phil = transpose!(scratch.Phil, model.mgr, new_Phil)
    Wl = transpose!(scratch.Wl, model.mgr, new_Wl)
    fast_dPhil = transpose!(fast.Phil, model.mgr, dPhil)
    fast_dWl = transpose!(fast.Wl, model.mgr, dWl)

    # from now on we need to compute horizontal operators
    # this is best done in layout [k,ij] => transpose fields
    fast_VH = fields_VH!(scratch.fast_VH, model.mgr, common, fast_HV)

    fast_ducov, fast_ucov = fast_tendencies_ucov!(fast.ucov, scratch.fast_ucov, model, state.ucov,
                                               fast_VH.sk, fast_VH.dHdm, fast_VH.dHdS)
    zero_mass = ZeroArray(state.mass_air)
    fast = model_state(zero_mass, zero_mass, fast_ducov, fast_dPhil, fast_dWl) # air, consvar, uv, Phi, W

    new_ucov = (@. scratch.new_ucov = state.ucov + tau*fast_ducov)
    new_state = (; mk=state.mass_air, sk=fast_VH.sk, invml=fast_VH.invml, ucov=new_ucov, Phil, Wl) # masses are unchanged

    let # debug
        @info "common" map(size, common)
        @info "batched_bwd_Euler!" size(new_Phil) size(new_Wl) map(size, tridiag)
        @info "tendencies! with tau = $tau" extrema(common.ps) extrema(common.Phil)
        dw = dWl./common.ml
        @info "fast" extrema(fast_dPhil) extrema(dw).*model.planet.gravity^2 
        @info "new_state" map(size, new_state)
    end

    # step 6
    (dmass_air, dmass_consvar, dW, dPhi), slow_mass = mass_budgets!(slow,
                                                                    scratch.slow_mass,
                                                                    model,
                                                                    new_state)
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
        Nz = size(mk, 2)
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
        for ij in ijrange
            ps[ij] = ptop
        end
        for k in 1:Nz
            @vec for ij in ijrange
                ps[ij] += inv_Jac*mk[ij, k]
                sk[ij, k] = Sk[ij, k] / mk[ij, k]
            end
        end
    end # @with
    @info "spatial_fields!" extrema(inv_Jac*mk) extrema(ps) extrema(sum(state.mass_air; dims=1)*inv_Jac) extrema(sum(mk; dims=2)*inv_Jac)

    return (; mk, Sk, Phil, Wl, sk, ml, ps)
end

function fields_VH!(scratch, mgr, common, fast_spat)
    sk = transpose!(scratch.sk, mgr, common.sk)
    dHdm = transpose!(scratch.dHdm, mgr, fast_spat.dHdm)
    dHdS = transpose!(scratch.dHdS, mgr, fast_spat.dHdS)
    invml = transpose!(scratch.invml, mgr, common.ml)
    @. invml = inv(invml)
    return (; sk, invml, dHdm, dHdS)
end

#============= fast tendencies ================#

zero!(x) = @. x=0

function fast_tendencies_PhiW!(dPhil, dWl, scratch, model, common, Phil, Wl)
    @assert axes(Wl) == axes(Phil)
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
        Nz = size(mk, 2)
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

    #=@with model.mgr, =#
    let (krange, ijrange) = axes(ducov)
        #=@inbounds=# for ij in ijrange
            grad = Stencils.gradient(vsphere, ij) # covariant gradient
            avg = Stencils.average_ie(vsphere, ij) # centered average from cells to edges
            #=@vec=# for k in krange
                ducov[k, ij] = - muladd(avg(consvar, k), grad(exner, k), grad(B, k))
            end
        end
    end
    return ducov
end

# ============= slow tendencies ================ #

# vertical averaging requires (ij,k)
# horizontal stencils (gradient) require (k, ij)

function mass_budgets!(dstate, scratch, model, new_state)
    # layout is [k,ij]
    (; mgr, planet) = model      # parameters
    factor = planet.radius^-2
    vsphere = model.domain.layer
    (; invml, mk, sk, ucov, Phil, Wl) = new_state # inputs
    Nz = size(mk, 1)

    # vsphere_grad = merge(Stencils.gradient(vsphere), Stencils.average_ie(vsphere))

    wl = wl!(similar!(scratch.wl, Wl), mgr, invml, Wl)
    U_ke, sU_ke = sU_ke!(similar!(scratch.U_ke, ucov), similar!(scratch.sU_ke, ucov), mgr, vsphere, factor, wl, Phil, mk, ucov, sk)
    U_le = U_le!(similar!(scratch.U_le, ucov, Nz+1, size(ucov,2)), mgr, U_ke)
    wU, ∇Φ = wU_gradPhi!(similar!(scratch.wU, U_le), similar!(scratch.∇Φ, U_le), mgr, vsphere, wl, U_le, Phil)

    dPhi = dPhi_dt!(similar!(dstate.Phil, Phil), mgr, vsphere, invml, U_le, ∇Φ)
    B = Bernoulli!(similar!(scratch.B, mk), mgr, vsphere, factor, mk, U_ke, wl, dPhi)

    dmass = dmass!(similar!(dstate.mass, mk), mgr, vsphere, U_ke)
    dmass_consvar = dmass!(similar!(dstate.mass_consvar, mk), mgr, vsphere, sU_ke)
    dW = dmass!(similar!(dstate.Wl, Wl), mgr, vsphere, wU)

    return (dmass, dmass_consvar, dPhi, dW), (; U_ke, sU_ke, wl, U_le, wU, ∇Φ, B)
end

function wl!(wl, mgr, invml, Wl)
    @with mgr let (lrange, cells) = axes(Wl)
        @vec for l in lrange, ij in cells            
            wl[l,ij] = invml[l,ij]*Wl[l,ij]
        end
    end
    return wl
end

function sU_ke!(U_ke, sU_ke, mgr, vsphere, factor, wl, Phil, mk, ucov, sk) # contravariant fluxes of mass and conservative variable
    le_de = vsphere.le_de
    @with mgr let (krange, edges) = axes(U_ke)
        for ij in edges
            grad = Stencils.gradient(vsphere, ij) # covariant gradient
            avg_ie = Stencils.average_ie(vsphere, ij) # centered average from cells to edges
            cov_to_contra = factor*le_de[ij]
            @vec for k in krange
                w∇Φ = (avg_ie(wl, k)*grad(Phil, k) + avg_ie(wl, k+1)*grad(Phil, k+1))/2
                U_ke[k, ij] = cov_to_contra*avg_ie(mk, k)*(ucov[k,ij]-w∇Φ)
                sU_ke[k,ij] = avg_ie(sk, k) * U_ke[k, ij]
            end
        end
    end
    return U_ke, sU_ke
end

function U_le!(U_le, mgr, U_ke) # contravariant mass flux at dual vertical cells
    Nz = size(U_ke, 1)
    # top and bottom interfaces
    @with mgr let edges = axes(U_le,2)
        for ij in edges
            U_le[1,ij] = U_ke[1, ij]/2
            U_le[Nz+1,ij] = U_ke[Nz, ij]/2
        end
    end
    # interior interfaces
    @with mgr let (lrange, edges) = (2:Nz, axes(U_le,2))
        for ij in edges
            @vec for l in lrange
                U_le[l,ij] = (U_ke[l, ij]+U_ke[l-1,ij])/2
            end
        end
    end
    return U_le
end

function wU_gradPhi!(wU, ∇Φ, mgr, vsphere, wl, U_le, Phil)
    @with mgr let (lrange, edges) = axes(wU)
        for ij in edges
            avg_ie = Stencils.average_ie(vsphere, ij) # centered average from cells to edges
            grad = Stencils.gradient(vsphere, ij) # covariant gradient
            @vec for l in lrange
                wU[l,ij] = avg_ie(wl, l)*U_le[l,ij]
                ∇Φ[l,ij] = grad(Phil, l)
            end
        end
    end
    return wU, ∇Φ
end

function dPhi_dt!(dPhi, mgr, vsphere, invml, U_le, ∇Φ) # ∂ₜΦ = -u⋅∇Φ 
    vsphere_dp = Stencils.dot_product(vsphere)
    degree = vsphere.primal_deg
    @with mgr let (lrange, cells) = axes(dPhi)
        for ij in cells
            deg = degree[ij]
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(vsphere_dp, ij, Val(deg))
                @vec for l in lrange
                    dPhi[l, ij] = -invml[l, ij] * dot_product(U_le, ∇Φ, l) # FIXME: no need for le_de
                end
            end
        end
    end
    return dPhi
end

function Bernoulli!(B, mgr, vsphere, factor, mk, U_ke, wl, dPhi) # Bernoulli function B = u⋅u/2 + (u⋅∇Φ)W/m
    vsphere_dp = Stencils.dot_product(vsphere)
    degree = vsphere.primal_deg
    @with mgr let (krange, cells) = axes(B)
        for ij in cells
            deg = degree[ij]
            @unroll deg in 5:7 begin
                dot_product = Stencils.dot_product(vsphere_dp, ij, Val(deg))
                @vec for k in krange 
                    K = dot_product(U_ke, U_ke, k)/(2*factor*mk[k,ij]^2) # a^2 u⋅u/2
                    B[k, ij] = K - (wl[k,ij]*dPhi[k,ij] + wl[k+1,ij]*dPhi[k+1,ij])/2
                end
            end
        end
    end
    return B
end

function dmass!(dmass, mgr, vsphere, U) # ∂ₜm = -∇⋅U 
    degree = vsphere.primal_deg
    @with mgr let (krange, cells) = axes(dmass)
        for ij in cells
            deg = degree[ij]
            @unroll deg in 5:7 begin
                dvg = Stencils.divergence(vsphere, ij, Val(deg))
                @vec for k in krange
                    dmass[k, ij] = -dvg(U,k) # FIXME: no need for le_de
                end
            end
        end
    end
    return dmass
end

end # module
