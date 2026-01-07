module BoxDynamics

using ..CFCompressible: FC2D
using CFPlanets: Tank2D
using CFBoxes
using CFDiffusionSchemes
using ManagedLoops: @vec, @with
using MutatingOrNot: Void, void, similar!


## TENDENCIES

function FC2D_tendencies!(dstate, scratch, model::FC2D, state, t)
    # compute pressure, conservative variable, composition, temperature
    (; p, consvar, comp, T) = scratch.thermo
    p, consvar, comp, T = p_cons_comp_T!((p, consvar, comp, T), model, state)

    # apply boundary conditions to state (and consvar, comp, T)
    state = state_bc!(model, state, (p, consvar, comp, T))

    ## VELOCITY DERIVATIVES; ∂u_x, ∂w_z, ∂u/∂z, ∂w/∂x
    (; ∂u_x, ∂w_z, ∂u_z, ∂w_x) = scratch.derivatives

    ∂u_x, ∂w_z  = grad_uw!((∂u_x, ∂w_z), model, state)
    ∂u_z, ∂w_x  = skew_uw!((∂u_z, ∂w_x), model, state)

    ## ADVECTION; ω×u = (ωw, -ωu), K = (u² + w²) /2
    (; K, ∂K_x, ∂K_z, ωw, ωu, adv_scratch, mu, mw) = scratch.advection

    K, ∂K_x, ∂K_z       = ke_adv!((K, ∂K_x, ∂K_z), model, state)
    mu, mw              = advective_flux!((mu, mw), model, state)
    ωw, ωu, adv_scratch = vort_adv!((ωw, ωu, adv_scratch), model, model.advection, state, (∂u_z, ∂w_x, mu, mw))

    ## VISCOUS DISSIPATION
    (; ρε, visc_scratch) = scratch.irreversible

    ρε, visc_scratch = viscous_dissipation!((ρε, visc_scratch), model, state, (∂u_x, ∂w_z, ∂u_z, ∂w_x))

    ## THERMODYNAMICS
    (; conjvar, chempot, Cp, ∂cons_∂q, Γ_pq) = scratch.thermo

    conjvar, chempot, Cp, ∂cons_∂q, Γ_pq = thermodynamics!((conjvar, chempot, Cp, ∂cons_∂q, Γ_pq), model, state, (consvar, comp, p, T))
    
    ## MOMENTUM FORCING ("buoyancy")
    (; B_x, B_z, B) = scratch.momentum

    B_x, B_z, B = buoyancy!((B_x, B_z, B), model, state, (consvar, comp, conjvar, chempot, p))
    
    ## HEAT & COMPOSITION FLUX
    (; Jcons_x, Jcons_z, Jq_x, Jq_z)  = scratch.irreversible

    Jcons_x, Jcons_z, Jq_x, Jq_z = fluxes!((Jcons_x, Jcons_z, Jq_x, Jq_z), model, model.heatflux, state, (consvar, comp, conjvar, p, T, Cp, ∂cons_∂q, chempot, Γ_pq))

    # ENTROPY PRODUCTION
    (; σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z) = scratch.irreversible

    σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z = entropy_production!((σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z), model, model.heatflux, state, (conjvar, chempot, Jcons_x, Jcons_z, Jq_x, Jq_z, ρε))

    # UPDATE DSTATE
    dm, du, dw  = (dstate.m, dstate.u, dstate.w)

    du, dw  = duw!((du, dw), model, state, (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, ∂u_x, ∂w_z, ∂u_z, ∂w_x))
    dm      = dm!(dm, model, state, (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons))

    dstate = (m = dm, u = du, w = dw)
   
    # apply boundary conditions to dstate
    dstate = dstate_bc!(model, dstate)


    scratch = (
        momentum        = (; B_x, B_z, B),
        thermo          = (; consvar, comp, conjvar, chempot, p, T, Cp, ∂cons_∂q, Γ_pq),
        derivatives     = (; ∂u_x, ∂w_z, ∂u_z, ∂w_x),
        irreversible    = (; ρε, visc_scratch, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z),
        advection       = (; mu, mw, K, ∂K_x, ∂K_z, ωw, ωu, adv_scratch),
    )
    return dstate, scratch
end

## PRESSURE, CONSERVATIVE VARIABLE, COMPOSITION, TEMPERATURE

function p_cons_comp_T!((p_, consvar_, comp_, T_), model::FC2D, (; m))
    (; mgr, fluid) = model

    p       = similar!(p_, m, size(m)[1:2]...)
    consvar = similar!(consvar_, m, size(m)[1:2]...)
    comp    = similar!(comp_,    m, size(m)[1:2]...)
    T       = similar!(T_,       m, size(m)[1:2]...)

    @with mgr, let (irange, jrange) = (axes(consvar, 1), axes(consvar, 2))
        @vec for i in irange, j in jrange
            v             = inv(m[i, j, 1])
            consvar[i, j] = m[i, j, 2] * v
            comp[i, j]    = m[i, j, 3] * v
            p[i, j]       = fluid(:v, :consvar, :q).pressure(v, consvar[i, j], comp[i, j])
            T[i, j]       = fluid(:p, :consvar, :q).temperature(p[i, j], consvar[i, j], comp[i, j])
        end
    end
    return p, consvar, comp, T
end

## VELOCITY DERIVATIVES

# skew derivatives : (u, w) at edges ↦ (∂u/∂z, ∂w/∂x) at vertices
function skew_uw!((∂u_z_, ∂w_x_), model::FC2D{F}, (; u, w)) where F
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_z = similar!(∂u_z_, Array{F}, Xdim(domain), Zdim(domain))
    ∂w_x = similar!(∂w_x_, Array{F}, Xdim(domain), Zdim(domain))
    
    @with mgr, let (irange, jrange) = (axes(∂u_z, 1), Zrange(domain))
        @vec for I in irange, J in jrange
            ∂u_z[I, J] = inv_dz * dif_Z(u, I, J)
        end
    end
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂w_x, 2))
        @vec for I in irange, J in jrange
            ∂w_x[I, J] = inv_dx * dif_X(w, I, J)
        end
    end

    periodize!(model, (∂u_z, ∂w_x))

    return ∂u_z, ∂w_x
end
# gradient of edge quantities : (u, w) at edges ↦ (∂u/∂x, ∂w/∂z) at centres
function grad_uw!((∂u_x_, ∂w_z_), model::FC2D, (; m, u, w))
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_x = similar!(∂u_x_, m, size(m)[1:2]...)
    ∂w_z = similar!(∂w_z_, m, size(m)[1:2]...)

    @with mgr, let (irange, jrange) = (xrange(domain), axes(∂u_x, 2))
        @vec for i in irange, j in jrange
            ∂u_x[i, j] = inv_dx * dif_x(u, i, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(∂w_z, 1), zrange(domain))
        @vec for i in irange, j in jrange
            ∂w_z[i, j] = inv_dz * dif_z(w, i, j)
        end
    end

    periodize!(model, (∂u_x, ∂w_z))

    return ∂u_x, ∂w_z
end

## ADVECTION

# vorticity term for advection
function vort_adv!((ωw_, ωu_, (ωwᴵᴶ_, ωuᴵᴶ_)), model::FC2D, advection::AdvEnergyCons, (; m, u, w), (∂u_z, ∂w_x, mu, mw))
    (; mgr, domain) = model

    ωw      = similar!(ωw_, u)
    ωu      = similar!(ωu_, w)
    ωwᴵᴶ    = similar!(ωwᴵᴶ_, ∂u_z)
    ωuᴵᴶ    = similar!(ωuᴵᴶ_, ∂u_z)
    
    mu1 = @view mu[:, :, 1]
    mw1 = @view mw[:, :, 1]
    m1  = @view m[:, :, 1]
    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωwᴵᴶ[I, J] = ω * avg_X(mw1, I, J) * inv(avg_XZ(m1, I, J))
            ωuᴵᴶ[I, J] = ω * avg_Z(mu1, I, J) * inv(avg_XZ(m1, I, J))
        end
    end
    periodize!(model, (ωwᴵᴶ, ωuᴵᴶ))

    # ωwⁱᴶ
    @with mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωwᴵᴶ, I, j)
        end
    end

    # ωuᴵʲ
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωuᴵᴶ, i, J)
        end
    end
    periodize!(model, (ωw, ωu))
    return ωw, ωu, (ωwᴵᴶ, ωuᴵᴶ)
end
function vort_adv!((ωw_, ωu_, (ωvᴵᴶ_, ρwᴵ_, ρuᴶ_)), model::FC2D, advection::AdvEnstrophyCons, (; u, w), (∂u_z, ∂w_x))
    (; mgr, domain) = model

    ωw      = similar!(ωw_, u)
    ωu      = similar!(ωu_, w)

    ωvᴵᴶ    = similar!(ωvᴵᴶ_, ∂u_z)
    ρwᴵ     = similar!(ρwᴵ_, ∂u_z)
    ρuᴶ     = similar!(ρuᴶ_, ∂u_z)

    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωvᴵᴶ[I, J] = @views ω / avg_XJ(m[:, :, 1], I, J)
            ρwᴵ[I, J] = @views avg_X(mw[:, :, 1], I, J)
            ρuᴶ[I, J] = @views avg_Z(mu[:, :, 1], I, J)
        end
    end
    periodize!(model, (ωvᴵᴶ, ρuᴶ, ρwᴵ))

    # ωwⁱᴶ
    @with model.mgr, let (irange, jrange) = (axes(ωw, 1), zrange(domain))
        @vec for I in irange, j in jrange
            ωw[I, j] = avg_z(ωvᴵᴶ, I, j) * avg_z(ρwᴵ, I, j)
        end
    end

    # ωuᴵʲ
    @with mgr, let (irange, jrange) = (xrange(domain), axes(ωu, 2))
        @vec for i in irange, J in jrange
            ωu[i, J] = avg_x(ωvᴵᴶ, i, J) * avg_x(ρuᴶ, i, J)
        end
    end    
    periodize!(model, (ωw, ωu))
    return ωw, ωu, (ωvᴵᴶ, ρwᴵ, ρuᴶ)
end

# kinetic energy term for advection
function ke_adv!((K_, ∂K_x_, ∂K_z_), model::FC2D, (; m, u, w))
    (; mgr, domain, inv_dx, inv_dz) = model

    K       = similar!(K_, m, size(m)[1:2]...)
    ∂K_x    = similar!(∂K_x_, u)
    ∂K_z    = similar!(∂K_z_, w)

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            K[i, j] = 0.5 * ( avg_x(abs2, u, i, j) + avg_z(abs2, w, i, j) )
        end
    end
    periodize!(model, K)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(∂K_x, 2))
        @vec for I in irange, j in jrange
            ∂K_x[I, j] = inv_dx * dif_X(K, I, j)
        end
    end    
    @with mgr, let (irange, jrange) = (axes(∂K_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ∂K_z[i, J] = inv_dz * dif_Z(K, i, J)
        end
    end
    periodize!(model, (∂K_x, ∂K_z))
    return K, ∂K_x, ∂K_z
end

# flux for tracer advection
function advective_flux!((mu_, mw_), model::FC2D, (; m, u, w))
    (; mgr, domain) = model

    mu = similar!(mu_, u, size(u)..., 3)
    mw = similar!(mw_, w, size(w)..., 3)

    m1 = @view m[:, :, 1]
    m2 = @view m[:, :, 2]
    m3 = @view m[:, :, 3]
    mu1 = @view mu[:, :, 1]
    mu2 = @view mu[:, :, 2]
    mu3 = @view mu[:, :, 3]
    mw1 = @view mw[:, :, 1]
    mw2 = @view mw[:, :, 2]
    mw3 = @view mw[:, :, 3]

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(mu, 2))
        @vec for I in irange, j in jrange
            mu1[I, j] = u[I, j] * avg_X(m1, I, j)
            mu2[I, j] = u[I, j] * avg_X(m2, I, j)
            mu3[I, j] = u[I, j] * avg_X(m3, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(mu, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            mw1[i, J] = w[i, J] * avg_Z(m1, i, J)
            mw2[i, J] = w[i, J] * avg_Z(m2, i, J)
            mw3[i, J] = w[i, J] * avg_Z(m3, i, J)
        end
    end
    periodize!(model, (mu1, mu2, mu3, mw1, mw2, mw3))
    return mu, mw
end

## VISCOUS DISSIPATION

function viscous_dissipation!((ρε_, (uτ_x_, uτ_z_, u∂τ_x_, u∂τ_z_)), model, (; m, u, w), (∂u_x, ∂w_z, ∂u_z, ∂w_x))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity

    ρε      = similar!(ρε_, m, size(m)[1:2]...)
    uτ_x    = similar!(uτ_x_, u)
    uτ_z    = similar!(uτ_z_, w)
    u∂τ_x   = similar!(u∂τ_x_, u)
    u∂τ_z   = similar!(u∂τ_z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            divu_x = avg_X(∂u_x, I, j) + avg_X(∂w_z, I, j)
            ∂divu_x = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))
            ∂ω_z  = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))
            # term inside gradient
            uτ_x[I, j] = ( u[I, j] * ( 2 * dyn_visc * avg_X(∂u_x, I, j) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_x )
                    + dyn_visc * avg_Xz(w, I, j) * ( avg_z(∂u_z, I, j) + avg_z(∂w_x, I, j) ) )
            # term inside average
            u∂τ_x[I, j] = ( u[I, j] * ( - dyn_visc * ∂ω_z - ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_x ) )
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange
            divu_z = avg_Z(∂u_x, i, J) + avg_Z(∂w_z, i, J)
            ∂divu_z = inv_dz * (dif_Z(∂u_x, i, J) + dif_Z(∂w_z, i, J))
            ∂ω_x  = inv_dx * (dif_x(∂u_z, i, J) - dif_x(∂w_x, i, J))
            # term inside gradient
            uτ_z[i, J] = ( w[i, J] * ( 2 * dyn_visc * avg_Z(∂w_z, i, J) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_z )
                    + dyn_visc * avg_xZ(u, i, J) * ( avg_x(∂u_z, i, J) + avg_x(∂w_x, i, J) ) )
            # term inside average
            u∂τ_z[i, J] = ( w[i, J] * ( dyn_visc * ∂ω_x - ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_z ) )
        end
    end

    periodize!(model, (uτ_x, u∂τ_x, uτ_z, u∂τ_z))

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            ρε[i, j] = inv_dx * dif_x(uτ_x, i, j) + inv_dz * dif_z(uτ_z, i, j) + avg_x(u∂τ_x, i, j) + avg_z(u∂τ_z, i, j)
        end
    end

    periodize!(model, ρε)

    return ρε, (uτ_x, uτ_z, u∂τ_x, u∂τ_z)
end

## THERMODYNAMICS

function thermodynamics!((conjvar_, chempot_, Cp_, ∂cons_∂q_, Γ_pq_), model::FC2D, (; m), (consvar, comp, p, T))
    (; mgr, fluid) = model

    conjvar = similar!(conjvar_, consvar)
    chempot = similar!(chempot_, consvar)
    Cp      = similar!(Cp_, consvar)
    ∂cons_∂q = similar!(∂cons_∂q_, consvar)
    Γ_pq   = similar!(Γ_pq_, consvar)
    
    @with mgr, let (irange, jrange) = (axes(consvar, 1), axes(consvar, 2))
        @vec for i in irange, j in jrange
            # variable conjugate to conservative variable
            conjvar[i, j]   = fluid(:p, :consvar, :q).conjugate_variable(p[i, j], consvar[i, j], comp[i, j])

            # modified chemical pontential (relative to choice of consvar)
            chempot[i, j]   = fluid(:p, :consvar, :q).modified_chemical_potential(p[i, j], consvar[i, j], comp[i, j])

            # specific heat capacity
            Cp[i, j]        = fluid(:p, :consvar, :q).heat_capacity(p[i, j], consvar[i, j], comp[i, j])

            # partial derivative ∂cons/∂q(p, T, q)
            ∂cons_∂q[i, j]   = fluid(:p, :consvar, :q).dcons_dq(p[i, j], consvar[i, j], comp[i, j])

            # derivatives of μ(p, T, q)
            _, ∂μ_∂p, _, ∂μ_∂q = fluid(:p, :T, :q).chemical_potential_derivatives(p[i, j], T[i, j], comp[i, j])
            Γ_pq[i, j]      = - ∂μ_∂p / ∂μ_∂q
        end
    end
    return conjvar, chempot, Cp, ∂cons_∂q, Γ_pq
end

## MOMENTUM FORCING
function buoyancy!((B_x_, B_z_, B_), model::FC2D, (; m, u, w), (consvar, comp, conjvar, chempot, p))
    (; mgr, domain, space, fluid, inv_dx, inv_dz) = model
    (; g) = space

    B_x = similar!(B_x_, u)
    B_z = similar!(B_z_, w)
    B   = similar!(B_, consvar)

    @with mgr, let (irange, jrange) = (axes(B, 1), axes(B, 2))
        @vec for i in irange, j in jrange
            h = fluid(:p, :consvar, :q).specific_enthalpy(p[i, j], consvar[i, j], comp[i, j])
            B[i, j] = h - conjvar[i, j] * consvar[i, j] - chempot[i, j] * comp[i, j]
        end
    end
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(B_x, 2))
        @vec for I in irange, j in jrange
            v_x         = @views inv(avg_X(m[:, :, 1], I, j))
            consvar_x   = @views avg_X(m[:, :, 2], I, j) * v_x
            comp_x      = @views avg_X(m[:, :, 3], I, j) * v_x
            cons∂conj_x = inv_dx * dif_X(conjvar, I, j) * consvar_x
            comp∂chem_x = inv_dx * dif_X(chempot, I, j) * comp_x
            ∂B_x        = inv_dx * dif_X(B, I, j)
            B_x[I, j]   = - cons∂conj_x  - comp∂chem_x - ∂B_x
        end
    end    
    @with mgr, let (irange, jrange) = (axes(B_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            v_z         = @views inv(avg_Z(m[:, :, 1], i, J))
            consvar_z   = @views avg_Z(m[:, :, 2], i, J) * v_z
            comp_z      = @views avg_Z(m[:, :, 3], i, J) * v_z
            cons∂conj_z = inv_dz * dif_Z(conjvar, i, J) * consvar_z
            comp∂chem_z = inv_dz * dif_Z(chempot, i, J) * comp_z
            ∂B_z        = inv_dz * dif_Z(B, i, J)
            B_z[i, J]   = - cons∂conj_z - comp∂chem_z - ∂B_z - g
        end
    end

    periodize!(model, (B_x, B_z, B))

    return B_x, B_z, B
end

## IRREVERSIBLE TERMS

function fluxes!((Jcons_x_, Jcons_z_, Jq_x_, Jq_z_), model::FC2D, heatflux::HeatFluxConsistent, (; m, u, w), (consvar, comp, conjvar, p, T, Cp, ∂cons_∂q, Γ_pq))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; k_T, k_q) = model.heatflux

    Jcons_x = similar!(Jcons_x_, u)
    Jcons_z = similar!(Jcons_z_, w)
    Jq_x    = similar!(Jq_x_, u)
    Jq_z    = similar!(Jq_z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons_x, 2))
        @vec for I in irange, j in jrange
            ρ_x             = @views avg_X(m[:, :, 1], I, j)
            conjvar_x       = avg_X(conjvar, I, j)
            ∂cons_∂q_x      = avg_X(∂cons_∂q, I, j)

            # choice of horizontal (reduced) heat flux and salt flux
            JT_x            = - ρ_x * avg_X(Cp, I, j) * k_T * inv_dx * dif_X(T, I, j)
            Jq_x[I, j]      = - ρ_x * k_q * ( inv_dx * dif_X(comp, I, j) - avg_X(Γ_pq, I, j) * inv_dx * dif_X(p, I, j) )
            
            # horizontal consvar flux
            Jcons_x[I, j]   = JT_x / conjvar_x + ∂cons_∂q_x * Jq_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ρ_z             = @views avg_Z(m[:, :, 1], i, J)
            conjvar_z       = avg_Z(conjvar, i, J)
            ∂cons_∂q_z      = avg_Z(∂cons_∂q, i, J)
            
            # choice of vertical (reduced) heat flux and salt flux
            JT_z            = - ρ_z * avg_Z(Cp, i, J) * k_T * inv_dz * dif_Z(T, i, J)
            Jq_z[i, J]      = - ρ_z * k_q * ( inv_dz * dif_Z(comp, i, J) - avg_Z(Γ_pq, i, J) * inv_dz * dif_Z(p, i, J) ) 

            # vertical consvar flux
            Jcons_z[i, J]   = JT_z / conjvar_z + ∂cons_∂q_z * Jq_z[i, J]
        end
    end

    periodize!(model, (Jcons_x, Jcons_z, Jq_x, Jq_z))

    return Jcons_x, Jcons_z, Jq_x, Jq_z
end

function fluxes!((Jcons_x_, Jcons_z_, Jq_x_, Jq_z_), model::FC2D, heatflux::HeatFluxSimple, (; u, w), (consvar, comp, conjvar, p, T, Cp, ∂cons_∂q, Γ_pq))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; k_T, k_q) = model.heatflux

    Jcons_x = similar!(Jcons_x_, u)
    Jcons_z = similar!(Jcons_z_, w)
    Jq_x    = similar!(Jq_x_, u)
    Jq_z    = similar!(Jq_z_, w)

    # JH = conjvar * Jcons = J_heat + conjvar * (∂cons / ∂q) * J_q
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons_x, 2))
        @vec for I in irange, j in jrange
            ρ_x             = @views avg_X(m[:, :, 1], I, j)
            Jcons_x[I, j]   = - ρ_x * k_T * inv_dx * dif_X(consvar, I, j)
            Jq_x[I, j]      = - ρ_x * k_q * inv_dx * dif_X(comp, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ρ_z             = @views avg_Z(m[:, :, 1], i, J)
            Jcons_z[i, J]   = - ρ_z * k_T * inv_dz * dif_Z(consvar, i, J)
            Jq_z[i, J]      = - ρ_z * k_q * inv_dz * dif_Z(comp, i, J)
        end
    end

    periodize!(model, (Jcons_x, Jcons_z, Jq_x, Jq_z))

    return Jcons_x, Jcons_z, Jq_x, Jq_z
end

function entropy_production!((σ_cons_, Jcons∂conj_∂x_, Jcons∂conj_∂z_, Jq∂chem_∂x_, Jq∂chem_∂z_), model, heatflux::HeatFluxConsistent, (; u, w), (conjvar, chempot, Jcons_x, Jcons_z, Jq_x, Jq_z, ρε))
    (; mgr, domain, inv_dx, inv_dz) = model
        
    σ_cons = similar!(σ_cons_, conjvar)
    Jcons∂conj_∂x = similar!(Jcons∂conj_∂x_, u)
    Jcons∂conj_∂z = similar!(Jcons∂conj_∂z_, w)
    Jq∂chem_∂x = similar!(Jq∂chem_∂x_, u)
    Jq∂chem_∂z = similar!(Jq∂chem_∂z_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons∂conj_∂x, 2))
        @vec for I in irange, j in jrange
            Jcons∂conj_∂x[I, j] = Jcons_x[I, j] * inv_dx * dif_X(conjvar, I, j) 
            Jq∂chem_∂x[I, j]    = Jq_x[I, j] * inv_dx * dif_X(chempot, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons∂conj_∂z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            Jcons∂conj_∂z[i, J] = Jcons_z[i, J] * inv_dz * dif_Z(conjvar, i, J)
            Jq∂chem_∂z[i, J]    = Jq_z[i, J] * inv_dz * dif_Z(chempot, i, J)
        end
    end
    periodize!(model, (Jcons∂conj_∂x, Jq∂chem_∂x, Jcons∂conj_∂z, Jq∂chem_∂z))

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            σ_cons[i, j] = inv(conjvar[i, j]) * ( ρε[i, j] - avg_x(Jcons∂conj_∂x, i, j) - avg_z(Jcons∂conj_∂z, i, j) - avg_x(Jq∂chem_∂x, i, j) - avg_z(Jq∂chem_∂z, i, j) ) 
        end
    end

    periodize!(model, σ_cons)

    return σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z
end

function entropy_production!((σ_cons_, Jcons∂conj_∂x_, Jcons∂conj_∂z_, Jq∂chem_∂x_, Jq∂chem_∂z_), model, heatflux::HeatFluxSimple, (; u, w), (conjvar, chempot))
    (; mgr, domain) = model
        
    σ_cons = similar!(σ_cons, conjvar)
    Jcons∂conj_∂x = similar!(Jcons∂conj_∂x_, u)
    Jcons∂conj_∂z = similar!(Jcons∂conj_∂z_, w)
    Jq∂chem_∂x = similar!(Jq∂chem_∂x_, u)
    Jq∂chem_∂z = similar!(Jq∂chem_∂z_, w)

    @with mgr, let (irange, jrange) = (axes(σ_cons, 1), axes(σ_cons, 2))
        @vec for i in irange, j in jrange
            σ_cons[i, j] = 0. * σ_cons[i, j]
        end
    end
    return σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z
end

## UPDATE DSTATE


function duw!((du_, dw_), model::FC2D, (; m, u, w), (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, ∂u_x, ∂w_z, ∂u_z, ∂w_x))
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity

    du = similar!(du_, u)
    dw = similar!(dw_, w)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            v_x         = @views inv(avg_X(m[:, :, 1], I, j))
            ∂ω_z        = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))
            ∂divu_x     = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))
            du[I, j]    = B_x[I, j] - ∂K_x[I, j] - ωw[I, j] + v_x * (dyn_visc * ∂ω_z + (bulk_visc + (4 / 3) * dyn_visc) * ∂divu_x)
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange
            v_z         = @views inv(avg_Z(m[:, :, 1], i, J))
            ∂ω_x        = inv_dx * (dif_x(∂u_z, i, J) - dif_x(∂w_x, i, J))
            ∂divu_z     = inv_dz * (dif_Z(∂u_x, i, J) + dif_Z(∂w_z, i, J))
            dw[i, J]    = B_z[i, J] - ∂K_z[i, J] + ωu[i, J] + v_z * (-dyn_visc * ∂ω_x + (bulk_visc + (4 / 3) * dyn_visc) * ∂divu_z)
        end
    end
    return du, dw
end

function dm!(dm_, model::FC2D, (; m), (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons))
    (; mgr, domain, inv_dx, inv_dz) = model
    
    dm = similar!(dm_, m)
    dm1 = @view dm[:, :, 1]
    dm2 = @view dm[:, :, 2]
    dm3 = @view dm[:, :, 3]
    mu1 = @view mu[:, :, 1]
    mu2 = @view mu[:, :, 2]
    mu3 = @view mu[:, :, 3] 
    mw1 = @view mw[:, :, 1]
    mw2 = @view mw[:, :, 2]
    mw3 = @view mw[:, :, 3]
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange                      
            dm1[i, j] = - inv_dx * dif_x(mu1, i, j) - inv_dz * dif_z(mw1, i, j)
            dm2[i, j] = - inv_dx * dif_x(mu2, i, j) - inv_dz * dif_z(mw2, i, j) - inv_dx * dif_x(Jcons_x, i, j) - inv_dz * dif_z(Jcons_z, i, j) + σ_cons[i, j] 
            dm3[i, j] = - inv_dx * dif_x(mu3, i, j) - inv_dz * dif_z(mw3, i, j) - inv_dx * dif_x(Jq_x, i, j)    - inv_dz * dif_z(Jq_z, i, j)
        end       
    end
    return dm
end

## STATE BOUNDARY CONDITION

"""
    state_bc!(model, state, scratch) -> NamedTuple

Apply physical boundary conditions in-place to the prognostic **state variables**
of a 2D fully-compressible model.

This routine enforces boundary conditions on the velocity components
`u`, `w`, and on either the conservative thermodynamic variable `consvar`
*or* the temperature `T`, depending on the boundary-condition configuration.
The remaining thermodynamic quantity is reconstructed consistently.

# Arguments
- `model`
- `(; m, u, w)`
- `(p, consvar, comp, T)`:

# Boundary-condition logic
Exactly one of the following must be specified:
- Boundary conditions on `(consvar, q)`
- Boundary conditions on `(T, q)`

Specifying both `consvar` and `T` boundary conditions is an error.

# Returns
A named tuple `(m = m, u = u, w = w)` with boundary conditions applied.
"""
function state_bc!(model::FC2D, (; m, u, w)::NamedTuple, (p, consvar, comp, T)::Tuple)
    (; boundary, fluid) = model

    bcs = boundary.conditions
    bc_keys = boundary.keys # (:u, :w, :consvar, :q) or (:u, :w, :T, :q)

    if (:consvar in bc_keys) && ~(:T in bc_keys)
        # apply boundary conditions to (u, w, p, consvar, q)
        apply_bc_xz!(model, bcs.p, p)
        apply_bc_xz!(model, bcs.consvar, consvar)
        apply_bc_xz!(model, bcs.q, comp)
        apply_bc_Xz!(model, bcs.u, u)                                                                                                                                                                                                                                      
        apply_bc_xZ!(model, bcs.w, w)
        # construct m = (ρ_R * consvar, ρ_R * comp)
        @with model.mgr, let (irange, jrange) = (axes(m, 1), axes(m, 2))
            @vec for i in irange, j in jrange
                T[i, j] = fluid(:p, :consvar, :q).temperature(p[i, j], consvar[i, j], comp[i, j])
                density = inv(fluid(:p, :T, :q).specific_volume(p[i, j], T[i, j], comp[i, j]))
                m[i, j, 1] = density
                m[i, j, 2] = consvar[i, j] * density
                m[i, j, 3] = comp[i, j] * density
            end
        end
    elseif (:T in bc_keys) && ~(:consvar in bc_keys)                                                                                                                                                
        # apply boundary conditions to (u, w, p, T, q)
        apply_bc_xz!(model, bcs.p, p)
        apply_bc_xz!(model, bcs.T, T)
        apply_bc_xz!(model, bcs.q, comp)
        apply_bc_Xz!(model, bcs.u, u)
        apply_bc_xZ!(model, bcs.w, w)
        # construct m = (ρ_R * consvar, ρ_R * comp)
        @with model.mgr, let (irange, jrange) = (axes(m, 1), axes(m, 2))
            @vec for i in irange, j in jrange
                consvar[i, j] = fluid(:p, :T, :q).conservative_variable(p[i, j], T[i, j], comp[i, j])
                density = inv(fluid(:p, :T, :q).specific_volume(p[i, j], T[i, j], comp[i, j]))
                m[i, j, 1] = density
                m[i, j, 2] = consvar[i, j] * density
                m[i, j, 3] = comp[i, j] * density
            end
        end
    elseif (:consvar in bc_keys) && (:T in bc_keys)
        error("Boundary conditions cannot be specified for both consvar and T.")
    else 
        error("Boundary conditions must be specified for either (p, consvar, q) or (p, T, q).")
    end
    return (; m = m, u = u, w = w)
end

end # module BoxDynamics