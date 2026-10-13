module BoxDynamics

using ..CFCompressible: FC2D
using CFPlanets: Tank2D
using CFBoxes
using CFBoxes: foreach_dim, pick, slice, across
using CFDiffusionSchemes
using ManagedLoops: @vec, @with


## TENDENCIES

function FC2D_tendencies!(dstate, scratch, model::FC2D, state, t)
    # fill the halo of the state (never its interior) from the boundary conditions
    state = state_bc!(model, state)

    ## THERMODYNAMICS, pointwise on all cells: p, consvar, q, T, conjvar, chempot, Cp, ∂consvar/∂q, Γ, B
    (; p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ) = scratch.thermo
    (; B) = scratch.momentum
    p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B =
        thermodynamics!((p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B), model, state)

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

    ## MOMENTUM FORCING ("buoyancy") on faces
    (; B_x, B_z) = scratch.momentum

    B_x, B_z = buoyancy!((B_x, B_z), model, state, (conjvar, chempot, B))
    
    ## CONSVAR & COMPOSITION FLUXES, and their products with the gradients of conjvar and chempot
    (; Jcons_x, Jcons_z, Jq_x, Jq_z, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z) = scratch.irreversible

    Jcons_x, Jcons_z, Jq_x, Jq_z, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z =
        fluxes!((Jcons_x, Jcons_z, Jq_x, Jq_z, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z), model, model.thermoflux, state,
                (consvar, q, conjvar, chempot, p, T, Cp, ∂consvar_∂q, Γ))

    # ENTROPY PRODUCTION
    (; σ_cons) = scratch.irreversible

    σ_cons = entropy_production!(σ_cons, model, model.thermoflux, (conjvar, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z, ρε))

    # UPDATE DSTATE
    dm, du, dw  = (dstate.m, dstate.u, dstate.w)

    du, dw  = velocity_tendencies!((du, dw), model, state, (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, visc_scratch[3], visc_scratch[4]))
    dm      = dm!(dm, model, state, (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons))

    dstate = (m = dm, u = du, w = dw)
   
    # apply boundary conditions to dstate
    dstate = dstate_bc!(model, dstate)


    scratch = (
        momentum        = (; B_x, B_z, B),
        thermo          = (; consvar, q, conjvar, chempot, p, T, Cp, ∂consvar_∂q, Γ),
        derivatives     = (; ∂u_x, ∂w_z, ∂u_z, ∂w_x),
        irreversible    = (; ρε, visc_scratch, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z),
        advection       = (; mu, mw, K, ∂K_x, ∂K_z, ωw, ωu, adv_scratch),
    )
    return dstate, scratch
end

## VELOCITY DERIVATIVES

# skew derivatives : (u, w) at edges ↦ (∂u/∂z, ∂w/∂x) at vertices
function skew_uw!((∂u_z_, ∂w_x_), model::FC2D{F}, (; u, w)) where F
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_z = alloc!(∂u_z_, F, domain, :XZ)
    ∂w_x = alloc!(∂w_x_, F, domain, :XZ)
    
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
function grad_uw!((∂u_x_, ∂w_z_), model::FC2D{F}, (; m, u, w)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model

    ∂u_x = alloc!(∂u_x_, F, domain, :xz)
    ∂w_z = alloc!(∂w_z_, F, domain, :xz)

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
function vort_adv!((ωw_, ωu_, (ωwᴵᴶ_, ωuᴵᴶ_)), model::FC2D{F}, advection::AdvEnergyCons, (; m, u, w), (∂u_z, ∂w_x, mu, mw)) where {F}
    (; mgr, domain) = model

    ωw      = alloc!(ωw_, F, domain, :Xz)
    ωu      = alloc!(ωu_, F, domain, :xZ)
    ωwᴵᴶ    = alloc!(ωwᴵᴶ_, F, domain, :XZ)
    ωuᴵᴶ    = alloc!(ωuᴵᴶ_, F, domain, :XZ)
    
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
function vort_adv!((ωw_, ωu_, (ωvᴵᴶ_, ρwᴵ_, ρuᴶ_)), model::FC2D{F}, advection::AdvEnstrophyCons, (; m, u, w), (∂u_z, ∂w_x, mu, mw)) where {F}
    (; mgr, domain) = model

    ωw      = alloc!(ωw_, F, domain, :Xz)
    ωu      = alloc!(ωu_, F, domain, :xZ)

    ωvᴵᴶ    = alloc!(ωvᴵᴶ_, F, domain, :XZ)
    ρwᴵ     = alloc!(ρwᴵ_, F, domain, :XZ)
    ρuᴶ     = alloc!(ρuᴶ_, F, domain, :XZ)

    mu1 = @view mu[:, :, 1]
    mw1 = @view mw[:, :, 1]
    m1  = @view m[:, :, 1]
    @with mgr, let (irange, jrange) = (Xrange(domain), Zrange(domain))
        @vec for I in irange, J in jrange
            ω = ∂u_z[I, J] - ∂w_x[I, J]
            ωvᴵᴶ[I, J] = ω * inv(avg_XZ(m1, I, J))
            ρwᴵ[I, J] = avg_X(mw1, I, J)
            ρuᴶ[I, J] = avg_Z(mu1, I, J)
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
function ke_adv!((K_, ∂K_x_, ∂K_z_), model::FC2D{F}, (; m, u, w)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model

    K       = alloc!(K_, F, domain, :xz)
    ∂K_x    = alloc!(∂K_x_, F, domain, :Xz)
    ∂K_z    = alloc!(∂K_z_, F, domain, :xZ)

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
function advective_flux!((mu_, mw_), model::FC2D{F}, (; m, u, w)) where {F}
    (; mgr, domain) = model

    mu = alloc!(mu_, F, domain, :Xz, 3)
    mw = alloc!(mw_, F, domain, :xZ, 3)

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
    @with mgr, let (irange, jrange) = (axes(mw, 1), Zrange(domain))
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

# Viscous force on faces, F = ∇·τ (per unit volume), and the dissipation ρε = ∇·(u·τ) - u·F at cell centres. F is
# used again by the momentum tendencies (`velocity_tendencies!`), so that the viscous derivatives are formed once.
function viscous_dissipation!((ρε_, (uτ_x_, uτ_z_, F_x_, F_z_)), model::FC2D{F}, (; m, u, w), (∂u_x, ∂w_z, ∂u_z, ∂w_x)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    (; dyn_visc, bulk_visc) = model.viscosity

    ρε      = alloc!(ρε_, F, domain, :xz)
    uτ_x    = alloc!(uτ_x_, F, domain, :Xz)
    uτ_z    = alloc!(uτ_z_, F, domain, :xZ)
    F_x     = alloc!(F_x_, F, domain, :Xz)
    F_z     = alloc!(F_z_, F, domain, :xZ)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            divu_x = avg_X(∂u_x, I, j) + avg_X(∂w_z, I, j)
            ∂divu_x = inv_dx * (dif_X(∂u_x, I, j) + dif_X(∂w_z, I, j))
            ∂ω_z  = inv_dz * (dif_z(∂u_z, I, j) - dif_z(∂w_x, I, j))
            # term inside gradient
            uτ_x[I, j] = ( u[I, j] * ( 2 * dyn_visc * avg_X(∂u_x, I, j) + ( bulk_visc - (2 / 3) * dyn_visc ) * divu_x )
                    + dyn_visc * avg_Xz(w, I, j) * ( avg_z(∂u_z, I, j) + avg_z(∂w_x, I, j) ) )
            # viscous force
            F_x[I, j] = dyn_visc * ∂ω_z + ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_x
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
            # viscous force
            F_z[i, J] = - dyn_visc * ∂ω_x + ( bulk_visc + ( 4 / 3) * dyn_visc ) * ∂divu_z
        end
    end

    periodize!(model, (uτ_x, F_x, uτ_z, F_z))

    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            ρε[i, j] = inv_dx * dif_x(uτ_x, i, j) + inv_dz * dif_z(uτ_z, i, j) - avg_x(u, F_x, i, j) - avg_z(w, F_z, i, j)
        end
    end

    periodize!(model, ρε)

    return ρε, (uτ_x, uτ_z, F_x, F_z)
end

## THERMODYNAMICS

# All pointwise thermodynamics, in one pass over all cells (halo included, valid after `state_bc!`): from
# m = (ρ, ρ consvar, ρ q), the pressure p(v, consvar, q), the temperature, the conjugate variable and modified
# chemical potential, Cp, ∂consvar/∂q, Γ = -∂μ/∂p / ∂μ/∂q (barodiffusion, per unit pressure) and
# B = h - conjvar consvar - chempot q.
function thermodynamics!((p_, consvar_, q_, T_, conjvar_, chempot_, Cp_, ∂consvar_∂q_, Γ_, B_), model::FC2D{F}, (; m)) where {F}
    (; mgr, fluid) = model

    p           = alloc!(p_, F, model.domain, :xz)
    consvar     = alloc!(consvar_, F, model.domain, :xz)
    q           = alloc!(q_, F, model.domain, :xz)
    T           = alloc!(T_, F, model.domain, :xz)
    conjvar     = alloc!(conjvar_, F, model.domain, :xz)
    chempot     = alloc!(chempot_, F, model.domain, :xz)
    Cp          = alloc!(Cp_, F, model.domain, :xz)
    ∂consvar_∂q = alloc!(∂consvar_∂q_, F, model.domain, :xz)
    Γ           = alloc!(Γ_, F, model.domain, :xz)
    B           = alloc!(B_, F, model.domain, :xz)

    @with mgr, let (irange, jrange) = (axes(m, 1), axes(m, 2))
        @vec for i in irange, j in jrange
            v                 = inv(m[i, j, 1])
            cons, qij         = m[i, j, 2] * v, m[i, j, 3] * v
            pij               = fluid(:v, :consvar, :q).pressure(v, cons, qij)
            Tij               = fluid(:p, :consvar, :q).temperature(pij, cons, qij)
            conj              = fluid(:p, :consvar, :q).conjugate_variable(pij, cons, qij)
            chem              = fluid(:p, :consvar, :q).modified_chemical_potential(pij, cons, qij)
            _, ∂μ_∂p, _, ∂μ_∂q = fluid(:p, :T, :q).chemical_potential_derivatives(pij, Tij, qij)
            consvar[i, j], q[i, j], p[i, j], T[i, j] = cons, qij, pij, Tij
            conjvar[i, j], chempot[i, j] = conj, chem
            Cp[i, j]          = fluid(:p, :consvar, :q).heat_capacity(pij, cons, qij)
            ∂consvar_∂q[i, j] = fluid(:p, :consvar, :q).dcons_dq(pij, cons, qij)
            Γ[i, j]           = - ∂μ_∂p / ∂μ_∂q
            B[i, j]           = fluid(:p, :consvar, :q).specific_enthalpy(pij, cons, qij) - conj * cons - chem * qij
        end
    end
    return p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B
end

## MOMENTUM FORCING

# thermodynamic momentum forcing - consvar ∂conjvar - q ∂chempot - ∂B on the face between cells a and b (b after a,
# at inverse distance inv_d), from the densities (ρ, ρ consvar, ρ q) and (conjvar, chempot, B) of each cell.
# Shared by the tendencies and by the construction of discretely balanced states.
@inline function face_forcing(inv_d, ρa, ra, qa, ρb, rb, qb, conj_a, chem_a, B_a, conj_b, chem_b, B_b)
    v = inv(ρa + ρb)
    consvar, q = (ra + rb) * v, (qa + qb) * v
    return - inv_d * ((conj_b - conj_a) * consvar + (chem_b - chem_a) * q + (B_b - B_a))
end

# (conjvar, chempot, B) in a cell with density ρ, conservative variable cons and composition q
@inline function cell_thermo(fluid, ρ, cons, q)
    p = fluid(:v, :consvar, :q).pressure(inv(ρ), cons, q)
    conj = fluid(:p, :consvar, :q).conjugate_variable(p, cons, q)
    chem = fluid(:p, :consvar, :q).modified_chemical_potential(p, cons, q)
    B = fluid(:p, :consvar, :q).specific_enthalpy(p, cons, q) - conj * cons - chem * q
    return conj, chem, B
end
# momentum forcing on u- and w-faces, from the pointwise conjvar, chempot and B
function buoyancy!((B_x_, B_z_), model::FC2D{F}, (; m, u, w), (conjvar, chempot, B)) where {F}
    (; mgr, domain, space, inv_dx, inv_dz) = model
    (; g) = space

    B_x = alloc!(B_x_, F, domain, :Xz)
    B_z = alloc!(B_z_, F, domain, :xZ)
    m1, m2, m3 = (@view m[:, :, 1]), (@view m[:, :, 2]), (@view m[:, :, 3])
    @with mgr, let (irange, jrange) = (Xrange(domain), axes(B_x, 2))
        @vec for I in irange, j in jrange
            B_x[I, j] = face_forcing(inv_dx, m1[I-1, j], m2[I-1, j], m3[I-1, j], m1[I, j], m2[I, j], m3[I, j],
                                     conjvar[I-1, j], chempot[I-1, j], B[I-1, j], conjvar[I, j], chempot[I, j], B[I, j])
        end
    end
    @with mgr, let (irange, jrange) = (axes(B_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            B_z[i, J] = face_forcing(inv_dz, m1[i, J-1], m2[i, J-1], m3[i, J-1], m1[i, J], m2[i, J], m3[i, J],
                                     conjvar[i, J-1], chempot[i, J-1], B[i, J-1], conjvar[i, J], chempot[i, J], B[i, J]) - g
        end
    end

    periodize!(model, (B_x, B_z))

    return B_x, B_z
end

## IRREVERSIBLE TERMS

# Fluxes of the conservative variable and composition on faces, for both thermo flux schemes:
#   Jq = salt_flux(ρ, Γ, ∂q, ∂p),  Jcons = consvar_flux(ρ, Cp, ∂T, ∂consvar, Jq, conjvar, ∂consvar/∂q),
# with face densities and coefficients averaged from the two cells, and gradients normal to the face. The same
# pointwise formulas give the fluxes through walls with a prescribed `Value` (see `wall_flux`). The products
# J ∂conjvar and Jq ∂chempot on faces, for the production σ, are formed in the same pass.
function fluxes!((Jcons_x_, Jcons_z_, Jq_x_, Jq_z_, Jcons∂conj_∂x_, Jcons∂conj_∂z_, Jq∂chem_∂x_, Jq∂chem_∂z_), model::FC2D{F}, thermoflux, (; m, u, w),
                 (consvar, q, conjvar, chempot, p, T, Cp, ∂consvar_∂q, Γ)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model

    Jcons_x = alloc!(Jcons_x_, F, domain, :Xz)
    Jcons_z = alloc!(Jcons_z_, F, domain, :xZ)
    Jq_x    = alloc!(Jq_x_, F, domain, :Xz)
    Jq_z    = alloc!(Jq_z_, F, domain, :xZ)
    Jcons∂conj_∂x = alloc!(Jcons∂conj_∂x_, F, domain, :Xz)
    Jcons∂conj_∂z = alloc!(Jcons∂conj_∂z_, F, domain, :xZ)
    Jq∂chem_∂x    = alloc!(Jq∂chem_∂x_, F, domain, :Xz)
    Jq∂chem_∂z    = alloc!(Jq∂chem_∂z_, F, domain, :xZ)
    m1 = @view m[:, :, 1]

    @with mgr, let (irange, jrange) = (Xrange(domain), axes(Jcons_x, 2))
        @vec for I in irange, j in jrange
            ρ_x           = avg_X(m1, I, j)
            Jq            = salt_flux(thermoflux, ρ_x, avg_X(Γ, I, j), inv_dx * dif_X(q, I, j), inv_dx * dif_X(p, I, j))
            Jcons         = consvar_flux(thermoflux, ρ_x, avg_X(Cp, I, j), inv_dx * dif_X(T, I, j), inv_dx * dif_X(consvar, I, j), Jq, avg_X(conjvar, I, j), avg_X(∂consvar_∂q, I, j))
            Jq_x[I, j], Jcons_x[I, j] = Jq, Jcons
            Jcons∂conj_∂x[I, j] = Jcons * inv_dx * dif_X(conjvar, I, j)
            Jq∂chem_∂x[I, j]    = Jq * inv_dx * dif_X(chempot, I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(Jcons_z, 1), Zrange(domain))
        @vec for i in irange, J in jrange
            ρ_z           = avg_Z(m1, i, J)
            Jq            = salt_flux(thermoflux, ρ_z, avg_Z(Γ, i, J), inv_dz * dif_Z(q, i, J), inv_dz * dif_Z(p, i, J))
            Jcons         = consvar_flux(thermoflux, ρ_z, avg_Z(Cp, i, J), inv_dz * dif_Z(T, i, J), inv_dz * dif_Z(consvar, i, J), Jq, avg_Z(conjvar, i, J), avg_Z(∂consvar_∂q, i, J))
            Jq_z[i, J], Jcons_z[i, J] = Jq, Jcons
            Jcons∂conj_∂z[i, J] = Jcons * inv_dz * dif_Z(conjvar, i, J)
            Jq∂chem_∂z[i, J]    = Jq * inv_dz * dif_Z(chempot, i, J)
        end
    end

    thermo = (; consvar, q, conjvar, p, T, Cp, ∂consvar_∂q, Γ)
    wall_fluxes!(model, thermoflux, m, thermo, ((Jcons_x, Jq_x), (Jcons_z, Jq_z)))
    foreach(a -> zero_walls!(domain, a, Val(1)), (Jcons∂conj_∂x, Jq∂chem_∂x))     # wall faces: see `entropy_production!`
    foreach(a -> zero_walls!(domain, a, Val(2)), (Jcons∂conj_∂z, Jq∂chem_∂z))
    periodize!(model, (Jcons_x, Jcons_z, Jq_x, Jq_z, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z))

    return Jcons_x, Jcons_z, Jq_x, Jq_z, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z
end

# composition flux -ρ k_q (∂q - Γ ∂p), with barodiffusion in the consistent scheme only
@inline salt_flux(hf::ThermoFluxConsistent, ρ, Γ, ∂q, ∂p) = - ρ * hf.k_q * (∂q - Γ * ∂p)
@inline salt_flux(hf::ThermoFluxDowngradient, ρ, Γ, ∂q, ∂p) = - ρ * hf.k_q * ∂q
# heat flux -ρ Cp k_T ∂T of the consistent scheme
@inline heat_flux(hf::ThermoFluxConsistent, ρ, Cp, ∂T) = - ρ * Cp * hf.k_T * ∂T
# flux of consvar carrying the heat flux JT: JT / conjvar, plus ∂consvar/∂q Jq in the consistent scheme
@inline heat_to_consvar(::ThermoFluxConsistent, JT, Jq, conj, ∂consvar_∂q) = JT / conj + ∂consvar_∂q * Jq
@inline heat_to_consvar(::ThermoFluxDowngradient, JT, Jq, conj, ∂consvar_∂q) = JT / conj
# flux of consvar: from the heat flux in the consistent scheme, down the gradient of consvar in the downgradient one
@inline consvar_flux(hf::ThermoFluxConsistent, ρ, Cp, ∂T, ∂cons, Jq, conj, ∂consvar_∂q) =
    heat_to_consvar(hf, heat_flux(hf, ρ, Cp, ∂T), Jq, conj, ∂consvar_∂q)
@inline consvar_flux(hf::ThermoFluxDowngradient, ρ, Cp, ∂T, ∂cons, Jq, conj, ∂consvar_∂q) = - ρ * hf.k_T * ∂cons

## WALL FLUXES OF TRACERS

# Tracer fluxes on the wall faces, from the wall conditions, replacing the values computed across the wall (which
# would involve the halo). Walls are impermeable: the advective fluxes vanish there with the normal velocity.
wall_fluxes!(model::FC2D, thermoflux, m, thermo, fluxes) = foreach_dim(model.boundary.walls) do D
    wall_fluxes!(pick(model.boundary.walls, D), D, pick(fluxes, D), model, thermoflux, m, thermo)
end
wall_fluxes!(::Nothing, _...) = nothing
function wall_fluxes!((lower, upper)::Tuple{Wall, Wall}, D::Val{d}, (Jcons, Jq), model::FC2D, thermoflux, m, thermo) where d
    M, H = dims(model.domain)[d], halo_size(model.domain)[d]
    Δ, g = (model.dx, model.dz)[d], (zero(model.space.g), model.space.g)[d]
    for I in slice(Jcons, D)
        lo, hi = across(I, D, H+1), across(I, D, H+M+1)
        Jcons[lo], Jq[lo] = wall_flux(model, thermoflux, lower.tracers, I, across(I, D, H+1), 1, Δ/2, g, m, thermo)
        Jcons[hi], Jq[hi] = wall_flux(model, thermoflux, upper.tracers, I, across(I, D, H+M), -1, Δ/2, g, m, thermo)
    end
end

# Fluxes (Jcons, Jq) through a wall face, along the coordinate normal to the wall, from the conditions on
# (consvar, q). `I` indexes the conditions along the wall; `cell` is next to the wall, at distance `d`; `n` = ±1
# points from the wall into the fluid; `g` is gravity along the normal, used to extrapolate the pressure
# hydrostatically to the wall. Gradients across the half cell: ∂X ≈ n (X_cell - X_wall) / d.
#
# Choice: the consvar flux through a wall is derived from a heat flux JT (a `HeatFlux` condition, or the consistent
# scheme's flux at a `Value` wall) with the conjugate variable of the cell next to the wall, Jcons = JT / conjvar_cell
# (+ ∂consvar/∂q Jq for the consistent scheme). The energy delivered by the wall is then exactly JT, and the irreversible production in the half cell between the wall and the cell
# centre is contained in Jcons, so that the wall faces add nothing to the production σ (see `entropy_production!`).
@inline function wall_flux(model, thermoflux, (cons_bc, q_bc), I, cell, n, d, g, m, thermo)
    (; consvar, q, conjvar, p, T, Cp, ∂consvar_∂q, Γ) = thermo
    ρ = m[cell, 1]
    p_w = p[cell] + n * ρ * g * d
    Jq = wall_flux(q_bc, thermoflux, I, n, d, ρ, Γ[cell], q[cell], n * (p[cell] - p_w) / d)
    Jcons = wall_flux(cons_bc, model.fluid, thermoflux, I, n, d, ρ, q[cell], p_w, T[cell], consvar[cell], conjvar[cell], Cp[cell], ∂consvar_∂q[cell], Jq)
    return Jcons, Jq
end

# composition: prescribed flux of ρq, or the face flux with the wall value as neighbour
@inline wall_flux(c::Flux, hf, I, n, d, ρ, Γ, q, ∂p) = n * bc_value(c, I)
@inline wall_flux(c::Value, hf, I, n, d, ρ, Γ, q, ∂p) = salt_flux(hf, ρ, Γ, n * (q - bc_value(c, I)) / d, ∂p)
# consvar: prescribed flux of ρ consvar, prescribed heat flux, or the face flux with the wall value as neighbour,
# with T_wall = T(p_wall, consvar_wall, q_cell)
@inline wall_flux(c::Flux{V, :density}, fluid, hf, I, n, d, ρ, q, p_w, T, cons, conj, Cp, ∂consvar_∂q, Jq) where V = n * bc_value(c, I)
@inline wall_flux(c::Flux{V, :heat}, fluid, hf, I, n, d, ρ, q, p_w, T, cons, conj, Cp, ∂consvar_∂q, Jq) where V =
    heat_to_consvar(hf, n * bc_value(c, I), Jq, conj, ∂consvar_∂q)
@inline function wall_flux(c::Value, fluid, hf, I, n, d, ρ, q, p_w, T, cons, conj, Cp, ∂consvar_∂q, Jq)
    cons_w = bc_value(c, I)
    T_w = fluid(:p, :consvar, :q).temperature(p_w, cons_w, q)
    return consvar_flux(hf, ρ, Cp, n * (T - T_w) / d, n * (cons - cons_w) / d, Jq, conj, ∂consvar_∂q)
end

# Choice: the face terms J ∂conjvar and Jq ∂chempot of the production σ vanish on wall faces. Each face term is
# shared half-and-half by the two cells around the face; on a wall face the other "cell" is the halo, which is not
# fluid. Instead, the energy delivered by a wall flux is credited with the conjugate variables of the cell next to
# the wall: conjvar_cell Jcons + chempot_cell Jq. By the construction of `wall_flux`, this is exactly the heat flux
# JT for `HeatFlux` conditions, and for `Value` conditions with the consistent scheme; for `Flux` on consvar (a prescribed flux F of ρ consvar), it is
# conjvar_cell F. Use `HeatFlux` to prescribe the energy input itself.
# production of the conservative variable, σ = (ρε - avg(Jcons ∂conjvar) - avg(Jq ∂chempot)) / conjvar, from the face
# products formed by `fluxes!` (zero on wall faces)
function entropy_production!(σ_cons_, model::FC2D{F}, thermoflux::ThermoFluxConsistent, (conjvar, Jcons∂conj_∂x, Jcons∂conj_∂z, Jq∂chem_∂x, Jq∂chem_∂z, ρε)) where {F}
    (; mgr, domain) = model
    σ_cons = alloc!(σ_cons_, F, domain, :xz)
    @with mgr, let (irange, jrange) = (xrange(domain), zrange(domain))
        @vec for i in irange, j in jrange
            σ_cons[i, j] = inv(conjvar[i, j]) * ( ρε[i, j] - avg_x(Jcons∂conj_∂x, i, j) - avg_z(Jcons∂conj_∂z, i, j) - avg_x(Jq∂chem_∂x, i, j) - avg_z(Jq∂chem_∂z, i, j) )
        end
    end
    periodize!(model, σ_cons)
    return σ_cons
end
# downgradient scheme: no source of the conservative variable (neither viscous nor diffusive heating)
entropy_production!(σ_cons_, model::FC2D{F}, thermoflux::ThermoFluxDowngradient, (conjvar, _...)) where {F} = fill!(alloc!(σ_cons_, F, model.domain, :xz), 0)

## UPDATE DSTATE


function velocity_tendencies!((du_, dw_), model::FC2D{F}, (; m, u, w), (B_x, B_z, ∂K_x, ∂K_z, ωw, ωu, F_x, F_z)) where {F}
    (; mgr, domain) = model

    du = alloc!(du_, F, domain, :Xz)
    dw = alloc!(dw_, F, domain, :xZ)

    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange
            v_x         = @views inv(avg_X(m[:, :, 1], I, j))
            du[I, j]    = B_x[I, j] - ∂K_x[I, j] - ωw[I, j] + v_x * F_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange
            v_z         = @views inv(avg_Z(m[:, :, 1], i, J))
            dw[i, J]    = B_z[i, J] - ∂K_z[i, J] + ωu[i, J] + v_z * F_z[i, J]
        end
    end
    return du, dw
end

function dm!(dm_, model::FC2D{F}, (; m), (mu, mw, Jcons_x, Jcons_z, Jq_x, Jq_z, σ_cons)) where {F}
    (; mgr, domain, inv_dx, inv_dz) = model
    
    dm = alloc!(dm_, F, domain, :xz, size(m, 3))
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

## BOUNDARY CONDITIONS

# tendencies vanish in the halo, and for the normal velocity on walls
function dstate_bc!(model::FC2D, (; m, u, w))
    (; domain) = model
    zero_halo!(domain, m, (Centred(), Centred()))
    zero_halo!(domain, u, (Faced(), Centred()))
    zero_halo!(domain, w, (Centred(), Faced()))
    return (; m, u, w)
end

## STATE BOUNDARY CONDITION

# velocity ghost values realize the wall (or periodic) conditions; the halo of m mirrors the interior, so that the
# equation of state is only ever evaluated on valid states. Wall fluxes of consvar and q do not use the halo of m.
function state_bc!(model::FC2D, (; m, u, w))
    (; domain, boundary, dx, dz) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), model.viscosity.dyn_visc)
    mirror_halo!(domain, m)
    return (; m, u, w)
end

end # module BoxDynamics