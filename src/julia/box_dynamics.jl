module BoxDynamics

using ..CFCompressible: FC2D
using CFPlanets: Tank2D
using CFBoxes
using CFBoxes: slice, across
using CFDiffusionSchemes
using ManagedLoops: @vec, @with


## TENDENCIES
#
# ∂ₜm = -∇·(m u) - ∇·J + σ ẑ_consvar,   ∂ₜu = -consvar ∇conjvar - q ∇chempot - ∇B - g ẑ - ∇K - ω×u + F/ρ,
# with m = (ρ, ρ consvar, ρ q) at cell centres and (u, w) on faces. Averages and differences come from
# `operators(model.spatial_scheme)`.

function FC2D_tendencies!(dstate, scratch, model::FC2D, state, t)
    (; vortex_scheme, thermoflux) = model
    # fill the halo of the state (never its interior) from the boundary conditions
    state = state_bc!(model, state)

    ## THERMODYNAMICS; p, consvar, q, T, conjvar, chempot, Cp, ∂consvar/∂q, Γ, B
    (; p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B) = scratch.thermo

    p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B =
        thermodynamics!((p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B), model, state)
    thermo = (; p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B)

    ## KINEMATICS; ∂u/∂x, ∂w/∂z, ∂u/∂z, ∂w/∂x, K = (u² + w²) / 2
    (; ∂u_x, ∂w_z, ∂u_z, ∂w_x) = scratch.derivatives
    (; K) = scratch.advection

    ∂u_x, ∂w_z, ∂u_z, ∂w_x, K = kinematics!((∂u_x, ∂w_z, ∂u_z, ∂w_x, K), model, state)

    ## ADVECTION; fluxes m u of the densities, vortex force ω×u = (vortex_x, vortex_z)
    (; mu, mw, vortex_x, vortex_z, vortex_scratch) = scratch.advection

    mu, mw                 = advective_flux!((mu, mw), model, state)
    vortex_x, vortex_z, vortex_scratch = vortex_force!((vortex_x, vortex_z, vortex_scratch), model, vortex_scheme, (state.u, state.w), view(state.m, :, :, 1))

    ## VISCOUS STRESS; u·τ and F = ∇·τ on faces
    (; uτ_x, uτ_z, F_x, F_z, wτ_xz, uτ_xz) = scratch.irreversible

    uτ_x, uτ_z, F_x, F_z, wτ_xz, uτ_xz = viscous_stress!((uτ_x, uτ_z, F_x, F_z, wτ_xz, uτ_xz), model, state,
                                                          (∂u_x, ∂w_z, ∂u_z, ∂w_x))

    ## IRREVERSIBLE FLUXES of consvar and q, with the wall conditions, and the diffusive heating on faces
    (; consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z) = scratch.irreversible

    consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z =
        fluxes!((consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z), model, state, thermo)

    ## DISSIPATIVE HEATING; ρε, σ_cons
    (; ρε, σ_cons) = scratch.irreversible

    ρε, σ_cons = dissipative_heating!((ρε, σ_cons), model, thermoflux, state, (uτ_x, uτ_z, F_x, F_z), thermo,
                                      (diffusive_heating_x, diffusive_heating_z))

    ## MOMENTUM
    du, dw = velocity_tendencies!((dstate.u, dstate.w), model, state, thermo, (K, vortex_x, vortex_z, F_x, F_z))

    ## DENSITIES
    dm = dm!(dstate.m, model, state, (mu, mw, consflux_x, consflux_z, qflux_x, qflux_z, σ_cons))

    dstate = dstate_bc!(model, (m = dm, u = du, w = dw))

    scratch = (
        thermo          = thermo,
        derivatives     = (; ∂u_x, ∂w_z, ∂u_z, ∂w_x),
        advection       = (; K, mu, mw, vortex_x, vortex_z, vortex_scratch),
        irreversible    = (; uτ_x, uτ_z, F_x, F_z, wτ_xz, uτ_xz, ρε, consflux_x, consflux_z, qflux_x, qflux_z,
                             diffusive_heating_x, diffusive_heating_z, σ_cons),
    )
    return dstate, scratch
end

## THERMODYNAMICS

# Pointwise thermodynamics from m = (ρ, ρ consvar, ρ q), on the interior cells: p(v, consvar, q), T, conjvar, chempot,
# Cp, ∂consvar/∂q, Γ = -∂μ/∂p / ∂μ/∂q (barodiffusion) and B = h - conjvar consvar - chempot q, the potential of the
# momentum forcing. Zero-gradient halo: every difference across a wall vanishes.
function thermodynamics!((p_, consvar_, q_, T_, conjvar_, chempot_, Cp_, ∂consvar_∂q_, Γ_, B_), model::FC2D{F}, (; m)) where {F}
    (; mgr, domain, fluid) = model

    p           = alloc!(p_, F, mgr, domain, (Cell(), Cell()))
    consvar     = alloc!(consvar_, F, mgr, domain, (Cell(), Cell()))
    q           = alloc!(q_, F, mgr, domain, (Cell(), Cell()))
    T           = alloc!(T_, F, mgr, domain, (Cell(), Cell()))
    conjvar     = alloc!(conjvar_, F, mgr, domain, (Cell(), Cell()))
    chempot     = alloc!(chempot_, F, mgr, domain, (Cell(), Cell()))
    Cp          = alloc!(Cp_, F, mgr, domain, (Cell(), Cell()))
    ∂consvar_∂q = alloc!(∂consvar_∂q_, F, mgr, domain, (Cell(), Cell()))
    Γ           = alloc!(Γ_, F, mgr, domain, (Cell(), Cell()))
    B           = alloc!(B_, F, mgr, domain, (Cell(), Cell()))

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
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

    fill_halo!((p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B), domain, ZeroGradient())

    return p, consvar, q, T, conjvar, chempot, Cp, ∂consvar_∂q, Γ, B
end

## KINEMATICS

# cells: ∂u/∂x, ∂w/∂z and K = (ℑ(u², (1,), (Cell(),)) + ℑ(w², (2,), (Cell(),))) / 2;  vertices: ∂u/∂z, ∂w/∂x
function kinematics!((∂u_x_, ∂w_z_, ∂u_z_, ∂w_x_, K_), model::FC2D{F}, (; u, w)) where {F}
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)

    ∂u_x    = alloc!(∂u_x_, F, mgr, domain, (Cell(), Cell()))
    ∂w_z    = alloc!(∂w_z_, F, mgr, domain, (Cell(), Cell()))
    ∂u_z    = alloc!(∂u_z_, F, mgr, domain, (Face(), Face()))
    ∂w_x    = alloc!(∂w_x_, F, mgr, domain, (Face(), Face()))
    K       = alloc!(K_, F, mgr, domain, (Cell(), Cell()))

    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ∂u_x[i, j] = inv_dx * δ(u, (1,), (Cell(),), i, j)
            ∂w_z[i, j] = inv_dz * δ(w, (2,), (Cell(),), i, j)
            K[i, j]    = 0.5 * ( ℑ((u, u), (1,), (Cell(),), i, j) + ℑ((w, w), (2,), (Cell(),), i, j) )
        end
    end
    # each vertex derivative over the full axis along which it is not differenced
    @with mgr, let (irange, jrange) = (axes(∂u_z, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            ∂u_z[I, J] = inv_dz * δ(u, (2,), (Face(),), I, J)
        end
    end
    @with mgr, let (irange, jrange) = (faces(domain, 1), axes(∂w_x, 2))
        @inbounds @vec for I in irange, J in jrange
            ∂w_x[I, J] = inv_dx * δ(w, (1,), (Face(),), I, J)
        end
    end

    periodize!(model, (∂u_x, ∂w_z, ∂u_z, ∂w_x, K))

    return ∂u_x, ∂w_z, ∂u_z, ∂w_x, K
end

## ADVECTION

# fluxes of the densities on faces: m u = u ℑ(m, (1,), (Face(),)), m w = w ℑ(m, (2,), (Face(),)) (third index: ρ, ρ consvar, ρ q)
function advective_flux!((mu_, mw_), model::FC2D{F}, (; m, u, w)) where {F}
    (; mgr, domain, spatial_scheme) = model
    (; ℑ) = operators(spatial_scheme)

    mu = alloc!(mu_, F, mgr, domain, (Face(), Cell()), 3)
    mw = alloc!(mw_, F, mgr, domain, (Cell(), Face()), 3)

    m1, m2, m3      = eachslice(m; dims = 3)
    mu1, mu2, mu3   = eachslice(mu; dims = 3)
    mw1, mw2, mw3   = eachslice(mw; dims = 3)

    @with mgr, let (irange, jrange) = (faces(domain, 1), axes(mu, 2))
        @inbounds @vec for I in irange, j in jrange
            mu1[I, j] = u[I, j] * ℑ(m1, (1,), (Face(),), I, j)
            mu2[I, j] = u[I, j] * ℑ(m2, (1,), (Face(),), I, j)
            mu3[I, j] = u[I, j] * ℑ(m3, (1,), (Face(),), I, j)
        end
    end
    @with mgr, let (irange, jrange) = (axes(mw, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            mw1[i, J] = w[i, J] * ℑ(m1, (2,), (Face(),), i, J)
            mw2[i, J] = w[i, J] * ℑ(m2, (2,), (Face(),), i, J)
            mw3[i, J] = w[i, J] * ℑ(m3, (2,), (Face(),), i, J)
        end
    end
    periodize!(model, (mu, mw))
    return mu, mw
end

## VISCOUS STRESS

# Constant dynamic viscosity μ and the Stokes hypothesis: τ = 2μ (S - (∇·u) I / 3). On faces: the viscous force F = ∇·τ
# (per unit volume) and the energy flux u·τ; ρε = ∇·(u·τ) - u·F is formed in `dissipative_heating!`.
# The normal stresses τ_xx, τ_zz live at the cells and the shear stress τ_xz = μ (∂u/∂z + ∂w/∂x) at the vertices. The
# energy flux pairs each stress with the velocity it multiplies before interpolating,
#   (u·τ)_x = u ℑX(τ_xx) + ℑz(ℑX(w) τ_xz),   (u·τ)_z = w ℑZ(τ_zz) + ℑx(ℑZ(u) τ_xz),
# so that, summing by parts, ρε is in each cell the discrete dissipation τ:∇u, a sum of squares (≥ 0):
#   ρε = (2μ/3) (∂u_x² + ∂w_z² + (∂u_x - ∂w_z)²) + μ ℑxz((∂u_z + ∂w_x)²),
# the first term being 2μ times the squared deviatoric normal strains ∂u_x - D/3, ∂w_z - D/3 and -D/3 (∂v/∂y = 0),
# with D = ∇·u.
# F is written in the rotational form F = -μ ∇×ω + (4μ/3) ∇(∇·u), identical to ∇·τ on the C-grid.
function viscous_stress!((uτ_x_, uτ_z_, F_x_, F_z_, wτ_xz_, uτ_xz_), model::FC2D{F}, (; u, w), (∂u_x, ∂w_z, ∂u_z, ∂w_x)) where {F}
    (; mgr, domain, inv_dx, inv_dz, viscosity, spatial_scheme) = model
    (; dyn_visc) = viscosity
    (; ℑ, δ) = operators(spatial_scheme)

    uτ_x    = alloc!(uτ_x_, F, mgr, domain, (Face(), Cell()))
    uτ_z    = alloc!(uτ_z_, F, mgr, domain, (Cell(), Face()))
    F_x     = alloc!(F_x_, F, mgr, domain, (Face(), Cell()))
    F_z     = alloc!(F_z_, F, mgr, domain, (Cell(), Face()))
    wτ_xz   = alloc!(wτ_xz_, F, mgr, domain, (Face(), Face()))
    uτ_xz   = alloc!(uτ_xz_, F, mgr, domain, (Face(), Face()))

    # vertices: the work of the shear stress, ℑX(w) τ_xz and ℑZ(u) τ_xz
    @with mgr, let (irange, jrange) = (faces(domain, 1), faces(domain, 2))
        @inbounds @vec for I in irange, J in jrange
            τ_xz        = dyn_visc * ( ∂u_z[I, J] + ∂w_x[I, J] )
            wτ_xz[I, J] = ℑ(w, (1,), (Face(),), I, J) * τ_xz
            uτ_xz[I, J] = ℑ(u, (2,), (Face(),), I, J) * τ_xz
        end
    end
    periodize!(model, (wτ_xz, uτ_xz))

    @with mgr, let (irange, jrange) = (faces(domain, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            divu_x  = ℑ(∂u_x, (1,), (Face(),), I, j) + ℑ(∂w_z, (1,), (Face(),), I, j)                   # ∇·u
            ∂divu_x = inv_dx * (δ(∂u_x, (1,), (Face(),), I, j) + δ(∂w_z, (1,), (Face(),), I, j))        # ∂ₓ(∇·u)
            ∂ω_z    = inv_dz * (δ(∂u_z, (2,), (Cell(),), I, j) - δ(∂w_x, (2,), (Cell(),), I, j))        # ∂_z ω
            # u·τ along x: u ℑX(τ_xx) + ℑz(ℑX(w) τ_xz), with τ_xx = 2μ (∂u/∂x - ∇·u / 3)
            uτ_x[I, j] = 2 * dyn_visc * u[I, j] * ( ℑ(∂u_x, (1,), (Face(),), I, j) - divu_x / 3 ) + ℑ(wτ_xz, (2,), (Cell(),), I, j)
            # Fₓ = μ (∂_z ω + (4/3) ∂ₓ(∇·u))
            F_x[I, j] = dyn_visc * ( ∂ω_z + (4 / 3) * ∂divu_x )
        end
    end
    @with mgr, let (irange, jrange) = (cells(domain, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            divu_z  = ℑ(∂u_x, (2,), (Face(),), i, J) + ℑ(∂w_z, (2,), (Face(),), i, J)                   # ∇·u
            ∂divu_z = inv_dz * (δ(∂u_x, (2,), (Face(),), i, J) + δ(∂w_z, (2,), (Face(),), i, J))        # ∂_z(∇·u)
            ∂ω_x    = inv_dx * (δ(∂u_z, (1,), (Cell(),), i, J) - δ(∂w_x, (1,), (Cell(),), i, J))        # ∂ₓ ω
            # u·τ along z: w ℑZ(τ_zz) + ℑx(ℑZ(u) τ_xz), with τ_zz = 2μ (∂w/∂z - ∇·u / 3)
            uτ_z[i, J] = 2 * dyn_visc * w[i, J] * ( ℑ(∂w_z, (2,), (Face(),), i, J) - divu_z / 3 ) + ℑ(uτ_xz, (1,), (Cell(),), i, J)
            # F_z = μ (-∂ₓ ω + (4/3) ∂_z(∇·u))
            F_z[i, J] = dyn_visc * ( -∂ω_x + (4 / 3) * ∂divu_z )
        end
    end

    periodize!(model, (uτ_x, uτ_z, F_x, F_z))

    return uτ_x, uτ_z, F_x, F_z, wτ_xz, uτ_xz
end

## IRREVERSIBLE FLUXES

# On faces: qflux = -ρ k_q (∂q - Γ ∂p) and consflux (carrying the heat flux -ρ Cp k_T ∂T, or down the gradient of
# consvar), ρ and coefficients averaged from the two cells; wall faces from `apply_tracer_bcs!`. Also the diffusive
# heating -(consflux ∂conjvar + qflux ∂chempot), zero on wall faces (zero-gradient halo of conjvar, chempot).
function fluxes!((consflux_x_, consflux_z_, qflux_x_, qflux_z_, diffusive_heating_x_, diffusive_heating_z_), model::FC2D{F}, (; m), thermo) where {F}
    (; mgr, domain, thermoflux, inv_dx, inv_dz, spatial_scheme) = model
    (; p, consvar, q, conjvar, chempot, T, Cp, ∂consvar_∂q, Γ) = thermo
    (; ℑ, δ) = operators(spatial_scheme)

    consflux_x          = alloc!(consflux_x_, F, mgr, domain, (Face(), Cell()))
    consflux_z          = alloc!(consflux_z_, F, mgr, domain, (Cell(), Face()))
    qflux_x             = alloc!(qflux_x_, F, mgr, domain, (Face(), Cell()))
    qflux_z             = alloc!(qflux_z_, F, mgr, domain, (Cell(), Face()))
    diffusive_heating_x = alloc!(diffusive_heating_x_, F, mgr, domain, (Face(), Cell()))
    diffusive_heating_z = alloc!(diffusive_heating_z_, F, mgr, domain, (Cell(), Face()))

    m1 = view(m, :, :, 1)

    @with mgr, let (irange, jrange) = (faces(domain, 1), axes(consflux_x, 2))
        @inbounds @vec for I in irange, j in jrange
            ρ_x      = ℑ(m1, (1,), (Face(),), I, j)
            qflux    = salt_flux(thermoflux, ρ_x, ℑ(Γ, (1,), (Face(),), I, j), inv_dx * δ(q, (1,), (Face(),), I, j), inv_dx * δ(p, (1,), (Face(),), I, j))
            consflux = consvar_flux(thermoflux, ρ_x, ℑ(Cp, (1,), (Face(),), I, j), inv_dx * δ(T, (1,), (Face(),), I, j), inv_dx * δ(consvar, (1,), (Face(),), I, j), qflux, ℑ(conjvar, (1,), (Face(),), I, j), ℑ(∂consvar_∂q, (1,), (Face(),), I, j))
            qflux_x[I, j], consflux_x[I, j] = qflux, consflux
            diffusive_heating_x[I, j] = - inv_dx * ( consflux * δ(conjvar, (1,), (Face(),), I, j) + qflux * δ(chempot, (1,), (Face(),), I, j) )
        end
    end
    @with mgr, let (irange, jrange) = (axes(consflux_z, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            ρ_z      = ℑ(m1, (2,), (Face(),), i, J)
            qflux    = salt_flux(thermoflux, ρ_z, ℑ(Γ, (2,), (Face(),), i, J), inv_dz * δ(q, (2,), (Face(),), i, J), inv_dz * δ(p, (2,), (Face(),), i, J))
            consflux = consvar_flux(thermoflux, ρ_z, ℑ(Cp, (2,), (Face(),), i, J), inv_dz * δ(T, (2,), (Face(),), i, J), inv_dz * δ(consvar, (2,), (Face(),), i, J), qflux, ℑ(conjvar, (2,), (Face(),), i, J), ℑ(∂consvar_∂q, (2,), (Face(),), i, J))
            qflux_z[i, J], consflux_z[i, J] = qflux, consflux
            diffusive_heating_z[i, J] = - inv_dz * ( consflux * δ(conjvar, (2,), (Face(),), i, J) + qflux * δ(chempot, (2,), (Face(),), i, J) )
        end
    end

    apply_tracer_bcs!((consflux_x, qflux_x, consflux_z, qflux_z), model, m1, thermo)
    periodize!(model, (consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z))

    return consflux_x, consflux_z, qflux_x, qflux_z, diffusive_heating_x, diffusive_heating_z
end

# composition flux -ρ k_q (∂q - Γ ∂p), with barodiffusion in the consistent scheme only
@inline salt_flux(hf::ThermoFluxConsistent, ρ, Γ, ∂q, ∂p) = - ρ * hf.k_q * (∂q - Γ * ∂p)
@inline salt_flux(hf::ThermoFluxDowngradient, ρ, Γ, ∂q, ∂p) = - ρ * hf.k_q * ∂q
# consvar flux carrying the heat flux JT: JT / conjvar (+ ∂consvar/∂q qflux in the consistent scheme)
@inline heat_to_consvar(::ThermoFluxConsistent, JT, qflux, conj, ∂consvar_∂q) = JT / conj + ∂consvar_∂q * qflux
@inline heat_to_consvar(::ThermoFluxDowngradient, JT, qflux, conj, ∂consvar_∂q) = JT / conj
# consvar flux: carrying -ρ Cp k_T ∂T (consistent), or -ρ k_T ∂consvar (downgradient)
@inline consvar_flux(hf::ThermoFluxConsistent, ρ, Cp, ∂T, ∂cons, qflux, conj, ∂consvar_∂q) =
    heat_to_consvar(hf, - ρ * Cp * hf.k_T * ∂T, qflux, conj, ∂consvar_∂q)
@inline consvar_flux(hf::ThermoFluxDowngradient, ρ, Cp, ∂T, ∂cons, qflux, conj, ∂consvar_∂q) = - ρ * hf.k_T * ∂cons

## TRACER BOUNDARY CONDITIONS

# On wall faces, the fluxes of consvar and q are set by the wall conditions, replacing those computed by `fluxes!`.
# Plain loops: about 2(Mx + Mz) faces, too few for vectorisation or threads to pay off.
function apply_tracer_bcs!((consflux_x, qflux_x, consflux_z, qflux_z), model::FC2D, ρ, thermo)
    (; boundary) = model
    left_right, bottom_top = boundary.walls
    apply_tracer_bcs!((consflux_x, qflux_x), left_right, Val(1), model, ρ, thermo)
    apply_tracer_bcs!((consflux_z, qflux_z), bottom_top, Val(2), model, ρ, thermo)
    return nothing
end
apply_tracer_bcs!(fluxes, ::Nothing, D, model, ρ, thermo) = nothing      # periodic direction: no walls
function apply_tracer_bcs!(fluxes, (lower, upper)::Tuple{Wall, Wall}, D, model, ρ, thermo)
    (; domain) = model
    lo, hi = wall_sides(domain, D)                                         # wall face, adjacent cell, inward normal n
    apply_tracer_bcs!(fluxes, lower.tracers, lo, D, model, ρ, thermo)
    apply_tracer_bcs!(fluxes, upper.tracers, hi, D, model, ρ, thermo)
end
function apply_tracer_bcs!((consflux, qflux), (cons_bc, q_bc), side, D, model::FC2D, ρ, thermo)
    for I in slice(qflux, D)                                               # each position along the wall
        i_face, i_cell = across(I, D, side.face), across(I, D, side.cell)
        wall = wall_state(model, D, side, i_cell, ρ, thermo)              # (; n, d, ρ, p, ∂p) on the wall face
        # q flux: Flux → n F;  Value → face flux with the wall value at distance d
        qflux[i_face] = q_flux_bc(q_bc, bc_value(q_bc, I), thermo, i_cell, wall, model)
        # consvar flux: Flux → n F;  HeatFlux → n F / conjvar (+ cross term);  Value → face flux with T_wall (uses qflux)
        consflux[i_face] = cons_flux_bc(cons_bc, bc_value(cons_bc, I), thermo, i_cell, wall, model, qflux[i_face])
    end
end

# wall face state: inward normal n, distance d to the cell centre, density ρ of the cell, pressure p extrapolated
# hydrostatically to the wall and the normal pressure gradient ∂p = -ρ g_D (g_D: gravity along D)
@inline function wall_state(model::FC2D, ::Val{d}, side, i_cell, ρ, (; p)) where d
    (; dx, dz, space) = model
    (; g) = space
    Δ, g_D = (dx, dz)[d], (zero(g), g)[d]
    ρc = ρ[i_cell]
    return (; n = side.n, d = Δ / 2, ρ = ρc, p = p[i_cell] + side.n * ρc * g_D * Δ / 2, ∂p = - ρc * g_D)
end

# composition: prescribed flux F of ρq, or the face flux with the wall value q_w at distance d
@inline q_flux_bc(::Flux, F, thermo, i_cell, wall, model) = wall.n * F
@inline function q_flux_bc(::Value, q_w, (; q, Γ), i_cell, wall, model)
    (; thermoflux) = model
    return salt_flux(thermoflux, wall.ρ, Γ[i_cell], wall.n * (q[i_cell] - q_w) / wall.d, wall.∂p)
end
# consvar: prescribed flux F of ρ consvar, prescribed heat flux F, or the face flux with the wall value cons_w at
# distance d and T_wall = T(p_wall, cons_w, q_cell). The wall delivers exactly the heat flux (HeatFlux, or Value with
# the consistent scheme): the production between the wall and the cell centre is inside consflux.
@inline cons_flux_bc(::Flux{V, :density}, F, thermo, i_cell, wall, model, qflux) where V = wall.n * F
@inline function cons_flux_bc(::Flux{V, :heat}, F, (; conjvar, ∂consvar_∂q), i_cell, wall, model, qflux) where V
    (; thermoflux) = model
    return heat_to_consvar(thermoflux, wall.n * F, qflux, conjvar[i_cell], ∂consvar_∂q[i_cell])
end
@inline function cons_flux_bc(::Value, cons_w, (; consvar, q, T, conjvar, Cp, ∂consvar_∂q), i_cell, wall, model, qflux)
    (; fluid, thermoflux) = model
    T_w = fluid(:p, :consvar, :q).temperature(wall.p, cons_w, q[i_cell])
    return consvar_flux(thermoflux, wall.ρ, Cp[i_cell], wall.n * (T[i_cell] - T_w) / wall.d, wall.n * (consvar[i_cell] - cons_w) / wall.d,
                        qflux, conjvar[i_cell], ∂consvar_∂q[i_cell])
end

## DISSIPATIVE HEATING

# At interior cells: viscous heating ρε = ∇·(u·τ) - u·F, and the production of consvar
# σ = (ρε + avg(diffusive heating)) / conjvar. Wall faces carry no diffusive heating (zero-gradient halo): the
# energy a wall delivers is conjvar_cell consflux + chempot_cell qflux, see `cons_flux_bc`.
function dissipative_heating!((ρε_, σ_cons_), model::FC2D{F}, ::ThermoFluxConsistent, (; u, w), (uτ_x, uτ_z, F_x, F_z), (; conjvar),
                              (diffusive_heating_x, diffusive_heating_z)) where {F}
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)

    ρε      = alloc!(ρε_, F, mgr, domain, (Cell(), Cell()))
    σ_cons  = alloc!(σ_cons_, F, mgr, domain, (Cell(), Cell()))

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ρεij         = inv_dx * δ(uτ_x, (1,), (Cell(),), i, j) + inv_dz * δ(uτ_z, (2,), (Cell(),), i, j) - ℑ((u, F_x), (1,), (Cell(),), i, j) - ℑ((w, F_z), (2,), (Cell(),), i, j)
            ρε[i, j]     = ρεij
            σ_cons[i, j] = inv(conjvar[i, j]) * ( ρεij + ℑ(diffusive_heating_x, (1,), (Cell(),), i, j) + ℑ(diffusive_heating_z, (2,), (Cell(),), i, j) )
        end
    end

    fill_halo!((ρε, σ_cons), domain, ZeroGradient())

    return ρε, σ_cons
end
# downgradient scheme: no source of consvar (neither viscous nor diffusive heating)
function dissipative_heating!((ρε_, σ_cons_), model::FC2D{F}, ::ThermoFluxDowngradient, (; u, w), (uτ_x, uτ_z, F_x, F_z), _, _) where {F}
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; ℑ, δ) = operators(spatial_scheme)

    ρε      = alloc!(ρε_, F, mgr, domain, (Cell(), Cell()))
    σ_cons  = fill!(alloc!(σ_cons_, F, mgr, domain, (Cell(), Cell())), 0)

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ρε[i, j] = inv_dx * δ(uτ_x, (1,), (Cell(),), i, j) + inv_dz * δ(uτ_z, (2,), (Cell(),), i, j) - ℑ((u, F_x), (1,), (Cell(),), i, j) - ℑ((w, F_z), (2,), (Cell(),), i, j)
        end
    end

    fill_halo!(ρε, domain, ZeroGradient())

    return ρε, σ_cons
end

## MOMENTUM

# Velocity tendencies on faces: -consvar ∇conjvar - q ∇chempot - ∇B - g ẑ - ∇K - ω×u + F/ρ, with face values
# ρ = avg(m₁) and consvar, q = avg(m₂), avg(m₃) / ρ
function velocity_tendencies!((du_, dw_), model::FC2D{F}, (; m), (; conjvar, chempot, B), (K, vortex_x, vortex_z, F_x, F_z)) where {F}
    (; mgr, domain, space, inv_dx, inv_dz, spatial_scheme) = model
    (; g) = space
    (; ℑ, δ) = operators(spatial_scheme)

    du = alloc!(du_, F, mgr, domain, (Face(), Cell()))
    dw = alloc!(dw_, F, mgr, domain, (Cell(), Face()))

    m1, m2, m3 = eachslice(m; dims = 3)

    @with mgr, let (irange, jrange) = (faces(domain, 1), cells(domain, 2))
        @inbounds @vec for I in irange, j in jrange
            v_x      = inv(ℑ(m1, (1,), (Face(),), I, j))
            du[I, j] = - inv_dx * ( v_x * ( ℑ(m2, (1,), (Face(),), I, j) * δ(conjvar, (1,), (Face(),), I, j) + ℑ(m3, (1,), (Face(),), I, j) * δ(chempot, (1,), (Face(),), I, j) )
                                   + δ(B, (1,), (Face(),), I, j) + δ(K, (1,), (Face(),), I, j) ) - vortex_x[I, j] + v_x * F_x[I, j]
        end
    end
    @with mgr, let (irange, jrange) = (cells(domain, 1), faces(domain, 2))
        @inbounds @vec for i in irange, J in jrange
            v_z      = inv(ℑ(m1, (2,), (Face(),), i, J))
            dw[i, J] = - inv_dz * ( v_z * ( ℑ(m2, (2,), (Face(),), i, J) * δ(conjvar, (2,), (Face(),), i, J) + ℑ(m3, (2,), (Face(),), i, J) * δ(chempot, (2,), (Face(),), i, J) )
                                   + δ(B, (2,), (Face(),), i, J) + δ(K, (2,), (Face(),), i, J) ) - g - vortex_z[i, J] + v_z * F_z[i, J]
        end
    end
    return du, dw
end

## DENSITIES

# at interior cells: dm = -∇·(m u) - ∇·(0, consflux, qflux) + (0, σ_cons, 0)
function dm!(dm_, model::FC2D{F}, (; m), (mu, mw, consflux_x, consflux_z, qflux_x, qflux_z, σ_cons)) where {F}
    (; mgr, domain, inv_dx, inv_dz, spatial_scheme) = model
    (; δ) = operators(spatial_scheme)

    dm = alloc!(dm_, F, mgr, domain, (Cell(), Cell()), size(m, 3))

    dm1, dm2, dm3   = eachslice(dm; dims = 3)
    mu1, mu2, mu3   = eachslice(mu; dims = 3)
    mw1, mw2, mw3   = eachslice(mw; dims = 3)

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            dm1[i, j] = - inv_dx * δ(mu1, (1,), (Cell(),), i, j) - inv_dz * δ(mw1, (2,), (Cell(),), i, j)
            dm2[i, j] = - inv_dx * δ(mu2, (1,), (Cell(),), i, j) - inv_dz * δ(mw2, (2,), (Cell(),), i, j) - inv_dx * δ(consflux_x, (1,), (Cell(),), i, j) - inv_dz * δ(consflux_z, (2,), (Cell(),), i, j) + σ_cons[i, j]
            dm3[i, j] = - inv_dx * δ(mu3, (1,), (Cell(),), i, j) - inv_dz * δ(mw3, (2,), (Cell(),), i, j) - inv_dx * δ(qflux_x, (1,), (Cell(),), i, j) - inv_dz * δ(qflux_z, (2,), (Cell(),), i, j)
        end
    end
    return dm
end

## BOUNDARY CONDITIONS

# tendencies vanish in the halo, and for the normal velocity on walls
function dstate_bc!(model::FC2D, (; m, u, w))
    (; domain) = model
    zero_halo!(domain, m, (Cell(), Cell()))
    zero_halo!(domain, u, (Face(), Cell()))
    zero_halo!(domain, w, (Cell(), Face()))
    return (; m, u, w)
end

# State halo (never the interior): ghost velocities realise the wall conditions; m has a zero-gradient halo, so that
# the equation of state only sees valid states. Tracer fluxes through walls come from `apply_tracer_bcs!`.
function state_bc!(model::FC2D, (; m, u, w))
    (; domain, boundary, dx, dz, viscosity) = model
    fill_velocity_halo!(boundary, domain, (u, w), (dx, dz), viscosity.dyn_visc)
    fill_halo!(m, domain, ZeroGradient())
    return (; m, u, w)
end

end # module BoxDynamics
