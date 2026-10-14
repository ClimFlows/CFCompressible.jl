module BoxInitialize

using ..CFCompressible: FC2D
using ..CFCompressible.BoxDynamics: state_bc!
using CFBoxes

export initialize, balanced_density, divergence_free_velocity

"""
    state = initialize(model::FC2D; density, consvar, q, velocity = nothing)

State (m, u, w) of the FC2D model from the density (e.g. the rest density from `balanced_density`), the tracers
consvar(x, z) and q(x, z), functions of position with z = 0 at the bottom and x = 0 at the left, and the initial
velocity `velocity = (u, w)`, face arrays shaped like u and w (e.g. from `divergence_free_velocity`, plus a mean
flow); `nothing` is the fluid at rest. The halo is set by the boundary conditions.
"""
function initialize(model::FC2D{F}; density, consvar, q, velocity = nothing) where F
    (; domain, dx, dz) = model
    x(i) = xpoint(domain, dx, i); z(j) = zpoint(domain, dz, j)
    m = alloc_xz(F, domain, 3)
    for i in axes(m, 1), j in zrange_interior(domain)
        ρ = density[i, j]
        m[i, j, 1], m[i, j, 2], m[i, j, 3] = ρ, ρ * consvar(x(i), z(j)), ρ * q(x(i), z(j))
    end
    fill_halo!(m, domain, ZeroGradient())
    u, w = initial_velocity(velocity, F, domain)
    return state_bc!(model, (; m, u, w))
end

# initial velocity: a copy of the given face arrays, or zero (at rest)
initial_velocity((u, w)::Tuple, F, domain) = (copy(u), copy(w))
initial_velocity(::Nothing, F, domain) = (fill!(alloc_Xz(F, domain), 0), fill!(alloc_xZ(F, domain), 0))

"""
    ρ = balanced_density(model::FC2D; consvar, q, ptop)

Density of the fluid at rest with the tracers consvar(x, z) and q(x, z), on all cells (zero-gradient halo): each column is
in discrete hydrostatic balance (the model's vertical momentum forcing vanishes on every interior face), with
pressure `ptop` [Pa] in the top cell.
"""
function balanced_density(model::FC2D{F}; consvar, q, ptop) where F
    (; domain, dx, dz) = model
    x(i) = xpoint(domain, dx, i); z(j) = zpoint(domain, dz, j)
    jc = zrange_interior(domain)
    ρ = alloc_xz(F, domain)
    cons_column, q_column = zeros(F, zdim(domain)), zeros(F, zdim(domain))
    for i in axes(ρ, 1)
        for j in jc
            cons_column[j], q_column[j] = consvar(x(i), z(j)), q(x(i), z(j))
        end
        ρ[i, :] .= balanced_column(model, cons_column, q_column, ptop, jc)
    end
    return fill_halo!(ρ, domain, ZeroGradient())
end

"""
    (u, w) = divergence_free_velocity(model::FC2D, ρ, ψ)

Velocity on the faces with discretely divergence-free mass flux, ∇·(ρu) = 0 in every cell, for the density ρ (e.g.
from `balanced_density`), from a mass streamfunction ψ(X, Z) [kg m⁻¹ s⁻¹] evaluated at the cell vertices:
ρu = δ_z ψ / Δz on u-faces and ρw = -δ_x ψ / Δx on w-faces (ρ averaged from the two cells). The velocity through a
wall vanishes where ψ is constant along it. Faces without two neighbouring cells (halo edges) are zero.
"""
function divergence_free_velocity(model::FC2D{F}, ρ, ψ) where F
    (; domain, dx, dz) = model
    X(I) = Xpoint(domain, dx, I); Z(J) = Zpoint(domain, dz, J)
    u, w = fill!(alloc_Xz(F, domain), 0), fill!(alloc_xZ(F, domain), 0)
    for I in 2:size(ρ, 1), j in axes(u, 2)
        u[I, j] = (ψ(X(I), Z(j+1)) - ψ(X(I), Z(j))) / (dz * avg_X(ρ, I, j))
    end
    for i in axes(w, 1), J in 2:size(ρ, 2)
        w[i, J] = -(ψ(X(i+1), Z(J)) - ψ(X(i), Z(J))) / (dx * avg_Z(ρ, i, J))
    end
    return u, w
end

# Vertical momentum forcing - consvar ∂conjvar - q ∂chempot - ∂B on the face between cells a and b (b above a, at
# inverse distance inv_d), from the densities (ρ, ρ consvar, ρ q) and (conjvar, chempot, B) of each cell: the
# discrete forcing of `velocity_tendencies!` (BoxDynamics), written for one face; the two must agree.
@inline function face_forcing(inv_d, ρa, ra, qa, ρb, rb, qb, conj_a, chem_a, B_a, conj_b, chem_b, B_b)
    v = inv(ρa + ρb)
    consvar, q = (ra + rb) * v, (qa + qb) * v
    return - inv_d * ((conj_b - conj_a) * consvar + (chem_b - chem_a) * q + (B_b - B_a))
end

# (conjvar, chempot, B) in a cell with density ρ, conservative variable cons and composition q, as in
# `thermodynamics!` (BoxDynamics)
@inline function cell_thermo(fluid, ρ, cons, q)
    p = fluid(:v, :consvar, :q).pressure(inv(ρ), cons, q)
    conj = fluid(:p, :consvar, :q).conjugate_variable(p, cons, q)
    chem = fluid(:p, :consvar, :q).modified_chemical_potential(p, cons, q)
    B = fluid(:p, :consvar, :q).specific_enthalpy(p, cons, q) - conj * cons - chem * q
    return conj, chem, B
end

# Densities ρ[j] of a column at rest in discrete balance: the vertical momentum forcing of the model vanishes on
# every interior face, given the conservative variable and composition of the interior cells `jc`, and the pressure
# `ptop` in the top cell. Marches down, solving for each density with secant iterations to round-off.
function balanced_column((; fluid, space, inv_dz), consvar, q, ptop, jc)
    (; g) = space
    ρ = similar(consvar)
    jt = last(jc)
    ρ[jt] = inv(fluid(:p, :consvar, :q).specific_volume(ptop, consvar[jt], q[jt]))
    for j in jt-1:-1:first(jc)
        ρb, cb, qb = ρ[j+1], consvar[j+1], q[j+1]
        thermo_b = cell_thermo(fluid, ρb, cb, qb)
        forcing(ρa) = face_forcing(inv_dz, ρa, ρa * consvar[j], ρa * q[j], ρb, ρb * cb, ρb * qb,
                                   cell_thermo(fluid, ρa, consvar[j], q[j])..., thermo_b...) - g
        # hydrostatic first guess, dρ = ρ g dz / c²
        c² = fluid(:v, :consvar, :q).sound_speed2(inv(ρb), cb, qb)
        x0, x1 = ρb, ρb * (1 + g / (inv_dz * c²))
        f0, f1 = forcing(x0), forcing(x1)
        for _ in 1:50
            (f1 == f0 || abs(x1 - x0) <= 4eps(x1)) && break
            x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
            f0, f1 = f1, forcing(x1)
        end
        ρ[j] = x1
    end
    return ρ
end

end # module BoxInitialize
