"""
    BoxInitialize

Initial states of FC2D: the rest density in discrete hydrostatic balance (`balanced_density`), a discretely
divergence-free velocity (`divergence_free_velocity`), and the state (`initialize`).
"""
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
    (; mgr, domain, dx, dz) = model
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    m = alloc(F, mgr, domain, (Cell(), Cell()), 3)
    for i in axes(m, 1), j in interior_cells(domain, 2)
        ρ = density[i, j]
        m[i, j, 1], m[i, j, 2], m[i, j, 3] = ρ, ρ * consvar(x(i), z(j)), ρ * q(x(i), z(j))
    end
    fill_halo!(domain, m, ZeroGradient())
    u, w = initial_velocity(F, velocity, mgr, domain)
    return state_bc!((; m, u, w), model)
end

"""
    u, w = initial_velocity(F, velocity, mgr, domain)

Initial velocity of `initialize`, in arrays of eltype `F` placed by the manager (see `alloc`): a copy of the face
arrays `velocity = (u, w)`, or zero (at rest) when `velocity` is `nothing`.
"""
initial_velocity(F, (u, w)::Tuple, mgr, domain) =
    (copyto!(alloc(F, mgr, domain, (Face(), Cell())), u), copyto!(alloc(F, mgr, domain, (Cell(), Face())), w))
initial_velocity(F, ::Nothing, mgr, domain) = (alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face())))

"""
    ρ = balanced_density(model::FC2D; consvar, q, pref)

Density of the fluid at rest with the tracers consvar(x, z) and q(x, z), on all cells (zero-gradient halo): each column is
in discrete hydrostatic balance (the model's vertical momentum forcing vanishes on every interior face), with
pressure `pref` [Pa] at the height `space.z0` where the geopotential vanishes. As for the anelastic reference state,
the pressure at z0 is that of the piecewise-linear interpolant of the pressures of the cell centers (extrapolated linearly
from the two nearest interior cells if z0 is outside the box).
"""
function balanced_density(model::FC2D{F}; consvar, q, pref) where F
    (; mgr, domain, dx, dz) = model
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    jc = interior_cells(domain, 2)
    ρ = alloc(F, mgr, domain, (Cell(), Cell()))
    cons_column, q_column = zeros(F, ncells(domain, 2)), zeros(F, ncells(domain, 2))
    for i in axes(ρ, 1)
        for j in jc
            cons_column[j], q_column[j] = consvar(x(i), z(j)), q(x(i), z(j))
        end
        ρ[i, :] .= anchored_column(model, cons_column, q_column, pref, jc)
    end
    return fill_halo!(domain, ρ, ZeroGradient())
end

"""
    (u, w) = divergence_free_velocity(model::FC2D, ρ, ψ)

Velocity on the faces with discretely divergence-free mass flux, ∇·(ρu) = 0 in every cell, for the density ρ (e.g.
from `balanced_density`), from a mass streamfunction ψ(X, Z) [kg m⁻¹ s⁻¹] evaluated at the cell vertices:
ρu = δ_z ψ / Δz on u-faces and ρw = -δ_x ψ / Δx on w-faces (ρ averaged from the two cells). The velocity through a
wall vanishes where ψ is constant along it. Faces without two neighboring cells (halo edges) are zero.
"""
function divergence_free_velocity(model::FC2D{F}, ρ, ψ) where F
    (; mgr, domain, dx, dz) = model
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)
    u, w = alloc(F, mgr, domain, (Face(), Cell())), alloc(F, mgr, domain, (Cell(), Face()))
    for I in 2:size(ρ, 1), j in axes(u, 2)
        u[I, j] = (ψ(X(I), Z(j+1)) - ψ(X(I), Z(j))) / (dz * ℑ(ρ, (1,), (Face(),), I, j))
    end
    for i in axes(w, 1), J in 2:size(ρ, 2)
        w[i, J] = -(ψ(X(i+1), Z(J)) - ψ(X(i), Z(J))) / (dx * ℑ(ρ, (2,), (Face(),), i, J))
    end
    return u, w
end

"""
    forcing = face_forcing(inv_d, ρa, ra, qa, ρb, rb, qb, conj_a, chem_a, B_a, conj_b, chem_b, B_b)

Vertical momentum forcing -consvar ∂conjvar - q ∂chempot - ∂B on the face between cells a and b (b above a, at
inverse distance `inv_d`), from the densities (ρ, ρ consvar, ρ q) and (conjvar, chempot, B) of each cell.
"""
@inline function face_forcing(inv_d, ρa, ra, qa, ρb, rb, qb, conj_a, chem_a, B_a, conj_b, chem_b, B_b)
    # the discrete forcing of `velocity_tendencies!` (BoxDynamics), written for one face: the two must agree
    v = inv(ρa + ρb)
    consvar, q = (ra + rb) * v, (qa + qb) * v
    return - inv_d * ((conj_b - conj_a) * consvar + (chem_b - chem_a) * q + (B_b - B_a))
end

"""
    conj, chem, B = cell_thermo(fluid, ρ, cons, q)

(conjvar, chempot, B) in a cell with density ρ, conservative variable `cons` and composition `q`, as in
`thermodynamics!` (BoxDynamics).
"""
@inline function cell_thermo(fluid, ρ, cons, q)
    p = fluid(:v, :consvar, :q).pressure(inv(ρ), cons, q)
    conj = fluid(:p, :consvar, :q).conjugate_variable(p, cons, q)
    chem = fluid(:p, :consvar, :q).modified_chemical_potential(p, cons, q)
    B = fluid(:p, :consvar, :q).specific_enthalpy(p, cons, q) - conj * cons - chem * q
    return conj, chem, B
end

"""
    ρ = anchored_column(model::FC2D, consvar, q, pref, jc)

`balanced_column` with the pressure `pref` at the height z0 (see `balanced_density`): the pressure `ptop` in the top cell is
found by secant iterations on the pressure at z0, which depends on `ptop` almost one for one.
"""
function anchored_column(model, consvar, q, pref, jc)
    (; fluid, dz, space) = model
    (; z0) = space
    j_first, j_last = first(jc), last(jc)
    z_cell(j) = cellpoint(model.domain, dz, 2, j)
    j = clamp(floor(Int, (z0 - z_cell(j_first)) / dz) + j_first, j_first, j_last - 1)      # lower cell of the segment
    function pressure_error(ptop)
        ρ = balanced_column(model, consvar, q, ptop, jc)
        p_1 = fluid(:v, :consvar, :q).pressure(inv(ρ[j]), consvar[j], q[j])
        p_2 = fluid(:v, :consvar, :q).pressure(inv(ρ[j+1]), consvar[j+1], q[j+1])
        return p_1 + (z0 - z_cell(j)) / dz * (p_2 - p_1) - pref
    end
    x0, x1 = pref, pref - pressure_error(pref)
    f0, f1 = pressure_error(x0), pressure_error(x1)
    for _ in 1:20
        (f1 == f0 || abs(f1) <= 8eps(pref)) && break
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, pressure_error(x1)
    end
    return balanced_column(model, consvar, q, x1, jc)
end

"""
    ρ = balanced_column(model::FC2D, consvar, q, ptop, jc)

Densities ρ[j] of a column at rest in discrete balance: the vertical momentum forcing of the model vanishes on every
interior face, given the conservative variable and composition of the interior cells `jc`, and the pressure `ptop`
[Pa] in the top cell. Marches down, solving for each density with secant iterations to round-off.
"""
function balanced_column((; fluid, space, inv_dz), consvar, q, ptop, jc)
    (; g) = space
    ρ = zero(consvar)        # defined in the halo too, which the caller fills
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
