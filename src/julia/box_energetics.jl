"""
    BoxEnergetics

Pointwise energies of FC2D (`energies!`), mirroring `CFAnelastics.energies!` for AN2D.
"""
module BoxEnergetics

using ..CFCompressible: FC2D
using CFBoxes
using ManagedLoops: @vec, @with

"""
    ke, pe, te = energies!((ke, pe, te), model::FC2D, state)

Energies per unit volume (J m⁻³) at cell centers: kinetic ke = ρ (⟨u²⟩ₓ + ⟨w²⟩_z)/2, potential and internal
pe = ρ (e + Φ) with e = h - p v the specific internal energy and Φ = g (z - Lz), and total te = ke + pe.
Their sum over the interior, times Δx Δz, is the energy conserved by the energy-consistent schemes.
"""
function energies!((ke_, pe_, te_), model::FC2D{F}, state) where {F}
    ke = kinetic_energy!(ke_, model, state)
    pe = potential_energy!(pe_, model, state)
    te = alloc!(te_, F, model.mgr, model.domain, (Cell(), Cell()))
    @. te = ke + pe
    return ke, pe, te
end

"""
    ke = kinetic_energy!(ke, model::FC2D, state)

Kinetic energy per unit volume ke = ρ (ℑx(u²) + ℑz(w²)) / 2 [J m⁻³] at the cells. Periodized.
"""
function kinetic_energy!(ke_, model::FC2D{F}, (; m, u, w)) where {F}
    (; mgr, domain) = model
    ke = alloc!(ke_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            ke[i, j] = m[i, j, 1] * (ℑ((u, u), (1,), (Cell(),), i, j) + ℑ((w, w), (2,), (Cell(),), i, j)) / 2
        end
    end
    periodize!(domain, ke)
    return ke
end

"""
    pe = potential_energy!(pe, model::FC2D, state)

Potential and internal energy per unit volume pe = ρ (h(p, consvar, q) - p v + Φ) [J m⁻³] at the cells, with
p = p(v, consvar, q) and Φ = g (z - Lz).
"""
function potential_energy!(pe_, model::FC2D{F}, (; m)) where {F}
    (; mgr, domain, fluid, space, dz) = model
    (; g, Lz) = space
    pe = alloc!(pe_, F, mgr, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @inbounds @vec for i in irange, j in jrange
            v = inv(m[i, j, 1])
            consvar, q = m[i, j, 2] * v, m[i, j, 3] * v
            p = fluid(:v, :consvar, :q).pressure(v, consvar, q)
            e = fluid(:p, :consvar, :q).specific_enthalpy(p, consvar, q) - p * v
            pe[i, j] = m[i, j, 1] * (e + g * (cellpoint(domain, dz, 2, j) - Lz))
        end
    end
    return pe
end

end # module BoxEnergetics
