module BoxEnergetics
# Pointwise energies of the FC2D model, mirroring `CFAnelastics.energies!` for AN2D

using ..CFCompressible: FC2D
using CFBoxes
using ManagedLoops: @vec, @with

"""
    ke, pe, te = energies!((ke, pe, te), state, model::FC2D)

Energies per unit volume (J m⁻³) at cell centres: kinetic ke = ρ (⟨u²⟩ₓ + ⟨w²⟩_z)/2, potential and internal
pe = ρ (e + Φ) with e = h - p v the specific internal energy and Φ = g (z - Lz), and total te = ke + pe.
Their sum over the interior, times Δx Δz, is the energy conserved by the energy-consistent schemes.
"""
function energies!((ke_, pe_, te_), state, model::FC2D{F}) where {F}
    ke = kinetic_energy!(ke_, state, model)
    pe = potential_energy!(pe_, state, model)
    te = alloc!(te_, F, model.domain, (Cell(), Cell()))
    @. te = ke + pe
    return ke, pe, te
end

# KE = ρ (⟨u²⟩ₓ + ⟨w²⟩_z) / 2
function kinetic_energy!(ke_, (; m, u, w), model::FC2D{F}) where {F}
    (; mgr, domain) = model
    ke = alloc!(ke_, F, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (cells(domain, 1), cells(domain, 2))
        @vec for i in irange, j in jrange
            ke[i, j] = 0.5 * m[i, j, 1] * (ℑ((u, u), (1,), (Cell(),), i, j) + ℑ((w, w), (2,), (Cell(),), i, j))
        end
    end
    periodize!(model, ke)
    return ke
end

# PE = ρ (h(p, consvar, q) - p v + Φ), with p = p(v, consvar, q)
function potential_energy!(pe_, (; m), model::FC2D{F}) where {F}
    (; mgr, domain, fluid, space, dz) = model
    (; g, Lz) = space
    pe = alloc!(pe_, F, domain, (Cell(), Cell()))
    @with mgr, let (irange, jrange) = (axes(pe, 1), axes(pe, 2))
        @vec for i in irange, j in jrange
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
