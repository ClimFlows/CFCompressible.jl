module HorizontalEnergies

using CFDomains

struct KineticEnergy{F}
    metric::F # contravariant metric coefficient, i.e. radius^-2 for a sphere of radius a
end

#================== energies ===================#

hvsum(f, hrange, vrange) = sum(kl->sum(i->f(i,kl), hrange), vrange)

function lsum(f, hrange, nz)
    boundary(l,k) = sum(ij->f(ij,l,k,k), hrange)
    bulk = hvsum((ij,l)->f(ij, l, l-1, l), hrange, 2:nz)
    return boundary(1,1)+boundary(nz, nz+1),  bulk
end

function ke_uu(ux, uy, m) 
    bulk = hvsum(axes(m,1), axes(m, 2)) do ij, k
        (ux[ij,k]^2+uy[ij,k]^2)*m[ij,k]
    end
    return bulk/2
end

function ke_uw(ux, uy, W, gx, gy)
    boundary, bulk = lsum(axes(ux,1), size(ux, 2), 1//2) do ij, l, kd, ku
        W[ij,l]*((ux[ij,kd]+ux[ij,ku])*gx[ij,l]+(uy[ij,kd]+uy[ij,ku])*gy[ij,l])
    end
    return (boundary/2+bulk)/2
end

function ke_ww(W, m, gx, gy)
    boundary, bulk = lsum(axes(m,1), size(m, 2), 2) do ij, l, kd, ku
        W[ij,l]^2/(m[ij,kd]+m[ij,ku])*(gx[ij,l]^2+gy[ij,l]^2)
    end
    return boundary*2+bulk
end

function kinetic_energy(KE::KineticEnergy, ux, uy, W, m, gx, gy)
    (; metric) = KE
    return metric*(ke_uu(ux, uy, m)-ke_uw(ux, uy, W, gx, gy)+ke_ww(W, m, gx, gy))
end

#=============== and their gradient w.r.t. the DOFs ==================#

function grad(::typeof(ke_uu), ux, uy, m)    
    return (@. m*ux), (@. m*uy), (@. (ux^2+uy^2)/2)
end

function grad(::typeof(ke_uw), ux, uy, W, gx, gy)
    hrange, nz = axes(ux,1), size(ux, 2)
    dHdux, dHduy, dHdW, dHdgx, dHdgy = map(zero, (ux,uy,W,gx,gy))
    for ij in hrange, k in 1:nz
        ld, lu = k, k+1
        dHdux[ij,k] = W[ij,ld]*gx[ij,ld]+W[ij,lu]*gx[ij,lu]
        dHduy[ij,k] = W[ij,ld]*gy[ij,ld]+W[ij,lu]*gy[ij,lu]
    end
    for ij in hrange, l in 2:nz
        kd, ku = l-1, l
        dHdW[ij,l] = ((ux[ij,ku]+ux[ij,kd])*gx[ij,l] + 
                      (uy[ij,ku]+uy[ij,kd])*gy[ij,l] )/2
        dHdgx[ij,l] = W[ij,l]*(ux[ij,kd]+ux[ij,ku])/2
        dHdgy[ij,l] = W[ij,l]*(uy[ij,kd]+uy[ij,ku])/2
    end
    for ij in hrange, (l,k) in ((1,1), (nz+1,nz))
        dHdW[ij,l] = (ux[ij,k]*gx[ij,l] + uy[ij,k]*gy[ij,l])
        dHdgx[ij,l] = W[ij,l]*ux[ij,k]
        dHdgy[ij,l] = W[ij,l]*uy[ij,k]
    end
    return dHdux, dHduy, dHdW, dHdgx, dHdgy
end

function grad(::typeof(ke_ww), W, m, gx, gy)
    hrange, nz = axes(m,1), size(m, 2)
    ml, dHdW, dHdm, dHdgx, dHdgy = map(zero, (W,W,m,gx,gy))
    for ij in hrange, l in 2:nz
        kd, ku = l-1, l
        ml[ij,l] = (m[ij,kd] + m[ij, ku])/2
    end
    for ij in hrange, (l,k) in ((1,1), (nz+1,nz))
        ml[ij,l] = m[ij,k]/2
    end
    for ij in hrange, k in 1:nz
        ld, lu = k, k+1
        dHdm[ij,k]
    end
end

end # module
