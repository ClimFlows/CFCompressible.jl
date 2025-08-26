module VerticalDynamics

import CFDomains
using CFBatchSolvers: Solvers, SingleSolvers
using MutatingOrNot: void, similar!
using ManagedLoops: @with, @vec

using ..CFCompressible: NewtonSolve

struct VerticalEnergy{Gas,F,A}
    gas::Gas
    gravity::F
    # Jacobian such that m = ρJ∂Φ/∂η , dM = ρJ dΦ d²S
    # different conventions possible, J = a/b with:
    #   a=L^2 (m in Newton or kg, S unitless)
    #   a=1 (m in Pa or kg/m^2, S in m^2)
    #   b=g (M in kg, m in kg or kg/m^2)
    #   b=1 (M in Newton, incorporates gravity, m in Newton or Pa)
    J::F  
    ptop::F # pressure at domain top
    Phis::A # surface geopotential, can be a 2D array
    pb::A   # equilibrium bottom pressure, for non-rigid ground - can be a 2D array
    rhob::F # stiffness of bottom BC: p_bot = pb + rhob*(Phi-Phis)
end

function VerticalEnergy(model, gravity, Phis, pb, rhob)
    # J = 1 assumes that m incorporates gravity and is per unit mass. m has units of Pa
    return VerticalEnergy(model.gas, gravity, one(Phis), model.vcoord.ptop, Phis, pb, rhob)
end

#============ initialize profile ============#

function initial(H::VerticalEnergy, vcoord, case, lon, lat)
    Nz = CFDomains.nlayer(vcoord)
    ps, Phis = case(lon, lat)
    W = zeros(Nz + 1)
    Phi = similar(W)
    m = zeros(Nz)
    S = similar(m)
    for l in 1:(Nz + 1)
        p = CFDomains.pressure_level(2l - 2, ps, vcoord)
        Phi[l], _, _, _ = case(lon, lat, p)
    end
    for k in 1:Nz
        p_down = CFDomains.pressure_level(2k - 2, ps, vcoord)
        p_mid = CFDomains.pressure_level(2k - 1, ps, vcoord)
        p_up = CFDomains.pressure_level(2k, ps, vcoord)
        m[k] = p_down - p_up
        vol = (case(lon, lat, p_up)[1] - case(lon, lat, p_down)[1]) / m[k] # specific volume
        s = H.gas(:p, :v).conservative_variable(p_mid, vol)
        S[k] = s * m[k]
    end
    return Phi, W, m, S
end


#================== energies ===================#

function boundary_energy(H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop) = H
    return J * ((rhob / 2) * (Phi[1] - Phis)^2 - pb * Phi[1] + ptop * Phi[end])
end

function internal_energy(H::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    gas, J = H.gas(:v, :consvar), H.J
    return sum(1:Nz) do k
        vol = J * (Phi[k + 1] - Phi[k]) / m[k] # specific volume
        return m[k] * gas.specific_internal_energy(vol, S[k] / m[k])
    end
end

function potential_energy(::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    return sum(1:Nz) do k
        return (Phi[k + 1] + Phi[k]) * (m[k] / 2)
    end
end

function kinetic_energy(H::VerticalEnergy, Phi, W, m, S)
    Nz = length(m)
    (; gravity) = H
    # W is extensive, on the dual mesh (interfaces)
    return sum(1:(Nz + 1)) do l
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        w = gravity * W[l] / mm
        return mm * w^2 / 2
    end
end

function total_energy(args...)
    return boundary_energy(args...) +
           potential_energy(args...) +
           internal_energy(args...) +
           kinetic_energy(args...)
end

#=============== and their gradient w.r.t. the DOFs ==================#

function grad(::typeof(boundary_energy), H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    dHdPhi[end] = ptop
    dHdPhi[1] = J * rhob * (Phi[1] - Phis) - pb
    return dHdPhi, dHdW, dHdm, dHdS
end

function grad(::typeof(internal_energy), H::VerticalEnergy, Phi, W, m, S)
    (; gas, J) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    for k in 1:Nz
        s = S[k] / m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / m[k]
        p = gas(:v, :consvar).pressure(vol, s)
        Jp = J * p
        dHdPhi[k] += Jp
        dHdPhi[k + 1] -= Jp
        h, _, exner = gas(:p, :consvar).exner_functions(p, s)
        dHdm[k] = h - s * exner
        dHdS[k] = exner
    end
    return dHdPhi, dHdW, dHdm, dHdS
end

function grad(::typeof(potential_energy), H::VerticalEnergy, Phi, W, m, S)
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    for k in 1:Nz
        dHdm[k] = (Phi[k + 1] + Phi[k]) / 2
        dHdPhi[k] += m[k] / 2
        dHdPhi[k + 1] += m[k] / 2
    end
    return dHdPhi, dHdW, dHdm, dHdS
end


function grad(::typeof(kinetic_energy), H::VerticalEnergy, Phi, W, m, S)
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    (; gravity) = H
    for l in 1:(Nz + 1)
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        gw = gravity^2 * (W[l] / mm)
        dHdW[l] = gw
        gw2 = gravity^2 * (W[l] / mm)^2
        l > 1 && (dHdm[l - 1] -= gw2/4)
        l <= Nz && (dHdm[l] -= gw2/4)
    end
    return dHdPhi, dHdW, dHdm, dHdS
end

function grad(::typeof(total_energy), H::VerticalEnergy, Phi, W, m, S)
    (; Phis, pb, rhob, J, ptop, gravity, gas) = H
    dHdPhi, dHdW, dHdm, dHdS = map(zero, (Phi, W, m, S))
    Nz = length(m)
    (; gravity) = H
    for l in 1:(Nz + 1)
        # kinetic
        if l == 1
            mm = m[1] / 2
        elseif l == Nz + 1
            mm = m[Nz] / 2
        else
            mm = (m[l - 1] + m[l]) / 2
        end
        gw = gravity^2 * W[l] / mm
        dHdW[l] += gw
        gw2 = gravity^2 * (W[l] / mm)^2
        l > 1 && (dHdm[l - 1] -= gw2/4)
        l <= Nz && (dHdm[l] -= gw2/4)
    end
    for k in 1:Nz
        # potential
        dHdm[k] += (Phi[k + 1] + Phi[k]) / 2
        dHdPhi[k] += m[k] / 2
        dHdPhi[k + 1] += m[k] / 2
        # internal
        s = S[k] / m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / m[k]
        p = gas(:v, :consvar).pressure(vol, s)
        Jp = J * p
        dHdPhi[k] += Jp
        dHdPhi[k + 1] -= Jp
        h, _, exner = gas(:p, :consvar).exner_functions(p, s)
        dHdm[k] += h - s * exner
        dHdS[k] += exner
    end
    # boundary
    dHdPhi[end] += J * ptop
    dHdPhi[1] += J * (rhob * (Phi[1] - Phis) - pb)

    return dHdPhi, dHdW, dHdm, dHdS
end

const energies = (boundary_energy, internal_energy, potential_energy, kinetic_energy, total_energy)

#============== 1D backward Euler step ============#

# we are solving 
#    x = x⋆ + τf(x)
# state is the current guess
# the residual is (x⋆-x) + τ f(x)

function residual(H, tau, state, Phi_star, W_star)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, Phi, W, m, S)
    rPhi = @. (Phi_star - Phi) + tau * dHdW
    rW = @. (W_star - W) - tau * dHdPhi
#    rPhi = @. 0 * dHdW
#    rW = @. - tau * dHdPhi
    return rPhi, rW
end

function hydrostatic_geopotential!(H, m, S, Phi)
    (; gas, Phis, ptop, J) = H
    pl, Phil = ptop + sum(m)/J, Phis # surface pressure & geopotential
    for k in eachindex(m)
        Phi[k] = Phil
        dp = m[k]/J
        vol = gas(:p, :consvar).specific_volume(pl - dp/2, S[k]/m[k])
        Phil += vol*m[k]/J
        pl -= dp
    end
    Phi[end] = Phil
    return Phi
end

function tridiag_problem!(tridiag, H, tau, (Phi, W, m, S), rPhi, rW)
    (; gas, gravity, rhob, J) = H
    Nz = length(m)
    ml = similar!(tridiag.ml, Phi)
    R = similar!(tridiag.R, Phi)
    B = similar!(tridiag.B, Phi)
    A = similar!(tridiag.A, m)

    # ml
    ml[1] = m[1] / 2
    for l in 2:Nz
        ml[l] = (m[l - 1] + m[l]) / 2
    end
    ml[Nz + 1] = m[Nz] / 2

    # off-diagonal coeffcient A[k]
    for k in 1:Nz
        mk = m[k]
        vol = J * (Phi[k + 1] - Phi[k]) / mk # specific volume
        consvar = S[k] / m[k] # conservative variable
        c2 = gas(:v, :consvar).sound_speed2(vol, consvar)
        A[k] = c2 / mk * (J * tau / vol)^2
    end

    # diagonal coefficient B[l] and reduced residual R[l]
    let ml_g2 = (gravity^-2) * ml[1]
        B[1] = A[1] + ml_g2 + tau^2 * J * rhob # includes spring BC
        R[1] = ml_g2 * rPhi[1] + tau * rW[1]
    end
    for l in 2:Nz
        ml_g2 = (gravity^-2) * ml[l]
        B[l] = (A[l] + A[l - 1]) + ml_g2
        R[l] = ml_g2 * rPhi[l] + tau * rW[l]
    end
    let ml_g2 = (gravity^-2) * ml[Nz + 1]
        B[Nz + 1] = A[Nz] + ml_g2
        R[Nz + 1] = ml_g2 * rPhi[Nz + 1] + tau * rW[Nz + 1]
    end
    return (; A, B, R, ml)
end

function bwd_Euler(tridiag_, H::VerticalEnergy, newton::NewtonSolve, tau, state)
    (; gravity) = H
    (; niter, flip_solve, update_W, verbose) = newton

    inv_tau_g2 = inv(tau * gravity^2)
    (Phi_star, W_star, m, S) = state
    Phi = copy(Phi_star)
    DPhi = Phi-Phi_star
    W = copy(W_star)

    verbose && @info "========= Start Newton iteration =======#" extrema(Phi_star) extrema(Phi) extrema(DPhi)

    function iteration(iter, scratch_)
        @. Phi = Phi_star + DPhi
        rPhi, rW = residual(H, tau, (Phi, W, m, S), Phi_star, W_star)
        (; A, B, R, ml) = scratch = tridiag_problem!(scratch_, H, tau, (Phi, W, m, S), rPhi, rW)

        verbose && @info "Residuals at iter=$iter" L2(R./ml) L2(rW./ml)
        dPhi = SingleSolvers.Thomas(A, B, R, flip_solve)
        @. DPhi += dPhi

        verbose && @info "Update at iter=$iter" extrema(dPhi) extrema(DPhi) dPhi[1] DPhi[1]

        if update_W || iter==niter
            # although the reduced residual R does not depend on W analytically,
            # updating W during the iteration improves the final residual and is cheap
            # W = ml * (Phi-Phi_star) / (tau*g^2)
            Nz = length(m)
            let l = 1, ml = m[l] / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
            for l in 2:Nz
                ml = (m[l - 1] + m[l]) / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
            let l = Nz + 1, ml = m[l - 1] / 2
                W[l] = inv_tau_g2 * (ml * DPhi[l])
            end
        end
        return scratch
    end

    # first iteration
    tridiag = iteration(1, tridiag_)
    # next iterations
    for iter in 2:niter
        iteration(iter, tridiag)
    end    

    return Phi, W, tridiag
end

#================== 3D backward Euler step ================#

function batched_bwd_Euler!(model, ps, state, tau, check=false)
    (; mgr, newton, vcoord, planet, gas, Phis, rhob) = model
    (; niter, flip_solve, verbose) = newton
    ptop, gravity, Jac = vcoord.ptop, planet.gravity, planet.radius^2/planet.gravity
    H = VerticalEnergy(gas, gravity, Jac, ptop, Phis, ps, rhob)

    (mk, ml, Sk, Phi_star, W_star) = state
    Wl, Phil, DPhil, dPhil = copy(W_star), copy(Phi_star), zero(Phi_star), similar(Phi_star)

    tridiag = batched_Newton_iteration(void, mgr, H, mk, Sk, Phi_star, W_star, Phil, DPhil, dPhil, tau, 1, flip_solve, check)

    if tau>0
        for iter in 2:niter
            batched_Newton_iteration(tridiag, mgr, H, mk, Sk, Phi_star, W_star, Phil, DPhil, dPhil, tau, iter, flip_solve, check)
        end
        verbose && @info "Batched update after $niter Newton iterations" extrema(Phi_star) extrema(DPhil) extrema(dPhil)
        # update W
        inv_tau_g2 = inv(tau * gravity^2)
        Wl = @. inv_tau_g2 * ml * DPhil
    else
        Wl = copy(W_star)
    end
    return Phil, Wl, tridiag
end

function batched_Newton_iteration(tridiag_, mgr, H, mk::M, Sk::M, Phi_star::M, W_star::M, Phil::M, DPhil::M, dPhil::M, tau, iter, flip_solve, check) where { M<:AbstractMatrix}
    # present 2D arrays as 3D
    reshp(x::M) = reshape(x, (size(x,1), 1, size(x, 2)))
    return batched_Newton_iteration(tridiag_, mgr, H, map(reshp, (mk, Sk, Phi_star, W_star, Phil, DPhil, dPhil))..., tau, iter, flip_solve, check)
end

function batched_Newton_iteration(tridiag_, mgr, H, mk, Sk, Phi_star, W_star, Phil, DPhil, dPhil, tau, iter, flip_solve, check)
    @. Phil = Phi_star + DPhil
    (; R, A, B) = tri = batched_tridiag_problem!(tridiag_, mgr, H, (mk, Sk, Phil), Phi_star, W_star, tau)
    Solvers.Thomas!(dPhil, A, B, R, flip_solve)
    @. DPhil += dPhil

    # verify batched_tridiag_problem! and batched_Thomas
    check && for i in axes(mk,1), j in axes(mk,2)
        H_ij = VerticalEnergy(gas, gravity, Jac, ptop, Phis[i,j], ps[i,j], rhob)
        column = (Phil[i,j,:], Wl[i,j,:], mk[i,j,:], Sk[i,j,:])
        rPhi, rW = residual(H_ij, tau, column, Phi_star[i,j,:], W_star[i,j,:])
        scratch = tridiag_problem!(void, H_ij, tau, column, rPhi, rW)
        dPhi = SingleSolvers.Thomas(A[i,j,:], B[i,j,:], R[i,j,:], flip_solve)

        function isok(tag, a,b) 
            if sum(abs, b-a) > 1e-6*sum(abs, a)
                @warn "at iter=$iter, $tag not ok" i j sum(abs, a) sum(abs, b) sum(abs, b-a)
            end
        end

        isok(:A, scratch.A, A[i,j,:])
        isok(:B, scratch.B, B[i,j,:])

        if(sum(abs, dPhi)) > 1e-4 # do not check when dPhi is too small
            isok(:R, scratch.R, R[i,j,:])
            isok(:dPhi, dPhi, dPhil[i,j,:])
        end
    end

    return tri
end

#=
function batched_tridiag_problem!(tridiag, mgr, H, state, Phi_star::M, W_star::M, tau) where { M<:AbstractMatrix}
    # present 2D arrays as 3D
    reshp(x::M) = reshape(x, (size(x,1), 1, size(x, 2)))
    return batched_tridiag_problem!(tridiag, mgr, H, map(reshp, state), reshp(Phi_star), reshp(W_star), tau)
end
=#

function batched_tridiag_problem!(tridiag, mgr, H, state, Phi_star, W_star, tau)
    (; Phis, pb, rhob, J, ptop, gravity, gas) = H  # Phis and pb are 2D arrays
    (m, S, Phi) = state # W terms cancel out => ignore W and let W=0

    Jp = similar!(tridiag.Jp, m)
    A = similar!(tridiag.A, m)
    @with mgr let (irange, jrange, krange) = axes(m)
        @inbounds for j in jrange, k in krange
            @vec for i in irange
                invm = inv(m[i,j,k])
                consvar = invm * S[i,j,k] 
                vol = J * invm * (Phi[i,j,k + 1] - Phi[i,j,k])
                p = @inline gas(:v, :consvar).pressure(vol, consvar)
                Jp[i,j,k] = J * p
                # off-diagonal coeffcient A[k]
                c2 = @inline gas(:p, :v).sound_speed2(p, vol)
                A[i,j,k] = c2 * invm * (J * tau / vol)^2
            end
        end
    end

    R = similar!(tridiag.R, Phi)
    B = similar!(tridiag.B, Phi)
    #=@with mgr=# let (irange, jrange, lrange) = axes(Phi)
        Nz = size(m,3)
        @inbounds for j in jrange, l in lrange
            #=@vec=# for i in irange
                if l == 1
                    Jp_up = Jp[i,j,l] 
                    Jp_down = J * (pb[i,j] - rhob * (Phi[i,j,1] - Phis[i,j]) ) 
                    Al = A[i,j,l] + tau^2 * J * rhob # bottom BC
                    ml = m[i,j,1] / 2
                elseif l == Nz + 1
                    Jp_up = J*ptop
                    Jp_down = Jp[i,j,l-1] 
                    Al = A[i,j,l-1]
                    ml = m[i,j,Nz] / 2
                else
                    Jp_up = Jp[i,j,l] 
                    Jp_down = Jp[i,j,l-1] 
                    Al = A[i,j,l]+A[i,j,l-1]
                    ml = (m[i,j,l - 1] + m[i,j,l]) / 2
                end
                force = ml + (Jp_up-Jp_down) # downward force
                ml_g2 = (gravity^-2) * ml
                R[i,j,l] = ml_g2 * (Phi_star[i,j,l] - Phi[i,j,l]) + tau * (W_star[i,j,l] - tau * force)
                B[i,j,l] = ml_g2 + Al
            end
        end
    end

    return (; Jp, R, A , B)
end

# reference implementation calling the 1D solver on each column
function ref_bwd_Euler!(model, ps, state, tau)
    (; newton, vcoord, planet, gas, Phis, rhob) = model
    ptop, gravity, Jac = vcoord.ptop, planet.gravity, planet.radius^2/planet.gravity

    (mk, ml, Sk, Phil, Wl) = state

    function bwd(scratch, i, j)
        H = VerticalEnergy(gas, gravity, Jac, ptop, Phis[i,j], ps[i,j], rhob)
        column = (Phil[i,j,:], Wl[i,j,:], mk[i,j,:], Sk[i,j,:])
        bwd_Euler(scratch, H, newton, tau, column)
    end

    Phil_new, Wl_new = similar(Phil), similar(Wl)

    if tau>0
        let tridiag = bwd(void, 1, 1)[3] # allocate `tridiag` scratch space
            for i in axes(mk,1), j in axes(mk,2)
                Phil_new[i,j,:], Wl_new[i,j,:], _ = bwd(tridiag, i,j)
                newton = (typeof(newton))(newton.niter, newton.flip_solve, newton.update_W, false) # switch off verbosity
            end
        end
    else
        @. Phil_new = Phil
        @. Wl_new = Wl
    end
    return Phil_new, Wl_new
end


L2(x) = sqrt(sum(x->x^2, x))

end # module
