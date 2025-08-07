using Test
using Enzyme

using CFCompressible: NewtonSolve
using CFCompressible.VerticalDynamics: initial, grad, energies, total_energy, VerticalEnergy

using ClimFluids: IdealPerfectGas
using CFDomains: SigmaCoordinate
using ClimFlowsTestCases: Jablonowski06 as Case

function test_grad(fun, H, state)
    dE = grad(fun, H, state...)
    E(state...) = fun(H, state...)
    dE_ = Enzyme.gradient(Reverse, E, state...)
    for (var, dHdX, dHdX_) in zip((:Φ, :W, :m, :S), dE, dE_)
        maximum(abs, dHdX_)>0 && @info "∂$fun/∂$var" extrema(dHdX) extrema(dHdX_) # show only non-zero derivatives
        @test dHdX.+1e-10 ≈ (dHdX.+1e-10) # we have to shift because some derivatives are exactly zero
    end
end

function test_canonical(H, state)
    (Phi, W, m, S) = state
    (dHdPhi, dHdW, _, _) = grad(total_energy, H, state...)
    function f(tau)
        Phitau = @. Phi - tau * dHdW
        Wtau = @. W + tau * dHdPhi
        return total_energy(H, Phitau, Wtau, m, S)
    end
    ((dH,),) = Enzyme.autodiff(set_runtime_activity(Reverse), Const(f), Active, Active(0.0))

    @test dH*1e18<f(0)
    return nothing
end

function test(H, state)
    @testset "Gradients" begin
        for fun in energies
            fun(H, state...)
            grad(fun, H, state...)
            test_grad(fun, H, state)
        end
    end
    @testset "Total energy" begin
        test_canonical(H, state)
    end
    return nothing
end

params = (gravity=9.81, ps=101325.0, Phis=42.0, rhob=1e5, kappa=2/7, Cp=1000.0, p0=1e5,
          T0=300.0, ptop=1000.0, consvar=:temperature, levels=30, niter=3, update_W=false, verbose=true)

model = (gas=IdealPerfectGas(params), vcoord=SigmaCoordinate(params.levels, params.ptop))
H = VerticalEnergy(model, params.gravity, params.Phis, params.ps, params.rhob)
state = (Phi, W, m, S) = initial(H, model.vcoord, Case(), 0.0, 0.0)
newton = NewtonSolve(; params...)

@testset "1D Hamiltonian structure" begin
    test(H, state)
end
