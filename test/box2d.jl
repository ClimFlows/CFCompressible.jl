# Property tests for FC2D (2D box). Self-contained: run with
#   julia -t 4 --project=<env with CFCompressible> box2d.jl
# or via runtests.jl / TraceableEnergetics/test/runtests.jl
using Test

using CFCompressible
using LoopManagers: MultiThread, VectorizedCPU
using CFBoxes, CFDiffusionSchemes, ClimFluids, CFTimeSchemes, CookBooks, ForwardDiff
using CFPlanets: Tank2D
import MutatingOrNot
using MutatingOrNot: void

include("box_helpers.jl")

#####
##### Configuration: small box, all physics switched on, closed (free-slip, insulating) walls
#####

const mgr = MultiThread(VectorizedCPU())
const space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
const Lx, Lz, g = space.Lx, space.Lz, space.g
const T0, q0, ρ0, U = 280.0, 0.035, 1000.0, 0.1

# nonlinear EOS with compressibility, thermobaricity, cabbeling and haline contraction
const fluid_params = (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1e-3,
    α_T = 2e-4, α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7)
const fluid = NonlinearBinaryFluid(:potential_temperature, fluid_params)

const thermoflux = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5)

# round-off level of the momentum tendency: B = h - πθ - μq is a difference of O(Cp θ) numbers, divided by Δx
momentum_roundoff(model) = 100 * eps() * fluid.Cp0 * T0 / min(model.dx, model.dz)

test_box(Mx = 32, Mz = 16; boundary = (Periodic, Bounded)) = Box2D(; Mx, Mz, Hx = 1, Hz = 1, boundary)

# closed (impermeable, insulating, no salt flux) walls
closed_wall(tangential = FreeSlip()) = Wall(; tangential, tracers = (Flux(0.0), Flux(0.0)))   # (consvar, q)

# smooth, x-asymmetric profiles; z = 0 at the bottom
θ_rest(z) = T0 + 1.0 * z / Lz
q_rest(z) = q0 - 0.002 * z / Lz
θ_active(x, z) = θ_rest(z) + 0.3 * sin(2π * x / Lx) * cos(π * z / Lz) + 0.1 * cos(4π * x / Lx + 1) * sin(2π * z / Lz)
q_active(x, z) = q_rest(z) + 0.001 * cos(2π * x / Lx + 0.5) * cos(π * z / Lz)

#####
##### FC2D configuration and states
#####

function fc_model(; Mx = 32, Mz = 16, mgr = mgr, advection_scheme = AdvEnergyCons(), thermoflux_scheme = thermoflux,
                   boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = test_box(Mx, Mz; boundary)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return FC2D((; mgr, domain, space, fluid, advection_scheme, thermoflux_scheme,
        viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3, bulk_visc = 1e-3), boundary_conditions))
end

fc_tend!(model, dstate, scratch, state) = CFCompressible.tendencies!(dstate, scratch, model, state, 0.0)

# state (m = (ρ, ρθ, ρq), u, w); w vanishes on the walls
function fc_state(model; active = true)
    (; domain, dx, dz) = model
    a = active ? 1.0 : 0.0
    # near-hydrostatic: ρ ≈ ρ0 (1 + g (Lz - z) / c²) with c² = v0 / α_p, plus a small perturbation
    c² = fluid.v0 / fluid.α_p
    ρ(x, z) = ρ0 * (1 + g * (Lz - z) / c² + a * 1e-4 * sin(2π * x / Lx + 0.7) * cos(π * z / Lz))
    θ(x, z) = active ? θ_active(x, z) : θ_rest(z)
    q(x, z) = active ? q_active(x, z) : q_rest(z)
    x(i) = xpoint(domain, dx, i); z(j) = zpoint(domain, dz, j)
    X(I) = Xpoint(domain, dx, I); Z(J) = Zpoint(domain, dz, J)

    m = alloc_xz(Float64, domain, 3)
    u = alloc_Xz(Float64, domain)
    w = alloc_xZ(Float64, domain)
    for i in axes(m, 1), j in axes(m, 2)
        r = ρ(x(i), z(j))
        m[i, j, 1] = r
        m[i, j, 2] = r * θ(x(i), z(j))
        m[i, j, 3] = r * q(x(i), z(j))
    end
    for I in axes(u, 1), j in axes(u, 2)
        u[I, j] = a * U * (cos(π * z(j) / Lz) + 0.3 * sin(2π * X(I) / Lx) * sin(π * z(j) / Lz))
    end
    for i in axes(w, 1), J in axes(w, 2)
        w[i, J] = a * 0.5U * sin(π * Z(J) / Lz) * cos(2π * x(i) / Lx + 0.2)
    end
    # impermeable walls
    boundary_topology(domain)[2] isa Bounded && (w[:, wfaces(domain)[2][[1, end]]] .= 0)
    boundary_topology(domain)[1] isa Bounded && (u[ufaces(domain)[1][[1, end]], :] .= 0)
    return (; m, u, w)
end

# per-cell energy ΔxΔz ρ (K + e(v, θ, q) + Φ), with e = h - p v and K = (avg_x u² + avg_z w²) / 2
function fc_cell_energies(model, (; m, u, w))
    (; domain, dx, dz, fluid) = model
    ic, jc = cells(domain)
    return [begin
        ρ, θ, q = m[i, j, 1], m[i, j, 2] / m[i, j, 1], m[i, j, 3] / m[i, j, 1]
        v = inv(ρ)
        p = fluid(:v, :consvar, :q).pressure(v, θ, q)
        e = fluid(:p, :consvar, :q).specific_enthalpy(p, θ, q) - p * v
        K = (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4
        ρ * (K + e + g * zpoint(domain, dz, j)) * dx * dz
    end for i in ic, j in jc]
end

function fc_kinetic(model, (; m, u, w))
    (; domain, dx, dz) = model
    ic, jc = cells(domain)
    return sum(m[i, j, 1] * (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4 * dx * dz for i in ic, j in jc)
end

# walls exercising every kind of condition (consvar = θ for this fluid)
const forced_box = (;
    left   = Wall(tangential = TangentialVelocity(z -> 0.02 * z), tracers = (Value(T0 + 0.5), Value(q0))),
    right  = Wall(tangential = NoSlip(), tracers = (Value(z -> T0 - 0.5 + 0.1 * z), Flux(1e-6))),
    bottom = Wall(tangential = TangentialVelocity(0.05), tracers = (HeatFlux(x -> 40.0 + 10.0 * cos(2π * x / Lx)), Flux(0.0))),
    top    = Wall(tangential = Stress(x -> 2e-4 * sin(2π * x / Lx)), tracers = (Flux(1e-3), Flux(-1e-6))))

# discrete mass divergence ∇·(ρu) at interior cells, and the size of its terms
function fc_divergence(model, (; m, u, w))
    (; domain, dx, dz) = model
    ρ = view(m, :, :, 1)
    ic, jc = cells(domain)
    ρu(I, j) = u[I, j] * (ρ[I-1, j] + ρ[I, j]) / 2
    ρw(i, J) = w[i, J] * (ρ[i, J-1] + ρ[i, J]) / 2
    div = [(ρu(east(domain, i), j) - ρu(i, j)) / dx + (ρw(i, j+1) - ρw(i, j)) / dz for i in ic, j in jc]
    scale = max(maximum(abs, ρu(i, j) / dx for i in ic, j in jc), maximum(abs, ρw(i, j) / dz for i in ic, j in jc), floatmin())
    return div, scale
end

#####
##### Tests
#####

@testset "FC2D" begin
    model = fc_model(mgr = VectorizedCPU())        # closed walls, periodic in x
    active = fc_state(model)
    box = fc_model(boundary = (Bounded, Bounded), walls = forced_box)    # threaded: memory safety also catches races

    @testset "closed walls, periodic in x" begin
        test_memory_safety(fc_tend!, model, active)
        test_symmetries(fc_tend!, model, active)
        test_walls(fc_tend!, model, active)
        test_conservation(fc_tend!, model, active, (; mass = 1, consvar = 2, q = 3))
        r = test_energy(fc_tend!, fc_cell_energies, model, active)
        test_no_allocations(fc_tend!, model, active)
        @info "FC2D relative dE/dt residual" r
    end

    @testset "forced walls, closed box, threaded" begin
        s = fc_state(box)
        test_memory_safety(fc_tend!, box, s)
        test_walls(fc_tend!, box, s)
        test_conservation(fc_tend!, box, s, (; mass = 1, consvar = 2, q = 3))   # wall fluxes enter the budgets
        r = test_energy_budget(fc_tend!, fc_cell_energies, box, s)
        @test abs(r.input) > 1e-8                                            # the forcing is felt
        _, scratch = evaluate(fc_tend!, box, s)
        @test wall_input(box, scratch, 2, 1).heat ≈ 40.0 * Lx rtol = 1e-12     # HeatFlux delivers exactly ∫F dx
    end

    @testset "schemes" begin
        # the other advection and heat flux schemes, together (independent code paths): same properties, except
        # energy conservation, which they don't have by design (AdvEnstrophyCons conserves enstrophy; ThermoFluxDowngradient
        # has no heating, σ = 0)
        variant = fc_model(mgr = VectorizedCPU(), advection_scheme = AdvEnstrophyCons(), thermoflux_scheme = ThermoFluxDowngradient(k_T = 1e-4, k_q = 1e-5))
        test_memory_safety(fc_tend!, variant, active; reuse = false)
        test_symmetries(fc_tend!, variant, active)
        test_walls(fc_tend!, variant, active)
        test_conservation(fc_tend!, variant, active, (; mass = 1, consvar = 2, q = 3))
    end

    @testset "initial states" begin
        # at rest, stratified, between closed walls: discretely balanced, with ptop in the top cell of every column
        consvar, q = (x, z) -> θ_rest(z), (x, z) -> q_rest(z)
        s = CFCompressible.initialize(model; density = CFCompressible.balanced_density(model; consvar, q, ptop = 1e5), consvar, q)
        ds, scratch = evaluate(fc_tend!, model, s)
        (ic, jc) = cells(model.domain)
        @test maximum(abs, scratch.thermo.p[ic, last(jc)] .- 1e5) < 1e-9
        @test maximum(abs, ds.w[wfaces(model.domain)...]) < momentum_roundoff(model)
        @test maximum(abs, ds.u[ufaces(model.domain)...]) < momentum_roundoff(model)
        # perturbation from a streamfunction vanishing on all walls: ∇·(ρu) = 0 discretely, no flow through walls
        ψ(X, Z) = 1e-3 * ρ0 * U * Lz * sin(2π * X / Lx) * sin(π * Z / Lz)
        for m in (model, box)
            density = CFCompressible.balanced_density(m; consvar = θ_active, q = q_active, ptop = 1e5)
            u′, w′ = CFCompressible.divergence_free_velocity(m, density, ψ)
            @test maximum(abs, w′[:, wfaces(m.domain)[2][[1, end]]]) < 1e-15 * maximum(abs, w′)
            s = CFCompressible.initialize(m; density, consvar = θ_active, q = q_active, velocity = (u′, w′))
            div, scale = fc_divergence(m, s)
            @test all_finite(s) && maximum(abs, div) <= 1e-12 * scale
        end
    end

    @testset "RK4 integration" begin
        test_integration(model, active, 2e-4, 20, fc_cell_energies, (1, 3), fc_kinetic)
    end

    @testset "diagnostics" begin
        test_diagnostics((; book = CFCompressible.diagnostics(model), internal = CFCompressible.internal_diagnostics(model),
                          info = CFCompressible.diagnostic_info(model)), fc_tend!, model, active, fluid_params)
        _, _, te = CFCompressible.energies!((void, void, void), active, model)
        mass = sum(active.m[cells(model.domain)..., 1]) * model.dx * model.dz
        # the test energy uses Φ = g z, the diagnostics Φ = g (z - Lz)
        @test sum(te[cells(model.domain)...]) * model.dx * model.dz ≈ sum(fc_cell_energies(model, active)) - g * Lz * mass rtol = 1e-12
    end

    @testset "driver loop" begin
        params = (; Nslice = 2, slice_size = 1e-3, dt_max = 2e-4, cfl = 0.5)
        @test all_finite(CFCompressible.loop(model, active, CFTimeSchemes.RungeKutta4(model), params))
    end
end
