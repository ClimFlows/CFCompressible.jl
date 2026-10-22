#=
Property tests for the FC2D tendencies (2D box): memory safety, impermeable walls, conservation, energy
conservation and budget, no allocations. Self-contained: run with
  julia -t 4 --project=<env with CFCompressible> box2d.jl
or via runtests.jl / TraceableEnergetics/test/runtests.jl
=#
using Test

using CFCompressible
using CFCompressible: CFTimeSchemes
using LoopManagers: MultiThread, VectorizedCPU
using CFBoxes, CFDiffusionSchemes, ClimFluids, ForwardDiff
using CFPlanets: Tank2D
import MutatingOrNot
using MutatingOrNot: void

include("box_helpers.jl")

# Configuration: small box, all physics switched on, closed (free-slip, insulating) walls

const mgr = MultiThread(VectorizedCPU())
const space = Tank2D(Lx = 1.0, Lz = 0.5, g = 10.0)
const Lx, Lz, g = space.Lx, space.Lz, space.g
const T0, q0, ρ0, U = 280.0, 0.035, 1000.0, 0.1

# nonlinear EOS with compressibility, thermobaricity, cabbeling and haline contraction
const fluid_params = (; p0 = 1e5, T0, q0, Cp0 = 4000.0, v0 = 1e-3,
    α_T = 2e-4, α_q = 0.8, α_p = 1e-6, M_s = 3.14038218e-2, R = 8.31446261815324, α_TT = 1e-5, γ = 1e-7)
const fluid = NonlinearBinaryFluid(:potential_temperature, fluid_params)

const thermoflux = ThermoFluxConsistent(k_T = 1e-4, k_q = 1e-5)

# closed (impermeable, insulating, no salt flux) walls
closed_wall(tangential = FreeSlip()) = Wall(; tangential, tracers = (Flux(0.0), Flux(0.0)))   # (consvar, q)

# smooth, x-asymmetric profiles; z = 0 at the bottom
θ_rest(z) = T0 + 1.0 * z / Lz
q_rest(z) = q0 - 0.002 * z / Lz
θ_active(x, z) = θ_rest(z) + 0.3 * sin(2π * x / Lx) * cos(π * z / Lz) + 0.1 * cos(4π * x / Lx + 1) * sin(2π * z / Lz)
q_active(x, z) = q_rest(z) + 0.001 * cos(2π * x / Lx + 0.5) * cos(π * z / Lz)

# FC2D configuration and states

#=
32 × 16 box with every physical term on: vortex force, thermodynamic diffusion and viscosity, using the
energy-conserving schemes
=#
function fc_model(; mgr = mgr, boundary = (Periodic, Bounded), walls = (; bottom = closed_wall(), top = closed_wall()))
    domain = Box2D(; Mx = 32, Mz = 16, boundary)
    boundary_conditions = BoxBoundaryConditions(domain; walls...)
    return FC2D((; mgr, domain, space, fluid, spatial_scheme = Centred2(), vortex_scheme = VortexEnergyCons(),
        thermoflux_scheme = thermoflux, viscosity_scheme = ViscosityScheme(dyn_visc = 1e-3), boundary_conditions))
end

# the model's tendencies!, in the argument order of the generic checks (box_helpers.jl)
fc_tend!(dstate, scratch, model, state) = CFCompressible.tendencies!(dstate, scratch, model, state, 0.0)

#=
active state (m = (ρ, ρθ, ρq), u, w), with no symmetry, so that every term of the tendencies is non-zero;
w vanishes on the walls
=#
function fc_state(model)
    (; domain, dx, dz) = model
    # near-hydrostatic: ρ ≈ ρ0 (1 + g (Lz - z) / c²) with c² = v0 / α_p, plus a small perturbation
    c² = fluid.v0 / fluid.α_p
    ρ(x, z) = ρ0 * (1 + g * (Lz - z) / c² + 1e-4 * sin(2π * x / Lx + 0.7) * cos(π * z / Lz))
    x(i) = cellpoint(domain, dx, 1, i); z(j) = cellpoint(domain, dz, 2, j)
    X(I) = facepoint(domain, dx, 1, I); Z(J) = facepoint(domain, dz, 2, J)
    nx, nz, nX, nZ = ncells(domain, 1), ncells(domain, 2), nfaces(domain, 1), nfaces(domain, 2)
    density = [ρ(x(i), z(j)) for i in 1:nx, j in 1:nz]
    u = [U * (cos(π * z(j) / Lz) + 0.3 * sin(2π * X(I) / Lx) * sin(π * z(j) / Lz)) for I in 1:nX, j in 1:nz]
    w = [0.5U * sin(π * Z(J) / Lz) * cos(2π * x(i) / Lx + 0.2) for i in 1:nx, J in 1:nZ]
    return CFCompressible.initialize(model; density, consvar = θ_active, q = q_active, velocity = (u, w))   # also zeroes the wall faces
end

# per-cell energy ΔxΔz ρ (K + e(v, θ, q) + Φ), with e = h - p v and K = (ℑ(u², (1,), (Cell(),)) + ℑ(w², (2,), (Cell(),))) / 2
function fc_cell_energies(model, (; m, u, w))
    (; domain, dx, dz, fluid) = model
    ic, jc = interior_xz(domain)
    return [begin
        ρ, θ, q = m[i, j, 1], m[i, j, 2] / m[i, j, 1], m[i, j, 3] / m[i, j, 1]
        v = inv(ρ)
        p = fluid(:v, :consvar, :q).pressure(v, θ, q)
        e = fluid(:p, :consvar, :q).specific_enthalpy(p, θ, q) - p * v
        K = (u[i, j]^2 + u[east(domain, i), j]^2 + w[i, j]^2 + w[i, j+1]^2) / 4
        ρ * (K + e + g * cellpoint(domain, dz, 2, j)) * dx * dz
    end for i in ic, j in jc]
end

# walls exercising every kind of condition (consvar = θ for this fluid)
const forced_box = (;
    left   = Wall(tangential = TangentialVelocity(z -> 0.02 * z), tracers = (Value(T0 + 0.5), Value(q0))),
    right  = Wall(tangential = NoSlip(), tracers = (Value(z -> T0 - 0.5 + 0.1 * z), Flux(1e-6))),
    bottom = Wall(tangential = TangentialVelocity(0.05), tracers = (HeatFlux(x -> 40.0 + 10.0 * cos(2π * x / Lx)), Flux(0.0))),
    top    = Wall(tangential = Stress(x -> 2e-4 * sin(2π * x / Lx)), tracers = (Flux(1e-3), Flux(-1e-6))))

# Tests

@testset "FC2D" begin
    model = fc_model(mgr = VectorizedCPU())        # closed walls, periodic in x; serial, for the allocation test
    active = fc_state(model)
    box = fc_model(boundary = (Bounded, Bounded), walls = forced_box)    # threaded: memory safety also catches races

    # energy-conserving configuration: every flux through the walls vanishes, so mass, tracers and energy are conserved
    @testset "closed walls, periodic in x" begin
        test_memory_safety(fc_tend!, model, active)
        test_walls(fc_tend!, model, active)
        test_conservation(fc_tend!, model, active, (; mass = 1, consvar = 2, q = 3))   # m = (ρ, ρθ, ρq)
        r = test_energy(fc_tend!, fc_cell_energies, model, active)
        @info "FC2D relative dE/dt residual" r
        test_no_allocations(fc_tend!, model, active)
    end

    # a short run of the time loop, 3 steps per slice
    @testset "loop" begin
        test_loop(CFCompressible.loop, CFTimeSchemes.RungeKutta4(model), model, active, (; mass = 1, q = 3),
                  (; Nslice = 2, slice_size = 2.5e-3, dt_max = 1.0, cfl = 0.9))
    end

    # every kind of wall condition: the wall fluxes and the work of the wall stresses enter the budgets exactly
    @testset "forced walls, closed box, threaded" begin
        s = fc_state(box)
        test_memory_safety(fc_tend!, box, s)
        test_walls(fc_tend!, box, s)
        test_conservation(fc_tend!, box, s, (; mass = 1, consvar = 2, q = 3))
        r = test_energy_budget(fc_tend!, fc_cell_energies, box, s)           # dE/dt = wall input
        @test abs(r.input) > 1e-8                                            # the forcing is felt
        _, scratch = evaluate(fc_tend!, box, s)
        @test wall_input(box, scratch, 2, 1).heat ≈ 40.0 * Lx rtol = 1e-12     # bottom HeatFlux delivers exactly ∫F dx
    end
end

# the balanced density has the pressure `pref` at the height z0 where the geopotential vanishes, inside or outside the box:
# the pressure of the piecewise-linear interpolant of the cell-center pressures (linear extrapolation outside)
@testset "FC2D balanced density anchored at z0" begin
    for z0 in (0.0, 0.2, Lz + 0.4)
        tank = Tank2D(Lx = Lx, Lz = Lz, g = g, z0 = z0)
        base = fc_model(boundary = (Periodic, Bounded))
        model = typeof(base)(map(f -> f === :space ? tank : getfield(base, f), fieldnames(typeof(base)))...)
        (; domain, dx, dz) = model
        ρ = CFCompressible.balanced_density(model; consvar = θ_active, q = q_active, pref = 1e5)
        jc, i = interior_cells(domain, 2), 5
        z_cell(j) = cellpoint(domain, dz, 2, j)
        p = [fluid(:v, :consvar, :q).pressure(inv(ρ[i, j]), θ_active(cellpoint(domain, dx, 1, i), z_cell(j)),
                                              q_active(cellpoint(domain, dx, 1, i), z_cell(j))) for j in jc]
        k = clamp(floor(Int, (z0 - z_cell(first(jc))) / dz) + 1, 1, length(jc) - 1)
        @test p[k] + (z0 - z_cell(jc[k])) / dz * (p[k+1] - p[k]) ≈ 1e5 rtol = 1e-9
    end
end
