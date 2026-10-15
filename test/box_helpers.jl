#####
##### Generic property checks shared by the anelastic and compressible tests
#####

# Fresh scratch arrays come from `CFBoxes.alloc!(void, …)` (or `MutatingOrNot.similar!(void, …)`). In the tests they
# are filled with `ALLOC_FILL[]`, so that we can check that results never depend on the contents of new allocations.
const ALLOC_FILL = Ref(0.0)
MutatingOrNot.similar!(::MutatingOrNot.Void, y...) = fill!(similar(y...), ALLOC_FILL[])
CFBoxes.alloc!(::MutatingOrNot.Void, F::Type, box::CFBoxes.AbstractBox, location::Symbol, dims...) =
    fill!(CFBoxes.alloc(F, box, location, dims...), ALLOC_FILL[])

function with_fill(f, value)
    old = ALLOC_FILL[]
    ALLOC_FILL[] = value
    try
        return f()
    finally
        ALLOC_FILL[] = old
    end
end

# apply f to every numeric array in a (nested) tuple / named tuple
foreach_array(f, x::AbstractArray{<:Number}) = f(x)
foreach_array(f, x::Union{Tuple, NamedTuple}) = foreach(y -> foreach_array(f, y), x)
foreach_array(f, x) = nothing

all_finite(x) = (ok = Ref(true); foreach_array(a -> (ok[] &= all(isfinite, a)), x); ok[])

copy_state(s) = map(copy, s)

# ‖a - b‖∞ / ‖b‖∞
relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), floatmin())

# interior index ranges: cells, x-faces (u) and z-faces (w), including walls in bounded directions
interior_xz(d) = (interior_cells(d, 1), interior_cells(d, 2))
interior_Xz(d) = (interior_faces(d, 1), interior_cells(d, 2))
interior_xZ(d) = (interior_cells(d, 1), interior_faces(d, 2))

# x-index of the face east of cell i, wrapping periodically (never reads the halo)
east(d, i) = boundary_topology(d)[1] isa Periodic && i == last(interior_cells(d, 1)) ? first(interior_cells(d, 1)) : i + 1

# periodic x-halo fill for any array whose first dimension is x (cells, u- or w-points)
function fill_xhalo!(a, d)
    Hx, _ = halo_size(d)
    Mx, _ = dims(d)
    for h in 1:Hx
        selectdim(a, 1, h)            .= selectdim(a, 1, h + Mx)
        selectdim(a, 1, Mx + Hx + h)  .= selectdim(a, 1, Hx + h)
    end
    return a
end

# b[i] = a[perm(i)] over interior x-indices, then refill the periodic halo; `sign` flips e.g. u under mirroring
function xpermute(a, d, perm; sign = 1)
    Hx, _ = halo_size(d)
    Mx, _ = dims(d)
    b = copy(a)
    for i in 1:Mx
        selectdim(b, 1, i + Hx) .= sign .* selectdim(a, 1, perm(i) + Hx)
    end
    return fill_xhalo!(b, d)
end

# x-shift by k cells, and x-mirror. Cells and w-points: cell i ↦ Mx+1-i.
# u-points: face I is the west face of cell I, so it ↦ west face of cell Mx+2-I (mod Mx), and u ↦ -u.
function shift_state(s, d, k)
    Mx, _ = dims(d)
    p(i) = mod1(i - k, Mx)
    return (m = xpermute(s.m, d, p), u = xpermute(s.u, d, p), w = xpermute(s.w, d, p))
end
function mirror_state(s, d)
    Mx, _ = dims(d)
    pc(i) = Mx + 1 - i
    pf(I) = mod1(Mx + 2 - I, Mx)
    return (m = xpermute(s.m, d, pc), u = xpermute(s.u, d, pf; sign = -1), w = xpermute(s.w, d, pc))
end

# compare two tendencies on interior points only
function tendency_mismatch(a, b, d)
    (ic, jc), (iu, ju), (iw, jw) = interior_xz(d), interior_Xz(d), interior_xZ(d)
    return max(relerr(a.m[ic, jc, :], b.m[ic, jc, :]),
               relerr(a.u[iu, ju], b.u[iu, ju]),
               relerr(a.w[iw, jw], b.w[iw, jw]))
end

# total and Σ|·| of the per-cell energy tendency dE/dt = d/dε E(state + ε dstate) at ε = 0
function energy_tendency(cell_energies, model, state, dstate)
    perturbed(ε) = map((x, dx) -> x .+ ε .* dx, state, dstate)
    dE = ForwardDiff.derivative(ε -> cell_energies(model, perturbed(ε)), 0.0)
    return sum(dE), sum(abs, dE)
end

#####
##### Test sets common to both models; `tend!(model, dstate, scratch, state)` wraps each model's tendencies!
#####

# tendencies from freshly allocated, zero-filled scratch: only the memory-safety test sees other contents
evaluate(tend!, model, state) = with_fill(() -> tend!(model, void, void, copy_state(state)), 0.0)

# `reuse = false` skips the pass with reused dstate/scratch (same code, already covered by another configuration)
function test_memory_safety(tend!, model, state; reuse = true)
    d = model.domain
    ref, _ = evaluate(tend!, model, state)
    ref = map(copy, ref)

    # fresh allocations filled with NaN: nothing may read memory it has not written
    s = copy_state(state)
    out, scratch = with_fill(() -> tend!(model, void, void, s), NaN)
    @test all_finite(out)
    @test all(map(isequal, out, ref))

    # re-using dstate and scratch gives the same result
    if reuse
        out2, _ = tend!(model, out, scratch, copy_state(state))
        @test all(map(isequal, out2, ref))
    end

    # the state interior is not modified (bitwise)
    @test s.m[interior_xz(d)..., :] == state.m[interior_xz(d)..., :]
    @test s.u[interior_Xz(d)...] == state.u[interior_Xz(d)...]
    @test s.w[interior_xZ(d)...] == state.w[interior_xZ(d)...]
end

function test_symmetries(tend!, model, state; tol = 1e-8)   # round-off from cancellation in B; index bugs give O(1e-3–1)
    d = model.domain
    tendency(s) = first(evaluate(tend!, model, s))
    ds = tendency(state)
    @test tendency_mismatch(tendency(shift_state(state, d, 5)), shift_state(ds, d, 5), d) < tol
    @test tendency_mismatch(tendency(mirror_state(state, d)), mirror_state(ds, d), d) < tol
end

# the normal velocity stays zero on walls
function test_walls(tend!, model, state; tol = 1e-12)
    d = model.domain
    Bx, Bz = boundary_topology(d)
    dstate, _ = evaluate(tend!, model, state)
    if Bz isa Bounded
        iw, jw = interior_xZ(d)
        scale = max(maximum(abs, dstate.w[iw, jw]), floatmin())
        @test maximum(abs, dstate.w[iw, first(jw)]) / scale < tol   # bottom wall
        @test maximum(abs, dstate.w[iw, last(jw)]) / scale < tol    # top wall
    end
    if Bx isa Bounded
        iu, ju = interior_Xz(d)
        scale = max(maximum(abs, dstate.u[iu, ju]), floatmin())
        @test maximum(abs, dstate.u[first(iu), ju]) / scale < tol   # left wall
        @test maximum(abs, dstate.u[last(iu), ju]) / scale < tol    # right wall
    end
end

# Σ dm over the interior equals the net inflow through the walls (zero for the mass), plus Σ σ for the
# conservative variable; `tracers = (; mass, consvar, q)` gives their indices in m (mass = nothing if not prognostic)
function test_conservation(tend!, model, state, (; mass, consvar, q); tol = 1e-12)
    d = model.domain
    (ic, jc), (; dx, dz) = interior_xz(d), model
    dstate, scratch = evaluate(tend!, model, state)
    (; consflux_x, consflux_z, qflux_x, qflux_z, σ_cons) = scratch.irreversible
    inflow(Jx, Jz) = sum(Jx[first(ic), j] - Jx[last(ic)+1, j] for j in jc) / dx + sum(Jz[i, first(jc)] - Jz[i, last(jc)+1] for i in ic) / dz
    total(k) = sum(dstate.m[ic, jc, k])
    scale(k) = sum(abs, dstate.m[ic, jc, k])
    isnothing(mass) || @test abs(total(mass)) / scale(mass) < tol
    @test abs(total(q) - inflow(qflux_x, qflux_z)) / scale(q) < tol
    @test abs(total(consvar) - inflow(consflux_x, consflux_z) - sum(σ_cons[ic, jc])) / (scale(consvar) + sum(abs, σ_cons[ic, jc])) < tol
end

function test_energy(tend!, cell_energies, model, state; tol = 1e-10)
    dstate, _ = evaluate(tend!, model, state)
    total, scale = energy_tendency(cell_energies, model, state, dstate)
    @test abs(total) / scale < tol
    return abs(total) / scale
end

# time and allocations of a tendencies! call, behind a function barrier (the arguments come from `evaluate`, whose
# result type is not inferred: a dynamic call would box them)
function measure(tend!, model, dstate, scratch, s)
    tend!(model, dstate, scratch, s)
    return (@elapsed tend!(model, dstate, scratch, s)), (@allocated tend!(model, dstate, scratch, s))
end

function test_no_allocations(tend!, model, state)
    dstate, scratch = evaluate(tend!, model, state)
    t, bytes = measure(tend!, model, dstate, scratch, copy_state(state))
    @info "$(nameof(typeof(model))) tendencies!" time_ms = 1e3t bytes
    @test bytes == 0
end

# RK4 steps; returns (states before/after) for model-specific checks
function integrate(model, state, dt, nsteps)
    solver = CFTimeSchemes.IVPSolver(CFTimeSchemes.RungeKutta4(model), dt, copy_state(state), 0.0)
    s = copy_state(state)
    for n in 1:nsteps
        s, _ = CFTimeSchemes.advance!(s, solver, s, (n - 1) * dt, 1)
    end
    return s
end

function test_integration(model, state, dt, nsteps, cell_energies, conserved, kinetic; tol_E = 1e-6)
    d = model.domain
    s = integrate(model, state, dt, nsteps)
    @test all_finite(s)
    for k in conserved
        @test abs(sum(s.m[interior_xz(d)..., k]) / sum(state.m[interior_xz(d)..., k]) - 1) < 1e-13
    end
    ΔE = sum(cell_energies(model, s)) - sum(cell_energies(model, state))
    @test abs(ΔE) / kinetic(model, state) < tol_E
    return s
end

# every diagnostic (and every internal one) is finite on the full array, halo included, and has a long name, units
# and grid location; s and θ follow the conservative variable for both choices of consvar
function test_diagnostics((; book, internal, info), tend!, model, state, fluid_params)
    dstate, scratch = evaluate(tend!, model, state)
    for b in (book, internal)
        session = open(b; state = copy_state(state), dstate, scratch, model)
        for sym in keys(getfield(b, :recipes))
            @testset "$sym" begin
                @test all(isfinite, CookBooks.get(session, sym))
            end
        end
        close(session)
    end
    @test issubset(keys(getfield(book, :recipes)), keys(info))
    # every diagnostic lies on its declared grid location
    session = open(book; state = copy_state(state), dstate, scratch, model)
    for sym in keys(getfield(book, :recipes))
        @test size(CookBooks.get(session, sym)) == size(alloc(Float64, model.domain, info[sym].location))
    end
    close(session)
    # consvar = θ for the test fluid; for a fluid with consvar = s, the same recipes give s = consvar and the same θ
    session = open(book; state = copy_state(state), dstate, scratch, model)
    p, θ, q = CookBooks.get(session, :pressure), CookBooks.get(session, :conservative_variable), CookBooks.get(session, :composition)
    @test CookBooks.get(session, :potential_temperature) ≈ θ rtol = 1e-12
    fluid_s = NonlinearBinaryFluid(:entropy, fluid_params)
    s = fluid_s(:p, :theta, :q).conservative_variable.(p, θ, q)
    recipes = getfield(book, :recipes)
    @test recipes[:specific_entropy]((; fluid = fluid_s), p, s, q) ≈ s rtol = 1e-12
    @test recipes[:potential_temperature]((; fluid = fluid_s), p, s, q) ≈ θ rtol = 1e-12
    close(session)
    # the computed diagnostics equal their direct formulas, and no recipe allocates (the book owns its outputs)
    session = open(book; state, dstate, scratch, model)
    (; ∂u_z, ∂w_x, ∂u_x, ∂w_z) = scratch.derivatives
    (; consflux_x, consflux_z) = scratch.irreversible
    ic, jc = interior_xz(model.domain)
    @test CookBooks.get(session, :vorticity) == ∂u_z .- ∂w_x
    @test CookBooks.get(session, :divergence) == ∂u_x .+ ∂w_z
    @test CookBooks.get(session, :div_consflux)[ic, jc] ≈
          [(consflux_x[i+1, j] - consflux_x[i, j]) / model.dx + (consflux_z[i, j+1] - consflux_z[i, j]) / model.dz for i in ic, j in jc]
    for (sym, recipe) in recipes
        args = map(name -> CookBooks.get(session, name), CookBooks.fun_arg_names(recipe))
        @testset "$sym allocations" begin
            @test allocations(recipe, args) == 0
        end
    end
    close(session)
end
allocations(f::F, args::A) where {F, A} = (f(args...); @allocated f(args...))

# energy input through the wall at side `s` (1 lower, 2 upper) of dimension `d`: thermal, Σ (conjvar consflux + chempot qflux)
# into the fluid, and mechanical, the work of the wall stress given by the viscous energy flux uτ on the wall faces
function wall_input(model, scratch, d, s)
    (; domain, dx, dz) = model
    (; conjvar, chempot) = scratch.thermo
    (; consflux_x, consflux_z, qflux_x, qflux_z, uτ_x, uτ_z) = scratch.irreversible
    (ic, jc) = interior_xz(domain)
    sign = s == 1 ? 1 : -1                                  # flux along +x/+z enters through the lower wall
    if d == 1
        I, i = s == 1 ? (first(ic), first(ic)) : (last(ic) + 1, last(ic))
        heat = sum(conjvar[i, j] * consflux_x[I, j] + chempot[i, j] * qflux_x[I, j] for j in jc) * dz
        work = -sum(uτ_x[I, j] for j in jc) * dz
    else
        J, j = s == 1 ? (first(jc), first(jc)) : (last(jc) + 1, last(jc))
        heat = sum(conjvar[i, j] * consflux_z[i, J] + chempot[i, j] * qflux_z[i, J] for i in ic) * dx
        work = -sum(uτ_z[i, J] for i in ic) * dx
    end
    return (; heat = sign * heat, work = sign * work)
end

# total energy input through all walls
function total_wall_input(model, scratch)
    walls = [(d, s) for d in 1:2 if boundary_topology(model.domain)[d] isa Bounded for s in 1:2]
    return sum(((d, s),) -> sum(wall_input(model, scratch, d, s)), walls; init = 0.0)
end

# dE/dt equals the energy input through the walls (zero for closed walls), relative to Σ|per-cell dE/dt|
function test_energy_budget(tend!, cell_energies, model, state; tol = 1e-12)
    dstate, scratch = evaluate(tend!, model, state)
    dE, scale = energy_tendency(cell_energies, model, state, dstate)
    input = total_wall_input(model, scratch)
    @test abs(dE - input) / scale < tol
    return (; residual = abs(dE - input) / scale, input = input / scale)
end
