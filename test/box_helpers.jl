#=
Generic property checks shared by the anelastic and compressible tests (this file is mirrored by hand between
CFAnelastics/test/helpers.jl and CFCompressible/test/box_helpers.jl)
=#

#=
Fresh scratch arrays come from `CFBoxes.alloc!(void, …)`. In the tests they are filled with `ALLOC_FILL[]`, so that
we can check that results never depend on the contents of new allocations. This method is deliberate type piracy,
confined to the tests.
=#
const ALLOC_FILL = Ref(0.0)
CFBoxes.alloc!(::MutatingOrNot.Void, F::Type, mgr, box::CFBoxes.AbstractBox, location::Tuple, dims...) =
    fill!(CFBoxes.alloc(F, mgr, box, location, dims...), ALLOC_FILL[])

# run f() with new allocations filled with `value`
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

# interior index ranges: cells, x-faces (u) and z-faces (w), including walls in bounded directions
interior_xz(d) = (interior_cells(d, 1), interior_cells(d, 2))
interior_Xz(d) = (interior_faces(d, 1), interior_cells(d, 2))
interior_xZ(d) = (interior_cells(d, 1), interior_faces(d, 2))

# x-index of the face east of cell i, wrapping periodically (never reads the halo)
east(d, i) = boundary_topology(d)[1] isa Periodic && i == last(interior_cells(d, 1)) ? first(interior_cells(d, 1)) : i + 1

#=
total and Σ|·| of the per-cell energy tendency dE/dt = d/dε E(state + ε dstate) at ε = 0, by forward-mode AD;
Σ|·| sets the round-off scale against which the total is compared
=#
function energy_tendency(cell_energies, model, state, dstate)
    perturbed(ε) = map((x, dx) -> x .+ ε .* dx, state, dstate)
    dE = ForwardDiff.derivative(ε -> cell_energies(model, perturbed(ε)), 0.0)
    return sum(dE), sum(abs, dE)
end

# Test sets common to both models; `tend!(dstate, scratch, model, state)` wraps each model's tendencies!

# tendencies from freshly allocated, zero-filled scratch: only the memory-safety test sees other contents
evaluate(tend!, model, state) = with_fill(() -> tend!(void, void, model, copy_state(state)), 0.0)

# tendencies never read memory they have not written, and never modify the state interior
function test_memory_safety(tend!, model, state)
    d = model.domain
    ref, _ = evaluate(tend!, model, state)
    ref = map(copy, ref)

    # fresh allocations filled with NaN: any read of unwritten memory would leave a NaN or change the result
    s = copy_state(state)
    out, scratch = with_fill(() -> tend!(void, void, model, s), NaN)
    @test all_finite(out)
    @test all(map(isequal, out, ref))

    # re-using dstate and scratch from the previous call gives the same result (no stale accumulation)
    out2, _ = tend!(out, scratch, model, copy_state(state))
    @test all(map(isequal, out2, ref))

    # the state interior is not modified (bitwise); the halo may be filled by the boundary conditions
    @test s.m[interior_xz(d)..., :] == state.m[interior_xz(d)..., :]
    @test s.u[interior_Xz(d)...] == state.u[interior_Xz(d)...]
    @test s.w[interior_xZ(d)...] == state.w[interior_xZ(d)...]
end

# the normal velocity tendency vanishes on walls (relative to its interior size), so walls stay impermeable
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

#=
Σ dm over the interior equals the net inflow through the walls (zero for the mass), plus Σ σ for the
conservative variable; `tracers = (; mass, consvar, q)` gives their indices in m (mass = nothing if not prognostic)
=#
function test_conservation(tend!, model, state, (; mass, consvar, q); tol = 1e-12)
    d = model.domain
    (ic, jc), (; dx, dz) = interior_xz(d), model
    dstate, scratch = evaluate(tend!, model, state)
    (; consflux_x, consflux_z, qflux_x, qflux_z, σ_cons) = scratch.irreversible
    # Σ over walls of the flux entering the domain, per unit cell volume
    inflow(Jx, Jz) = sum(Jx[first(ic), j] - Jx[last(ic)+1, j] for j in jc) / dx + sum(Jz[i, first(jc)] - Jz[i, last(jc)+1] for i in ic) / dz
    total(k) = sum(dstate.m[ic, jc, k])
    scale(k) = sum(abs, dstate.m[ic, jc, k])
    isnothing(mass) || @test abs(total(mass)) / scale(mass) < tol                                  # mass
    @test abs(total(q) - inflow(qflux_x, qflux_z)) / scale(q) < tol                               # composition
    @test abs(total(consvar) - inflow(consflux_x, consflux_z) - sum(σ_cons[ic, jc])) /            # conservative variable,
          (scale(consvar) + sum(abs, σ_cons[ic, jc])) < tol                                       # with its source σ
end

# closed walls: the total energy is conserved semi-discretely, dE/dt = 0 to round-off
function test_energy(tend!, cell_energies, model, state; tol = 1e-10)
    dstate, _ = evaluate(tend!, model, state)
    total, scale = energy_tendency(cell_energies, model, state, dstate)
    @test abs(total) / scale < tol
    return abs(total) / scale
end

#=
time and allocations of a tendencies! call, behind a function barrier (the arguments come from `evaluate`, whose
result type is not inferred: a dynamic call would box them)
=#
function measure(tend!, dstate, scratch, model, s)
    tend!(dstate, scratch, model, s)      # warm-up, compiles
    return (@elapsed tend!(dstate, scratch, model, s)), (@allocated tend!(dstate, scratch, model, s))
end

# tendencies! allocates nothing once dstate and scratch exist (type-stable, serial manager)
function test_no_allocations(tend!, model, state)
    dstate, scratch = evaluate(tend!, model, state)
    t, bytes = measure(tend!, dstate, scratch, model, copy_state(state))
    @info "$(nameof(typeof(model))) tendencies!" time_ms = 1e3t bytes
    @test bytes == 0
end

#=
energy input through the wall at side `s` (1 lower, 2 upper) of dimension `d`: thermal, Σ (conjvar consflux + chempot qflux)
into the fluid, and mechanical, the work of the wall stress given by the viscous energy flux uτ on the wall faces
=#
function wall_input(model, scratch, d, s)
    (; domain, dx, dz) = model
    (; conjvar, chempot) = scratch.thermo
    (; consflux_x, consflux_z, qflux_x, qflux_z, uτ_x, uτ_z) = scratch.irreversible
    (ic, jc) = interior_xz(domain)
    sign = s == 1 ? 1 : -1                                  # flux along +x/+z enters through the lower wall
    if d == 1
        I, i = s == 1 ? (first(ic), first(ic)) : (last(ic) + 1, last(ic))   # wall face, adjacent cell
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

# forced walls: dE/dt equals the energy input through the walls, relative to Σ|per-cell dE/dt|
function test_energy_budget(tend!, cell_energies, model, state; tol = 1e-12)
    dstate, scratch = evaluate(tend!, model, state)
    dE, scale = energy_tendency(cell_energies, model, state, dstate)
    input = total_wall_input(model, scratch)
    @test abs(dE - input) / scale < tol
    return (; residual = abs(dE - input) / scale, input = input / scale)
end

# records the times at which `loop` calls its outputs, the diagnostics book it hands them, and whether it closes them
mutable struct LoopRecord
    t::Vector{Float64}      # time of each call [s]
    book::Any               # book of the last call
    closed::Bool            # `close` was called
    fail::Bool              # every call after t = 0 throws
end
LoopRecord(; fail = false) = LoopRecord(Float64[], nothing, false, fail)
function (r::LoopRecord)(model, state, dstate, scratch, book, t)
    r.fail && t > 0 && error("output failure")
    push!(r.t, t)
    r.book = book
    return nothing
end
Base.close(r::LoopRecord) = (r.closed = true; nothing)

#=
a short run of `loop`: the outputs are called at t = 0 and at the end of each slice, with the `book` passed to `loop`
if any, then closed, also when an output throws; the state is advanced in place, stays finite, and with closed walls
keeps its total mass and composition (`tracers = (; mass, q)` gives their indices in m, mass = nothing if not
prognostic) to round-off
=#
function test_loop(loop, time_scheme, model, state, (; mass, q), params; tol = 1e-12)
    (ic, jc) = interior_xz(model.domain)
    total(s, k) = sum(s.m[ic, jc, k])
    s, record = copy_state(state), LoopRecord()
    out = redirect_stdout(() -> loop(s, model, time_scheme, params; outputs = record), devnull)
    @test out.u === s.u && s.u != state.u && all_finite(s)                  # advanced in place
    @test record.t ≈ (0:params.Nslice) .* params.slice_size && record.closed
    for k in filter(!isnothing, (mass, q))
        @test abs(total(s, k) - total(state, k)) / abs(total(state, k)) < tol
    end
    failing = LoopRecord(fail = true)
    @test_throws ErrorException redirect_stdout(() -> loop(copy_state(state), model, time_scheme, params; outputs = failing), devnull)
    @test failing.closed
    # a book passed to `loop` reaches the outputs in place of the default one (loop only hands it on)
    book, passed = (; label = "book passed to loop"), LoopRecord()
    redirect_stdout(() -> loop(copy_state(state), model, time_scheme, (; params..., Nslice = 1); outputs = passed, book), devnull)
    @test record.book !== nothing && passed.book === book
end
