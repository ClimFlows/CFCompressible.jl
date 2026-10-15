module Loops

import CFTimeSchemes: CFTimeSchemes, IVPSolver, advance!, scratch_space, max_time_step
using ..CFCompressible: FC2D, tendencies!, diagnostics
using CFBoxes: interior_cells, alloc, Cell, Face
using MutatingOrNot: void
using ManagedLoops: @vec, @with

export loop

function loop(model::FC2D{F}, state, time_scheme, params; outputs = nothing) where {F}
    (; Nslice, slice_size, dt_max, cfl) = params

    # initialise time
    t = zero(F)

    # create dstate and scratch space
    dstate, scratch = tendencies!(void, void, model, state, t)

    # diagnostics, built once: the book owns the output arrays of its computed diagnostics
    book = diagnostics(model)

    # the outputs are closed when the run ends, even on error
    try
        # record the initial state
        record_outputs(outputs, model, state, dstate, scratch, book, t)

        # scratch space of the time scheme
        solver_scratch = scratch_space(time_scheme, state, t)

        # run loop
        for t_iter = 1:Nslice
            # time step from the CFL condition, and a solver with that time step
            dt, Nperslice = cfl_condition(model, time_scheme, state, cfl, slice_size, dt_max)
            solver = IVPSolver(dt, time_scheme, solver_scratch)

            # advance state
            state, t = advance!(state, solver, state, t, Nperslice)

            # record the state at the end of the slice
            record_outputs(outputs, model, state, dstate, scratch, book, t)

            # advance progress meter
            progress(t_iter, slice_size, Nslice * slice_size)
        end
    finally
        close_outputs(outputs)
    end
    return state
end

# call the outputs, if any, with the tendencies and scratch at time t, and close them
function record_outputs(outputs, model, state, dstate, scratch, book, t)
    dstate, scratch = tendencies!(dstate, scratch, model, state, t)
    return outputs(model, state, dstate, scratch, book, t)
end
record_outputs(::Nothing, args...) = nothing
close_outputs(outputs) = close(outputs)
close_outputs(::Nothing) = nothing

## TIME STEPPING

# Runge-Kutta stage update new = old + Σₙ factorₙ incrementₙ (one to four increments) of every state array, halo
# included, run by the model's manager: the default update of CFTimeSchemes broadcasts on a single thread. A `void`
# `new` (non-mutating solver) keeps that default.
CFTimeSchemes.update!(new::NamedTuple, model::FC2D, old, a, ka) =
    update_state!(new, model, old, (a,), (ka,))
CFTimeSchemes.update!(new::NamedTuple, model::FC2D, old, a, ka, b, kb) =
    update_state!(new, model, old, (a, b), (ka, kb))
CFTimeSchemes.update!(new::NamedTuple, model::FC2D, old, a, ka, b, kb, c, kc) =
    update_state!(new, model, old, (a, b, c), (ka, kb, kc))
CFTimeSchemes.update!(new::NamedTuple, model::FC2D, old, a, ka, b, kb, c, kc, d, kd) =
    update_state!(new, model, old, (a, b, c, d), (ka, kb, kc, kd))

# one density of m at a time, so that the threads share out the levels j rather than the densities
function update_state!(new, (; mgr)::FC2D, old, factors, increments)
    for n in axes(new.m, 3)
        update_array!(view(new.m, :, :, n), mgr, view(old.m, :, :, n), factors, map(k -> view(k.m, :, :, n), increments))
    end
    update_array!(new.u, mgr, old.u, factors, map(k -> k.u, increments))
    update_array!(new.w, mgr, old.w, factors, map(k -> k.w, increments))
    return new
end

# x = u + Σₙ factorₙ kₙ at every point of the 2D arrays
function update_array!(x, mgr, u, factors, ks)
    @with mgr, let (irange, jrange) = axes(x)
        @inbounds @vec for i in irange, j in jrange
            x[i, j] = increment(factors, ks, i, j) + u[i, j]
        end
    end
    return x
end

# Σₙ factorₙ kₙ[i, j], accumulated in the order of n by multiply-adds
Base.@propagate_inbounds increment((a,)::Tuple{Any}, (ka,)::Tuple{Any}, i, j) = a * ka[i, j]
Base.@propagate_inbounds increment(factors::Tuple, ks::Tuple, i, j) =
    muladd(last(factors), last(ks)[i, j], increment(Base.front(factors), Base.front(ks), i, j))

"""
    dt, Nperslice = cfl_condition(model::FC2D, scheme, state, cfl, slice_size, dt_max)

Return the time step `dt` [s] and the number `Nperslice` of steps per time slice: the largest `dt` that fits a whole
number of times in `slice_size` [s] and is at most `dt_max` [s] and `cfl` times the stability limit of the explicit
`scheme` (`CFTimeSchemes.max_time_step`) for the fastest discrete frequency `max_frequency(model, state)`. `cfl` is a
fraction of that limit: `cfl ≤ 1` is stable for any explicit scheme.
"""
function cfl_condition(model::FC2D{F}, scheme, state, cfl, slice_size, dt_max) where {F}
    dt        = min(cfl * max_time_step(scheme, inv(max_frequency(model, state))), slice_size, dt_max)
    Nperslice = ceil(Int, slice_size / dt)
    return slice_size / Nperslice, Nperslice
end

"""
    ω = max_frequency(model::FC2D, state)

Return an upper bound `ω` [s⁻¹] on the fastest discrete frequency of the state over the interior, the sum of:
- advection, centred on the C-grid: max|u|/dx + max|w|/dz;
- sound, on the C-grid (ω² = c² (4/dx² sin²(k dx/2) + 4/dz² sin²(m dz/2))): 2 max(c) √(1/dx² + 1/dz²);
- diffusion: 4 max(ν, k_T, k_q) (1/dx² + 1/dz²), with ν = (4μ/3) / min(ρ), the compressive modes being damped
  by (4μ/3) ∇(∇·u);
- internal gravity waves (ω = N |kₓ|/|k| ≤ N): max(N), see `max_buoyancy_frequency`.
"""
function max_frequency(model::FC2D{F}, state) where {F}
    (; m, u, w) = state
    (; domain, dx, dz, viscosity, thermoflux) = model
    irange, jrange = interior_cells(domain, 1), interior_cells(domain, 2)
    inv_d2         = inv(dx^2) + inv(dz^2)
    advection      = maximum(abs, view(u, irange, jrange)) / dx + maximum(abs, view(w, irange, jrange)) / dz
    sound          = 2 * max_sound_speed(state, model) * sqrt(inv_d2)
    ν              = (4 / 3) * viscosity.dyn_visc / minimum(view(m, irange, jrange, 1))
    diffusion      = 4 * max(ν, thermoflux.k_T, thermoflux.k_q) * inv_d2
    return advection + sound + diffusion + max_buoyancy_frequency(model, m) + floatmin(F)
end

# max(N) over the w-faces J between interior cells J-1 and J, N² = g (α ∂consvar/∂z - β ∂q/∂z), with
# α = ∂v/∂consvar / v and β = -∂v/∂q / v at the face (mean pressure, consvar and q of the two cells);
# N² < 0 (static instability) does not oscillate and gives 0. N² is computed on the faces by the model's manager, then
# reduced on one thread; the walls and the halo stay at zero.
function max_buoyancy_frequency(model::FC2D{F}, m) where {F}
    (; mgr, fluid, domain, space, dz) = model
    (; g) = space
    pressure, volume_functions = fluid(:v, :consvar, :q).pressure, fluid(:p, :consvar, :q).volume_functions

    N2 = alloc(F, mgr, domain, (Cell(), Face()))

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2)[2:end])
        @inbounds @vec for i in irange, J in jrange
            v_1, v_2       = inv(m[i, J-1, 1]), inv(m[i, J, 1])
            cons_1, cons_2 = m[i, J-1, 2] * v_1, m[i, J, 2] * v_2
            q_1, q_2       = m[i, J-1, 3] * v_1, m[i, J, 3] * v_2
            p = (pressure(v_1, cons_1, q_1) + pressure(v_2, cons_2, q_2)) / 2
            v, _, ∂v_∂cons, ∂v_∂q = volume_functions(p, (cons_1 + cons_2) / 2, (q_1 + q_2) / 2)
            N2[i, J] = g * (∂v_∂cons * (cons_2 - cons_1) + ∂v_∂q * (q_2 - q_1)) / (v * dz)
        end
    end
    return sqrt(maximum(N2))
end

# maximum sound speed over the interior, from c² computed on the cells by the model's manager (zero in the halo)
function max_sound_speed((; m), model::FC2D{F}) where {F}
    (; mgr, fluid, domain) = model
    sound_speed2 = fluid(:v, :consvar, :q).sound_speed2

    c2 = alloc(F, mgr, domain, (Cell(), Cell()))

    @with mgr, let (irange, jrange) = (interior_cells(domain, 1), interior_cells(domain, 2))
        @inbounds @vec for i in irange, j in jrange
            v        = inv(m[i, j, 1])
            c2[i, j] = sound_speed2(v, m[i, j, 2] * v, m[i, j, 3] * v)
        end
    end
    return sqrt(maximum(c2))
end

# simple progress meter
function progress(t_iter, slice_size, total_time)
    seconds = round(t_iter*slice_size; digits=2)
    print("\rProgress: $seconds / $total_time s", " " ^ 10)
    flush(stdout)       
end

end # module Loops
