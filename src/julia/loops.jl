module Loops

import CFTimeSchemes: CFTimeSchemes, IVPSolver, advance!, scratch_space, max_time_step
using ..CFCompressible: FC2D, tendencies!, diagnostics
using CFBoxes: interior_cells
using MutatingOrNot: void

export loop

function loop(model::FC2D{F}, state0, time_scheme, params; outputs = nothing) where {F}
    (; Nslice, slice_size, dt_max, cfl) = params

    # copy the initial state
    state = deepcopy(state0)

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
        solver_scratch = scratch_space(time_scheme, state0, t)

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
- diffusion: 4 max(ν, k_T, k_q) (1/dx² + 1/dz²), with ν = (μ + μ_b) / min(ρ);
- internal gravity waves (ω = N |kₓ|/|k| ≤ N): max(N), see `max_buoyancy_frequency`.
"""
function max_frequency(model::FC2D{F}, state) where {F}
    (; m, u, w) = state
    (; domain, dx, dz, viscosity, thermoflux) = model
    irange, jrange = interior_cells(domain, 1), interior_cells(domain, 2)
    inv_d2         = inv(dx^2) + inv(dz^2)
    advection      = maximum(abs, view(u, irange, jrange)) / dx + maximum(abs, view(w, irange, jrange)) / dz
    sound          = 2 * max_sound_speed(state, model) * sqrt(inv_d2)
    ν              = (viscosity.dyn_visc + viscosity.bulk_visc) / minimum(view(m, irange, jrange, 1))
    diffusion      = 4 * max(ν, thermoflux.k_T, thermoflux.k_q) * inv_d2
    return advection + sound + diffusion + max_buoyancy_frequency(model, m) + floatmin(F)
end

# max(N) over the w-faces J between interior cells J-1 and J, N² = g (α ∂consvar/∂z - β ∂q/∂z), with
# α = ∂v/∂consvar / v and β = -∂v/∂q / v at the face (mean pressure, consvar and q of the two cells);
# N² < 0 (static instability) does not oscillate and gives 0
function max_buoyancy_frequency((; fluid, domain, space, dz)::FC2D{F}, m) where {F}
    pressure, volume_functions = fluid(:v, :consvar, :q).pressure, fluid(:p, :consvar, :q).volume_functions
    irange, jrange = interior_cells(domain, 1), interior_cells(domain, 2)
    N2max = zero(F)
    for J in first(jrange)+1:last(jrange), i in irange
        v_1, v_2       = inv(m[i, J-1, 1]), inv(m[i, J, 1])
        cons_1, cons_2 = m[i, J-1, 2] * v_1, m[i, J, 2] * v_2
        q_1, q_2       = m[i, J-1, 3] * v_1, m[i, J, 3] * v_2
        p = (pressure(v_1, cons_1, q_1) + pressure(v_2, cons_2, q_2)) / 2
        v, _, ∂v_∂cons, ∂v_∂q = volume_functions(p, (cons_1 + cons_2) / 2, (q_1 + q_2) / 2)
        N2max = max(N2max, space.g * (∂v_∂cons * (cons_2 - cons_1) + ∂v_∂q * (q_2 - q_1)) / (v * dz))
    end
    return sqrt(N2max)
end

# maximum sound speed over the interior
function max_sound_speed((; m), (; fluid, domain)::FC2D)
    c2 = fluid(:v, :consvar, :q).sound_speed2
    c2max = maximum(interior_cells(domain, 1)) do i
        maximum(interior_cells(domain, 2)) do j
            v = inv(m[i, j, 1])
            c2(v, m[i, j, 2] * v, m[i, j, 3] * v)
        end
    end
    return sqrt(c2max)
end

# simple progress meter
function progress(t_iter, slice_size, total_time)
    seconds = round(t_iter*slice_size; digits=2)
    print("\rProgress: $seconds / $total_time s", " " ^ 10)
    flush(stdout)       
end

end # module Loops
