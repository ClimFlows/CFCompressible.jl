module Loops

import CFTimeSchemes: CFTimeSchemes, advance!, scratch_space
using ..CFCompressible: FC2D
using MutatingOrNot: void, Void

export loop

"""
    loop(model::FC2D, time_scheme, params; write_data = write_data, data_storage = nothing, plot_data = plot_data)
Main time integration loop for the FC2D model.
- `model`: FC2D model instance
- `time_scheme`: time integration scheme
- `params`: parameters dictionary
- `write_func`: function to write data at each time slice
- `write_obj`: object to pass to write_func (e.g., data storage)
- `plot_func`: function to plot data at each time slice
- `plot_obj`: object to pass to plot_func (e.g., plot storage)
"""
function loop(model::FC2D{F}, time_scheme, params; write_func = write_func, write_obj = nothing, plot_func = plot_func, plot_obj = nothing) where {F}
    (; Nslice, slice_size, dt_max, cfl) = params

    # initialise state
    state0 = initialize(model, params)

    # assign initial state
    state = deepcopy(state0)

    # create dstate and scratch space
    dstate, scratch = tendencies!(void, void, model, state, nothing)
    
    # initialise time
    t = zero(F)

    # initialise solver
    solver_scratch = scratch_space(time_scheme, state0, t)
    solver = MutIVPSolver(dt_max, time_scheme, solver_scratch)

    # run loop
    for t_iter = 1:Nslice
        # determine dt from CFL condition
        dt, Nperslice = cfl_condition(model, state, cfl, slice_size, dt_max)
        
        # update solver with new dt
        solver.dt = dt

        # advance state
        state, t = advance!(state, solver, state, t, Nperslice)

        # compute dstate and scratch space at new time
        dstate, scratch = tendencies!(dstate, scratch, model, state, t)

        # write data
        write_obj = write_func(model, state, dstate, scratch, diagnostics(model), write_obj, t_iter, params)

        # plot data
        plot_obj = plot_func(model, state, dstate, scratch, diagnostics(model), plot_obj, t_iter, params)

        # advance progress meter
        progress(t_iter, slice_size, Nslice * slice_size)
    end
end

# empty write and plot functions
function write_func(model, state, dstate, scratch, cookbook, write_obj, t_iter, params) end
function plot_func(model, state, dstate, scratch, cookbook, plot_obj, t_iter, params) end

"""
    cfl_condition(model::FC2D, state, cfl, slice_size, dt_max)

Compute the time step based on the CFL condition.
"""
function cfl_condition(model::FC2D{F}, state, cfl, slice_size, dt_max) where {F}
    (; domain, dx, dz) = model
    max_u       = maximum(abs, state.u[xrange_interior(domain), zrange_interior(domain)]) + 1e-99
    max_w       = maximum(abs, state.w[xrange_interior(domain), zrange_interior(domain)]) + 1e-99
    max_c       = max_sound_speed(state, model)
    dt_cfl      = min(cfl * dx / (max_u + max_c), cfl * dz / (max_w + max_c), slice_size, dt_max)
    Nperslice   = ceil(Int, slice_size / dt_cfl)
    dt_cfl      = slice_size / Nperslice
    return dt_cfl, Nperslice
end

@inline function max_sound_speed((; m), model::FC2D)
    c_func(ρ, cons, q) = fluid(:v, :consvar, :q).soundspeed(inv(ρ), cons, q)
    c_func_m(m) = @views c_func.(m[:, :, 1], m[:, :, 2]./m[:, :, 1], m[:, :, 3]./m[:, :, 1])
    return maximum(c_func_m(m))
end

"""
    progress(t_iter, slice_size, total_time)

Display progress of the simulation in the terminal.
"""
function progress(t_iter, slice_size, total_time)
    seconds = round(t_iter*slice_size; digits=2)
    print("\rProgress: $seconds / $total_time s", " " ^ 10)
    flush(stdout)       
end

"""
    MutIVPSolver{F, Scheme, Scratch}
Mutable struct to hold the time step, time integration scheme, and scratch space.
Similar to CFTimeSchemes.IVPSolver but mutable to allow changing dt.
- `F`: floating point type
- `Scheme`: time integration scheme type
- `Scratch`: type of scratch space
"""
mutable struct MutIVPSolver{F, Scheme, Scratch}
    dt::F               # time step
    scheme::Scheme
    scratch::Scratch    # scratch space, or void
end

"""
    advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::MutIVPSolver, state::State, t, N::Int) where State
Advance the state by N time steps using the specified time integration scheme. Extension of CFTimeSchemes.advance! to handle mutable solver.
- `storage`: optional storage for intermediate states
- `dt`: time step size
- `scheme`: time integration scheme
- `scratch`: scratch space for computations
- `state`: current state of the model
- `t`: current time
- `N`: number of time steps to advance
"""
function advance!(storage::Union{Void, State}, (; dt, scheme, scratch)::MutIVPSolver, state::State, t, N::Int) where State
    @assert N>0
    @assert typeof(t)==typeof(dt)
    state = CFTimeSchemes.advance!(storage, scheme, state, t, dt, scratch)::State
    for i=2:N
        state = CFTimeSchemes.advance!(storage, scheme, state, t+(i-1)*dt, dt, scratch)::State
    end
    return state, t+N*dt
end

end # module Loops