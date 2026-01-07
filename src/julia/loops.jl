module Loops

import CFTimeSchemes: CFTimeSchemes, advance!, scratch_space
using ..CFCompressible: FC2D
using MutatingOrNot: void, Void

export loop

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

# CFL condition for FC2D model
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

# compute maximum sound speed in the domain
@inline function max_sound_speed((; m), model::FC2D)
    c_func(ρ, cons, q) = fluid(:v, :consvar, :q).soundspeed(inv(ρ), cons, q)
    c_func_m(m) = @views c_func.(m[:, :, 1], m[:, :, 2]./m[:, :, 1], m[:, :, 3]./m[:, :, 1])
    return maximum(c_func_m(m))
end

# simple progress meter
function progress(t_iter, slice_size, total_time)
    seconds = round(t_iter*slice_size; digits=2)
    print("\rProgress: $seconds / $total_time s", " " ^ 10)
    flush(stdout)       
end

# mutable solver based on CFTimeSchemes IVPSolver
mutable struct MutIVPSolver{F, Scheme, Scratch}
    dt::F               # time step
    scheme::Scheme
    scratch::Scratch    # scratch space, or void
end

# extension of CFTimeSchemes.advance! to handle mutable solver.
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