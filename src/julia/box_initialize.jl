module BoxInitialize

using ..CFCompressible: FC2D
using ..CFCompressible.BoxDynamics: p_cons_comp_T!, state_bc!
using CFBoxes
using MutatingOrNot: @with, @vec

export initialize

## INITIALISE STATE
function initialize(model::FC2D{F}, params) where F
    (; domain) = model
    (; rest_variable, experiment) = params

    # allocate empty state
    state = (m = alloc_xz(F, domain, 3), u = alloc_Xz(F, domain), w = alloc_xZ(F, domain))

    # compute rest profile from chosen variable
    if rest_variable == "θ" || rest_variable == "theta" || rest_variable == :θ || rest_variable == :theta
        rest_θ!(state, model, params)
    else
        error("Unknown rest variable: $rest_variable")
    end

    # add experiment layer
    if experiment == "shear_sin" || experiment == "shear"
        # add velocity shear
        shear!(state, model, params)
        # add sin perturbation
        sin_pert!(state, model, params)
    elseif experiment == "shear_rand"
        # add velocity shear
        shear!(state, model, params)
        # add random kick
        kick!(state, model, params)
    elseif experiment == "jet_rand"
        # add jet profile
        jet!(state, model, params)
        # add sin perturbation
        kick!(state, model, params)
    elseif experiment == "jet_sin"
        # add jet profile
        jet!(state, model, params)
        # add sin perturbation
        sin_pert!(state, model, params)
    elseif experiment == "random"
        kick!(state, model, params)
    else
        error("Unknown experiment: $experiment")
    end
    return state
end

## REST LAYER

function rest_θ!((; m, u, w), model::FC2D, params)
    (; mgr, domain, space, fluid, dz) = model
    (; Lz, g) = space
    (; rest_profile, ptop) = params

    # vertical coordinate vector
    z = zgrid(domain, dz)

    # choose initial temperature profile
    if rest_profile == "linear"
        (; T₀, ΔT, q₀, Δq) = params
        θ = linear_profile(z, T₀, ΔT, Lz)
        q = linear_profile(z, q₀, Δq, Lz)
    elseif rest_profile == "tanh"
        (; T₀, ΔT, δ_T, q₀, Δq, δ_q) = params
        θ = tanh_profile(z, T₀, ΔT, Lz, δ_T)
        q = tanh_profile(z, q₀, Δq, Lz, δ_q)
    elseif rest_profile == "constant" || rest_profile == "const"
        (; T₀, q₀) = params
        θ = const_profile(z, T₀)
        q = const_profile(z, q₀)
    elseif rest_profile == "sin"
        (; T₀, ΔT, q₀, Δq) = params
        θ = sin_profile(z, T₀, ΔT, Lz)
        q = sin_profile(z, q₀, Δq, Lz)
    else
        error("Unknown temperature profile: $rest_profile")
    end

    # use the fact that cons = cons(θ, q)
    cons_func(θ, q) = fluid(:p, :theta, :q).conservative_variable(0., θ, q)

    # vertical profile of conservative variable
    consvar = cons_func.(θ, q)

    # define density function
    ρ_func(p, cons, q) = inv(fluid(:p, :consvar, :q).specific_volume(p, cons, q))

    # initial guess: p⁰(z) = ptop
    p = ptop .* ones(size(consvar))
    
    # compute density profile and update pressure
    ρ = ρ_func.(p, consvar, q)
    p .= hydrostatic_pressure!(p, ρ, ptop, dz, g)

    # refine through iteration
    for _ in 1:5
        # ρⁿ(z) = ρ(pⁿ⁻¹, θ(z), q(z))
        ρ   .= ρ_func.(p, consvar, q)
        # construct p from ρ⁰(z) : p⁰(z) = ptop - ∫ g ρ⁰(z) dz
        p   .= hydrostatic_pressure!(p, ρ, ptop, dz, g)
    end

    # construct m 
    let (irange, jrange) = (axes(m, 1), axes(m, 2))
        @vec for i in irange, j in jrange
            m[i, j, 1] = ρ[j]
            m[i, j, 2] = ρ[j] * consvar[j]
            m[i, j, 3] = ρ[j] * q[j]
        end
    end

    # set velocity to zero
    u[:, :] .= 0
    w[:, :] .= 0
    
    # periodize m
    periodize!(model, (@views m[:, :, 1], @views m[:, :, 2], @views m[:, :, 3]))

    return (; m, u, w)
end

function hydrostatic_pressure!(p, ρ, ptop, dz, g)
    N = length(p)

    # pressure on upper edge
    pu = ptop
    for j in N:-1:1
        # pressure on lower edge
        pl = pu + ρ[j] * g * dz

        # average pressure at cell center
        p[j] = 0.5 * (pl + pu)

        # update pressure on upper edge
        pu = pl
    end
    return p
end

## EXPERIMENT LAYER

# shear flow
function shear!((; m, u, w), model, params)
    (; mgr, domain, space, dz) = model
    (; Lz) = space
    (; δ_u, Δu) = params
    # add on horizontal shear flow
    @with mgr, let (irange, jrange) = (axes(u, 1), axes(u, 2))
        @vec for I in irange, j in jrange        
            z = zpoint(domain, dz, j)
            u[I, j] += Δu * tanh((z .- Lz/2) ./ (δ_u / 2))
        end
    end
    return (; m, u, w)
end

# jet
function jet!((; m, u, w), model, params)
    (; mgr, domain, space, dz) = model
    (; Lz) = space
    (; δ_u, Δu) = params
    # add on horizontal shear flow
    @with mgr, let (irange, jrange) = (axes(u, 1), axes(u, 2))
        @vec for I in irange, j in jrange        
            z = zpoint(domain, dz, j)
            u[I, j] += Δu * exp(-(z .- Lz/2)^2 ./ (δ_u / 2)^2)
        end
    end
    return (; m, u, w)
end

function sin_pert!((; m, u, w), model::FC2D{F}, params) where F
    (; mgr, domain, space, dx, dz, inv_dx, inv_dz) = model
    (; Lx, Lz) = space
    (; kick_size, Δu, δ_u, λ_pert) = params
    
    # top density
    ρtop = 1.5 * m[1, end, 1] - 0.5 * m[1, end-1, 1]

    ψ = alloc_XZ(F, domain)
    let (irange, jrange) = (axes(ψ, 1), axes(ψ, 2))
        @vec for I in irange, J in jrange  
            # shift coordinates to centre (Lx/2, Lz/2)
            x = Xpoint(domain, dx, I) - Lx/2
            z = Zpoint(domain, dz, J) - Lz/2
            ψ[I, J] = -ρtop * exp(- z^2 / (δ_u / 2)^2) * kick_size * Δu * (λ_pert / (2*π)) * cos(2*π*x / λ_pert)
        end
    end
    periodize!(model, ψ)

    # add u, w perturbations from streamfunction
    @with mgr, let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange  
            u[I, j] += @views inv(avg_X(m[:, :, 1], I, j)) * dif_z(ψ, I, j) * inv_dz
        end
    end
    @with mgr, let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange  
            w[i, J] -= @views inv(avg_Z(m[:, :, 1], i, J)) * dif_x(ψ, i, J) * inv_dx
        end
    end
    periodize!(model, (u, w))

    return (; m, u, w)
end

# random perturbation of m = [s, q], u, w
# satisfies ∇·(ρu) = 0
function kick!((; m, u, w), model::FC2D{F}, params) where F
    (; mgr, domain) = model 
    (; kick_size, Δu) = params

    Mx, Mz = intdims(domain)
    Hx, Hz = halo_size(domain)

    # top density (linear extrapolation)
    ρtop = 1.5 * m[1, end, 1] - 0.5 * m[1, end-1, 1]

    # random streamfunction (with two extra vertical ghost cells)
    ψ = alloc_XZ(F, domain)
    let (irange, jrange) = (axes(ψ, 1), axes(ψ, 2))
        @vec for I in irange, J in jrange  
            ψ[I, J] = rand() * ρtop * Δu * kick_size
        end
    end
    # set ψ on boundaries to ensure that w(z=0) = w(z=Lz) = 0
    ψ[:, Hz+1]     .= rand() * ρtop * Δu * kick_size
    ψ[:, Hz+Mz+1]  .= rand() * ρtop * Δu * kick_size

    periodize!(model, ψ)
    
    # add u, w perturbations from streamfunction
    let (irange, jrange) = (Xrange(domain), zrange(domain))
        @vec for I in irange, j in jrange  
            u[I, j] += @views inv(avg_X(m[:, :, 1], I, j)) * dif_z(ψ, I, j)
        end
    end
    let (irange, jrange) = (xrange(domain), Zrange(domain))
        @vec for i in irange, J in jrange  
            w[i, J] -= @views inv(avg_Z(m[:, :, 1], i, J)) * dif_x(ψ, i, J)
        end
    end
    periodize!(model, (u, w))
    
    # apply boundary conditions to state
    p, cons, comp, T = p_cons_comp_T!(void, model, (; m, u, w))
    (; m, u, w) = state_bc!(model, (; m, u, w), (p, cons, comp, T))
    
    # compute max/min of density, conservative variable and composition
    Δrho  = maximum(@views m[:, :, 1]) - minimum(@views m[:, :, 1])
    Δcons = maximum(cons) - minimum(cons)
    Δcomp = maximum(comp) - minimum(comp)

    # add on perturbations to m
    let (irange, jrange) = (axes(m, 1), axes(m, 2))
        @vec for i in irange, j in jrange 
            m[i, j, 1] += rand() * Δrho * kick_size
            m[i, j, 2] += m[i, j, 1] * rand() * Δcons * kick_size
            m[i, j, 3] += m[i, j, 1] * rand() * Δcomp * kick_size
        end
    end
    # apply boundary conditions again to state
    p, cons, comp, T = p_cons_comp_T!(void, model, (; m, u, w))
    (; m, u, w) = state_bc!(model, (; m, u, w), (p, cons, comp, T))
    return (; m, u, w)
end

## VERTICAL PROFILES
const_profile(z, f0)            = f0 .* ones(size(z))
linear_profile(z, f0, Δf, Lz)    = f0 .+ Δf * (z .- Lz/2) / Lz
tanh_profile(z, f0, Δf, Lz, δz)  = f0 .+ 0.5 * Δf * tanh.( (z .- Lz/2) / (0.5 * δz)) 
sin_profile(z, f0, Δf, Lz)       = f0 .+ 0.5 * Δf * sin.( π * (z .- Lz/2) / Lz)

## USEFUL FUNCTIONS
function kin_visc(ρ, dyn_visc)
    return dyn_visc / ρ
end

function Re(ρ, U, ℓ, dyn_visc)
    return ρ * U * ℓ / dyn_visc
end

function Mach(U, c)
    return U / c
end

function Fr(U, g, ℓ)
    return U / sqrt(g * ℓ)
end

function Pr(ρ, k_T, dyn_visc)
    return dyn_visc / ( k_T * ρ )
end

function N0(ΔT, δ_T, g, T₀)
    return ΔT * g / (δ_T * T₀)
end

end # module BoxInitialize