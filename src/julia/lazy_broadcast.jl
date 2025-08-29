module LazyBroadcast

struct Lazy; end
@inline Base.materialize!(::Lazy, rhs::Broadcast.Broadcasted) = BCArray(rhs)

"""
    z = @.. x*y

    Constructs the lazy array-like `z` such that 
        z[i, ...] == x[i, ...]*z[i, ...]
"""
macro (>)(expr)
    lazy = Lazy()
    return esc(:( @. $lazy = $expr ))
end

struct BCArray{T,N,B} <: AbstractArray{T,N}
    bc::B
end
# @adapt_structure BCArray

function BCArray(rhs::B) where {N, B<:Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{N}}}
    T = Base.Broadcast.combine_eltypes(rhs.f, rhs.args)
    return BCArray{T,N,B}(rhs)
end

@inline Base.axes(x::BCArray) = axes(x.bc)
@inline Base.size(x::BCArray) = size(x.bc)
@inline Base.eltype(::BCArray{T}) where T = T
@inline Base.similar(x::BCArray{T}) where T = similar(x.bc, T, size(x.bc))

Base.@propagate_inbounds Base.getindex(x::BCArray, i...) = getindex(x.bc, i...)

end #module
