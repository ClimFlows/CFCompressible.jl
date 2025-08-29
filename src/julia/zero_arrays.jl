module ZeroArrays

struct Zero <: Real end

struct ZeroArray{N} <: AbstractArray{Zero,N}
    ax::NTuple{N, Base.OneTo{Int}}
end
ZeroArray(x::AbstractArray) = ZeroArray(axes(x))
ZeroArray(x::NamedTuple) = map(ZeroArray, x)

@inline Base.size(z::ZeroArray) = map(length, z.ax)
@inline Base.axes(z::ZeroArray) = z.ax
@inline Base.getindex(::ZeroArray, i...) = Zero()

@inline Base.:*(::Number, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Number) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()

@inline Base.:+(x::Number, ::Zero) = x
@inline Base.:+(::Zero, x::Number) = x
@inline Base.:+(x::Complex, ::Zero) = x   # needed to disambiguate ::Number + ::Zero
@inline Base.:+(::Zero, x::Complex) = x   # needed to disambiguate ::Zero + ::Number
@inline Base.:+(::Zero, ::Zero) = Zero()

end
