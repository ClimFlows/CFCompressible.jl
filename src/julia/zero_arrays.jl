module ZeroArrays

"""
    z = zero_array(a::AbstractArray)

Return array-like object `z` which behaves like an array with the same axes as `a`,
filled with zeros. `z` does not allocate any memory and is not writable.

`z[i...]` does not check bounds and always returns the special value `Zero()`, 
which behaves like `0` when passed to `+`, `*` and `muladd`. 
"""
zero_array(x::AbstractArray) = ZeroArray(axes(x))
zero_array(x::NamedTuple) = map(zero_array, x)

struct Zero <: Real end

struct ZeroArray{N} <: AbstractArray{Zero,N}
    ax::NTuple{N, Base.OneTo{Int}}
end

@inline Base.size(z::ZeroArray) = map(length, z.ax)
@inline Base.axes(z::ZeroArray) = z.ax
@inline Base.getindex(::ZeroArray, i...) = Zero()

@inline Base.:*(::Number, ::Zero) = Zero()
@inline Base.:*(::Zero, ::Number) = Zero()
@inline Base.:*(::Zero, ::Zero) = Zero()

@inline Base.:+(x, ::Zero) = x
@inline Base.:+(::Zero, x) = x
@inline Base.:+(x::Number, ::Zero) = x
@inline Base.:+(::Zero, x::Number) = x
@inline Base.:+(x::Complex, ::Zero) = x   # needed to disambiguate ::Number + ::Zero
@inline Base.:+(::Zero, x::Complex) = x   # needed to disambiguate ::Zero + ::Number
@inline Base.:+(::Zero, ::Zero) = Zero()

@inline Base.muladd(::Number, ::Zero, c::Number) = c

end
