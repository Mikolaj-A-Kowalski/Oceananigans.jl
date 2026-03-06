module FastFloats

export FastFloat

"""
    FastFloat(f::T) where {T<:Base.IEEEFloat}

A wrapper around a floating point type that allows to replace selected mathematical
function (i.e. `sqrt`, division, etc.) with faster, less accurate implementations, that
can be further optimised for specific backend (e.g. CUDA).

Should match the interface of a floating point numbers as much as possible
some functions may be missing.

Arguments
=========
- `f`: A floating point number to wrap in a `FastFloat`. The type of `f` determines the
       floating point type used in the scheme. For example, if `f` is a `Float32`, then
       the resulting `FastFloat` will use `Float32` as its underlying type.
"""
struct FastFloat{T<:Base.IEEEFloat} <: AbstractFloat
    value::T
end

# Disambiguity the constructors
FastFloat{T}(x::Rational) where {T<:Base.IEEEFloat} = FastFloat(T(x))
FastFloat{T}(x::FastFloat) where {T <: Base.IEEEFloat} = FastFloat(T(x.value))

# Promotions and conversions
# See unit test test/test_fast_float.jl for expected behaviour
Base.convert(::Type{FastFloat{T}}, x::FastFloat) where {T} = FastFloat(T(x.value))
Base.convert(::Type{FastFloat{T}}, x::Number) where {T} = FastFloat(T(x))
Base.promote_rule(::Type{FastFloat{T}}, ::Type{S}) where {T, S} = FastFloat{promote_type(T, S)}
Base.promote_rule(::Type{FastFloat{T}}, ::Type{FastFloat{S}}) where {T, S} = FastFloat{promote_type(T, S)}

# Irrationals need to be handled explicitly
Base.promote_rule(::Type{<:Base.AbstractIrrational}, ::Type{FastFloat{T}}) where {T} = FastFloat{T}

# BigFloat interoperability
Base.BigFloat(x::FastFloat) = BigFloat(x.value)

# Unary operators and functions
for (op, effective_op) in (
    :(Base.:+)           => :(Base.:+),
    :(Base.:-)           => :(Base.:-),
    # Misc
    :(Base.real)         => :(Base.real),
    :(Base.imag)         => :(Base.imag),
    :(Base.conj)         => :(Base.conj),
    # Rounding functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Rounding-functions
    :(Base.round)        => :(Base.round),
    :(Base.floor)        => :(Base.floor),
    :(Base.ceil)         => :(Base.ceil),
    :(Base.trunc)        => :(Base.trunc),
    # Sign and absolute value functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Sign-and-absolute-value-functions
    :(Base.abs)          => :(Base.abs),
    :(Base.abs2)         => :(Base.abs2),
    :(Base.sign)         => :(Base.sign),
    :(Base.flipsign)     => :(Base.flipsign),
    # Powers, logs and roots
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Powers,-logs-and-roots
    :(Base.sqrt)         => :(Base.sqrt),
    :(Base.cbrt)         => :(Base.cbrt),
    :(Base.fourthroot)   => :(Base.fourthroot),
    :(Base.exp)          => :(Base.exp),
    :(Base.expm1)        => :(Base.expm1),
    :(Base.log)          => :(Base.log),
    :(Base.log2)         => :(Base.log2),
    :(Base.log10)        => :(Base.log10),
    :(Base.log1p)        => :(Base.log1p),
    :(Base.significand)  => :(Base.significand),
    # Trigonometric and hyperbolic functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Trigonometric-and-hyperbolic-functions
    :(Base.sin)          => :(Base.sin),
    :(Base.cos)          => :(Base.cos),
    :(Base.tan)          => :(Base.tan),
    :(Base.cot)          => :(Base.cot),
    :(Base.sec)          => :(Base.sec),
    :(Base.csc)          => :(Base.csc),
    :(Base.sinh)         => :(Base.sinh),
    :(Base.cosh)         => :(Base.cosh),
    :(Base.tanh)         => :(Base.tanh),
    :(Base.coth)         => :(Base.coth),
    :(Base.sech)         => :(Base.sech),
    :(Base.csch)         => :(Base.csch),
    :(Base.asin)         => :(Base.asin),
    :(Base.acos)         => :(Base.acos),
    :(Base.atan)         => :(Base.atan),
    :(Base.acot)         => :(Base.acot),
    :(Base.asec)         => :(Base.asec),
    :(Base.acsc)         => :(Base.acsc),
    :(Base.asinh)        => :(Base.asinh),
    :(Base.acosh)        => :(Base.acosh),
    :(Base.atanh)        => :(Base.atanh),
    :(Base.acoth)        => :(Base.acoth),
    :(Base.asech)        => :(Base.asech),
    :(Base.acsch)        => :(Base.acsch),
    :(Base.sinc)         => :(Base.sinc),
    :(Base.cosc)         => :(Base.cosc),
    :(Base.sind)         => :(Base.sind),
    :(Base.cosd)         => :(Base.cosd),
    :(Base.tand)         => :(Base.tand),
    :(Base.cotd)         => :(Base.cotd),
    :(Base.secd)         => :(Base.secd),
    :(Base.cscd)         => :(Base.cscd),
    :(Base.asind)        => :(Base.asind),
    :(Base.acosd)        => :(Base.acosd),
    :(Base.atand)        => :(Base.atand),
    :(Base.acotd)        => :(Base.acotd),
    :(Base.asecd)        => :(Base.asecd),
    :(Base.acscd)        => :(Base.acscd),
    )
    @eval begin
        @inline $op(x::FastFloat) = FastFloat($effective_op(x.value))
    end
end

# Not floating point return type
for (op, effective_op) in (
    :(Base.isinf)  => :(Base.isinf),
    :(Base.isfinite) => :(Base.isfinite),
    :(Base.isnan)    => :(Base.isnan),
    :(Base.issubnormal) => :(Base.issubnormal),
)
    @eval begin
        @inline $op(x::FastFloat) = $effective_op(x.value)
    end
end

# Binary operators and functions
for (op, effective_op) in (
    # Basic arithmetic
    :(Base.:+)    => :(Base.:+),
    :(Base.:-)    => :(Base.:-),
    :(Base.:*)    => :(Base.:*),
    :(Base.:/)    => :(Base.:/),
    # Division functions
    :(Base.div)   => :(Base.div),
    :(Base.fld)   => :(Base.fld),
    :(Base.cld)   => :(Base.cld),
    :(Base.rem)   => :(Base.rem),
    :(Base.mod)   => :(Base.mod),
    :(Base.mod1)  => :(Base.mod1),
    # Trig functions
    :(Base.atan)  => :(Base.atan),
    )
    @eval begin
        @inline $op(x::FastFloat, y::FastFloat) = FastFloat($effective_op(x.value, y.value))
    end
end

# Patch for atan2
Base.atan(x::Real, y::FastFloat) = Base.atan(x, y.value)
Base.atan(x::FastFloat, y::Real) = Base.atan(x.value, y)

# Powers
Base.:^(x::FastFloat, y::FastFloat) = FastFloat(Base.:^(x.value, y.value))
Base.:^(x::FastFloat, y::Integer) = FastFloat(Base.:^(x.value, y))

# Not floating point return type
for (op, effective_op) in (
    :(Base.:(==))  => :(Base.:(==)),
    :(Base.:(!=))  => :(Base.:(!=)),
    :(Base.:(<))   => :(Base.:(<)),
    :(Base.:(<=))  => :(Base.:(<=)),
    :(Base.:(>))   => :(Base.:(>)),
    :(Base.:(>=))  => :(Base.:(>=)),
    :(Base.isless) => :(Base.isless),
    :(Base.signbit) => :(Base.signbit),
    )
    @eval begin
        @inline $op(x::FastFloat, y::FastFloat) = $effective_op(x.value, y.value)
    end
end

# Specialisations to fix for the problem to run
# Also why no promotion! What is a set with a TotalOrder (all numbers?)
Base.isless(x::Real, y::FastFloat) = isless(x , y.value)
Base.isless(x::FastFloat, y::Real) = isless(x.value , y)

# Ternary operators and functions
for (op, effective_op) in (
    :(Base.fma)    => :(Base.fma),
    :(Base.muladd) => :(Base.muladd),
    )
    @eval begin
        @inline $op(x::FastFloat, y::FastFloat, z::FastFloat) = FastFloat($effective_op(x.value, y.value, z.value))
    end
end

# Type functions
Base.rtoldefault(::Type{FastFloat{T}}) where {T} = Base.rtoldefault(T)
Base.eps(::Type{FastFloat{T}}) where {T} = Base.eps(T)


end # module
