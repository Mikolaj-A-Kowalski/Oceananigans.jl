
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
struct FastFloat{T<:Base.IEEEFloat} <: Number
    value::T
end

# Promotions and conversions
# See unit test test/test_fast_float.jl for expected behaviour
Base.convert(::Type{FastFloat{T}}, x::FastFloat) where {T} = FastFloat(T(x.value))
Base.convert(::Type{FastFloat{T}}, x::Number) where {T} = FastFloat(T(x))
Base.promote_rule(::Type{FastFloat{T}}, ::Type{S}) where {T, S} = FastFloat{promote_type(T, S)}
Base.promote_rule(::Type{FastFloat{T}}, ::Type{FastFloat{S}}) where {T, S} = FastFloat{promote_type(T, S)}

# Unary operators and functions
for (op, effective_op) in (
    :+ => :+,
    :- => :-,
    # Rounding functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Rounding-functions
    :round => :(Base.round),
    :floor => :(Base.floor),
    :ceil => :(Base.ceil),
    :trunc => :(Base.trunc),
    # Sign and absolute value functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Sign-and-absolute-value-functions
    :abs => :(Base.abs),
    :abs2 => :(Base.abs2),
    :sign => :(Base.sign),
    :flipsign => :(Base.flipsign),
    # Powers, logs and roots
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Powers,-logs-and-roots
    :sqrt => :(Base.sqrt),
    :cbrt => :(Base.cbrt),
    :fourthroot => :(Base.fourthroot),
    :exp => :(Base.exp),
    :expm1 => :(Base.expm1),
    :log => :(Base.log),
    :log2 => :(Base.log2),
    :log10 => :(Base.log10),
    :log1p => :(Base.log1p),
    :significand => :significand,
    # Trigonometric and hyperbolic functions
    # https://docs.julialang.org/en/v1/manual/mathematical-operations/#Trigonometric-and-hyperbolic-functions
    :sin => :(Base.sin),
    :cos => :(Base.cos),
    :tan => :(Base.tan),
    :cot => :(Base.cot),
    :sec => :(Base.sec),
    :csc => :(Base.csc),
    :sinh => :(Base.sinh),
    :cosh => :(Base.cosh),
    :tanh => :(Base.tanh),
    :coth => :(Base.coth),
    :sech => :(Base.sech),
    :csch => :(Base.csch),
    :asin => :(Base.asin),
    :acos => :(Base.acos),
    :atan => :(Base.atan),
    :acot => :(Base.acot),
    :asec => :(Base.asec),
    :acsc => :(Base.acsc),
    :asinh => :(Base.asinh),
    :acosh => :(Base.acosh),
    :atanh => :(Base.atanh),
    :acoth => :(Base.acoth),
    :asech => :(Base.asech),
    :acsch => :(Base.acsch),
    :sinc => :(Base.sinc),
    :cosc => :(Base.cosc),
    :sind => :(Base.sind),
    :cosd => :(Base.cosd),
    :tand => :(Base.tand),
    :cotd => :(Base.cotd),
    :secd => :(Base.secd),
    :cscd => :(Base.cscd),
    :asind => :(Base.asind),
    :acosd => :(Base.acosd),
    :atand => :(Base.atand),
    :acotd => :(Base.acotd),
    :asecd => :(Base.asecd),
    :acscd => :(Base.acscd),
    )
    @eval begin
        @inline Base.$op(x::FastFloat) = FastFloat($op(x.value))
    end
end

# Binary operators and functions
for (op, effective_op) in (
    # Basic arithmetic
    :+ => :+,
    :- => :-,
    :* => :*,
    :/ => :/,
    # Division functions
    :div => :(Base.div),
    :fld => :(Base.fld),
    :cld => :(Base.cld),
    :rem => :(Base.rem),
    :mod => :(Base.mod),
    :mod1 => :(Base.mod1)
    )
    @eval begin
        @inline Base.$op(x::FastFloat, y::FastFloat) = FastFloat($op(x.value, y.value))
    end
end

# Not floating point return type
for (op, effective_op) in (
    :(==) => :(==),
    :!= => :!=,
    :< => :<,
    :<= => :<=,
    :> => :>,
    :>= => :>=,
    :signbit => :(Base.signbit),
    )
    @eval begin
        @inline Base.$op(x::FastFloat, y::FastFloat) = $op(x.value, y.value)
    end
end

# Ternary operators and functions
for (op, effective_op) in (
    :fma => :(Base.fma),
    :muladd => :(Base.muladd),
    )
    @eval begin
        @inline Base.$op(x::FastFloat, y::FastFloat, z::FastFloat) = FastFloat($op(x.value, y.value, z.value))
    end
end

# Powers
@inline Base.:^(x::FastFloat, y::FastFloat) = FastFloat(x.value ^ y.value)
@inline Base.:^(x::FastFloat, y::Number) = FastFloat(x.value ^ y)
