include("dependencies_for_runtests.jl")

using Oceananigans.FastFloats: FastFloat
import InteractiveUtils: methodswith

@testset "Conversion and Promotion" begin
    # Null promotion
    @test promote(FastFloat(1.0), FastFloat(2.0)) == (FastFloat(1.0), FastFloat(2.0))

    # Promotion within FastFloat
    # Float32 gets promoted to Float64 but still wrapped
    @test promote(FastFloat(1.0), FastFloat(2.0f0)) == (FastFloat(1.0), FastFloat(2.0))

    # Promotion with other floating point (same precision)
    @test promote(FastFloat(1.0f0), 2.0f0) == (FastFloat(1.0f0), FastFloat(2.0f0))
    @test promote(FastFloat(1.0), 2.0) == (FastFloat(1.0), FastFloat(2.0))

    # Promotion with integers
    @test promote(FastFloat(1.0), 2) == (FastFloat(1.0), FastFloat(2.0))
    @test promote(FastFloat(1.0f0), 2) == (FastFloat(1.0f0), FastFloat(2.0f0))

    # Promotion with irrationals
    @test promote(FastFloat(1.0), π) == (FastFloat(1.0), FastFloat(Float64(π)))
    @test promote(FastFloat(1.0f0), π) == (FastFloat(1.0f0), FastFloat(Float32(π)))

end


function floating_point_test_inputs(::Type{FT}) where {FT<:Base.IEEEFloat}
    return FT[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1e-10,
        -1e-10,
        1e10,
        -1e10,
        # We don't want to enforce equivalent result at the very top of the range
        # only few binades below the max value
        floatmax(FT)/8,
        -floatmax(FT)/8,
    ]
end


# We don't care about the inputs out of 'normal' domain
# Hence we skip the test if a reference is NaN or Inf or it throws a DomainError
# Other wise we check for equality with several ULP tolerance for FP outputs
# exact equality for other
function test_single_case(f, ref_inputs, test_inputs)
    ref_output = try
        f(ref_inputs...)
    catch e
        if isa(e, DomainError)
            return
        else
            rethrow(e)
        end
    end

    test_output = f(test_inputs...)

    if isa(ref_output, Number) && (isnan(ref_output) || isinf(ref_output))
        return
    elseif isa(ref_output, AbstractFloat)
        @test isapprox(test_output, ref_output; atol = 0, rtol = 10*eps(typeof(ref_output)))
    else
        @test test_output == ref_output
    end
end


# Assumes all arguments of the function are FastFloats
function test_single_fastfloat_function(::Type{FT}, m::Method) where {FT}
    (_, argument_types...) = m.sig.parameters
    func = eval(m.name)
    n_args = length(argument_types)

    ref_argument_types = fill(FT, n_args)
    input_values_for_each_argument = fill(floating_point_test_inputs(FT), n_args)

    for inputs in Iterators.product(input_values_for_each_argument...)
        ref_inputs = map((t, v) -> t(v), ref_argument_types, inputs)
        test_inputs = map((t, v) -> t(v), argument_types, inputs)

        test_single_case(func, ref_inputs, test_inputs)
    end
end


function has_only_fast_float_args(m::Method)
    (_, arg_types...) = m.sig.parameters
    return all(t -> t <: FastFloat, arg_types)
end


# Some of the operators need to be skipped and be tested by explicit tests
const function_blacklist = Set((
    :convert,
    :promote,

    # Broken due to other problems...
    :flipsign,
    :signbit,
))


@testset "Test function: $(m.name)" for m in [
    m for m in methodswith(FastFloat, Base; supertypes = false) if
    m.name ∉ function_blacklist && has_only_fast_float_args(m)
]
    test_single_fastfloat_function(Float32, m)
    test_single_fastfloat_function(Float64, m)
end
