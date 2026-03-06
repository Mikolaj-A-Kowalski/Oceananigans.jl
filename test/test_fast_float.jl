include("dependencies_for_runtests.jl")

using Oceananigans.FastFloats: FastFloat
import InteractiveUtils: methodswith
import Supposition
import Supposition.Data


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

#################
# Function tests

# Test assertions
strictly_equal(x, y) = (x == y)


# Generator constructor
function make_generator(::Type{FT}, min, max) where {FT<:Base.IEEEFloat}
    return Supposition.Data.Floats{FT}(;
        minimum = min,
        maximum = max,
        infs = false,
        nans = false,
    )
end
function make_generator(::Type{IT}, min, max) where {IT<:Integer}
    return Supposition.Data.Integers(IT(min), IT(max))
end

function make_test_case_name(prefix, operator, arg_types...)
    return Symbol(prefix, "_", operator, "_", join(arg_types, "_"))
end


@testset "FastFloat unary functions" begin
    for (operator, arg1_spec, test_assertion) in
    (
        (:(sin), (Float64, -Inf, Inf), strictly_equal),
        (:(asin), (Float64, -1.0, 1.0), strictly_equal)
    )
        arg_types = collect(map(x -> x[1], (arg1_spec)))
        @eval begin
            Supposition.@check function $(make_test_case_name(operator, arg_types...))(
                fp_number = make_generator($(arg1_spec...)),
            )
                ff = FastFloat(fp_number)
                return $test_assertion($operator(ff), $operator(fp_number))
            end
        end
    end
end


@testset "FastFloat binary functions" begin
    for (operator, arg1_spec, arg2_spec, test_assertion) in
    (
        (:(+), (Float64, -Inf, Inf), (Float64, -Inf, Inf), strictly_equal),
        (:(^), (Float64, 0.0, Inf), (Float64, -Inf, Inf), strictly_equal),
        (:(^), (Float64, 0.0, Inf), (Int64, typemin(Int64), typemax(Int64)), strictly_equal),

    )
        arg_types = collect(map(x -> x[1], (arg1_spec, arg2_spec)))
        @eval begin
            Supposition.@check function $(make_test_case_name(operator, arg_types...))(
                arg1 = make_generator($(arg1_spec...)),
                arg2 = make_generator($(arg2_spec...)),
            )
                # Convert the floating point arguments to FastFloat
                ff_arg1 = arg1 isa AbstractFloat ? FastFloat(arg1) : arg1
                ff_arg2 = arg2 isa AbstractFloat ? FastFloat(arg2) : arg2
                return $test_assertion($operator(ff_arg1, ff_arg2), $operator(arg1, arg2))
            end
        end
    end
end

@testset "FastFloat ternary functions" begin
    for (operator, arg1_spec, arg2_spec, arg3_spec, test_assertion) in
    (
        (:fma, (Float64, -Inf, Inf), (Float64, -Inf, Inf), (Float64, -Inf, Inf), strictly_equal),
        (:muladd, (Float64, -Inf, Inf), (Float64, -Inf, Inf), (Float64, -Inf, Inf), strictly_equal),
    )
        arg_types = collect(map(x -> x[1], (arg1_spec, arg2_spec, arg3_spec)))
        @eval begin
            Supposition.@check function $(make_test_case_name(operator, arg_types...))(
                arg1 = make_generator($(arg1_spec...)),
                arg2 = make_generator($(arg2_spec...)),
                arg3 = make_generator($(arg3_spec...)),
            )
                # Convert the floating point arguments to FastFloat
                ff_arg1 = arg1 isa AbstractFloat ? FastFloat(arg1) : arg1
                ff_arg2 = arg2 isa AbstractFloat ? FastFloat(arg2) : arg2
                ff_arg3 = arg3 isa AbstractFloat ? FastFloat(arg3) : arg3
                return $test_assertion($operator(ff_arg1, ff_arg2, ff_arg3), $operator(arg1, arg2, arg3))
            end
        end
    end
end
