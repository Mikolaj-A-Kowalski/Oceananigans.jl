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
    return Supposition.Data.Integers{IT}(min, max)
end


@testset "FastFloat unary functions" begin
    for (operator, input_range, test_assertion) in(
        (:(sin), (-Inf, Inf), strictly_equal),
        (:(asin), (-1.0, 1.0), strictly_equal)
    )
        @eval begin
            Supposition.@check function $(Symbol("check_", operator))(
                fp_number = make_generator(Float64, $input_range...),
            )
                ff = FastFloat(fp_number)
                return $test_assertion($operator(ff), $operator(fp_number))
            end
        end
    end
end
