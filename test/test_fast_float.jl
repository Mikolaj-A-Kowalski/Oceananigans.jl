include("dependencies_for_runtests.jl")

using Oceananigans.Utils: FastFloat


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

end
