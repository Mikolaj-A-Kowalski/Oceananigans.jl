include("dependencies_for_runtests.jl")


# Generate some random points in a single binade [1;2) interval
function test_data_in_single_binade(::Type{FT}, size) where {FT}
    prng = Random.Xoshiro(44)
    return rand(prng, FT, size) .+ 1.0
end

@testset "CPU newton_div" for (FT, NDC) in Iterators.product((Float32, Float64),
                                                            (Oceananigans.Utils.NoNewtonDiv,
                                                             Oceananigans.Utils.NewtonDivWithConversion{Float32}))
    test_input = test_data_in_single_binade(FT, 1024)

    ref = similar(test_input)
    output = similar(test_input)

    ref .= FT(π) ./ test_input
    output .= Oceananigans.Utils.newton_div.(NDC, FT(π), test_input)

    @test isapprox(ref, output)
end


function append_newton_div_type!(list, weno::WENO{<:Any, <:Any, <:Any, NDC}) where {NDC}
    push!(list, NDC)
    append_newton_div_type!(list, weno.buffer_scheme)
end
append_newton_div_type!(::Any, ::Any) = nothing

# Extract all newton_div types from WENO
# Assumes a non-weno buffer scheme will not have WENO buffer scheme
function get_newton_div_from_weno_advection(weno::WENO)
    newton_div_types = DataType[]
    append_newton_div_type!(newton_div_types, weno)
    return newton_div_types
end

@testset "Verify WENO schemes construction" begin

    # WENO
    weno5 = WENO(order=7; newton_div=Oceananigans.Utils.NoNewtonDiv)
    newton_div_types = get_newton_div_from_weno_advection(weno5)
    @test all(newton_div_types .== Oceananigans.Utils.NoNewtonDiv)

    # Vector Invariant WENO
    vector_weno = WENOVectorInvariant(order=9, newton_div=Oceananigans.Utils.BackendOptimizedNewtonDiv)

    for field_name in fieldnames(typeof(vector_weno))
        field = getfield(vector_weno, field_name)
        if field isa WENO
            newton_div_types = get_newton_div_from_weno_advection(field)
            @test all(newton_div_types .== Oceananigans.Utils.BackendOptimizedNewtonDiv)
        end
    end
end
