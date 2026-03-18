module Models

import Oceananigans

import Oceananigans.Models: initialization_update_state!
import Oceananigans.Advection: materialize_advection
import Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: maybe_extend_halos, FixedSubstepNumber
import Oceananigans: initialize!

using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed
using Oceananigans.Models.HydrostaticFreeSurfaceModels: initialize_free_surface!, HydrostaticFreeSurfaceModel
using Oceananigans.Advection: WENO, replace_weight_computation
using Oceananigans.Utils: BackendOptimizedDivision, ConvertingDivision

using ..TimeSteppers: ReactantModel
using ..Grids: ReactantGrid, ReactantImmersedBoundaryGrid

const ReactantHFSM{TS, E} = Union{
    HydrostaticFreeSurfaceModel{TS, E, <:ReactantState},
    HydrostaticFreeSurfaceModel{TS, E, <:Distributed{<:ReactantState}},
}

initialize_immersed_boundary_grid!(grid) = nothing

using Oceananigans.ImmersedBoundaries:
    GridFittedBottom,
    PartialCellBottom

function initialize_immersed_boundary_grid!(ibg::ReactantImmersedBoundaryGrid)
    # TODO This assumes that the IBG is GridFittedBottom or PartialCellBottom
    needs_initialization = ibg.immersed_boundary isa GridFittedBottom ||
                           ibg.immersed_boundary isa PartialCellBottom

    if needs_initialization
        ib = ibg.immersed_boundary
        bottom_field = ib.bottom_height
        grid = ibg.underlying_grid
        Oceananigans.ImmersedBoundaries.compute_numerical_bottom_height!(bottom_field, grid, ib)
        Oceananigans.BoundaryConditions.fill_halo_regions!(bottom_field)
    end

    return nothing
end

function initialize!(model::ReactantHFSM)
    initialize_immersed_boundary_grid!(model.grid)
    Oceananigans.Models.HydrostaticFreeSurfaceModels.initialize_free_surface!(model.free_surface, model.grid, model.velocities)
    return nothing
end

# Reactant uses CUDA code paths to lift programs to MLIR. The default
# BackendOptimizedDivision on CUDA relies on LLVM NVPTX intrinsics that crash
# Reactant. Override materialize_advection for Reactant grids to substitute
# a compatible weight computation instead.
function materialize_advection(weno::WENO{N, FT, BackendOptimizedDivision}, grid::ReactantGrid) where {N, FT}
    return replace_weight_computation(weno, ConvertingDivision{Float32})
end

end # module
