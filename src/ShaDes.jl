module ShaDes


# --------------------------------------------------------------------------------------------------
# Julia inbuilt functions to import
# --------------------------------------------------------------------------------------------------
using LinearAlgebra
using Random
using Statistics
using LensFactory


# --------------------------------------------------------------------------------------------------
# Functions to export
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# Internal function
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------
struct init_BestModel
   D_d::Float64
   grid_x::Matrix{Float64}
   grid_y::Matrix{Float64}
   pot::Matrix{Float64}
   def_x::Matrix{Float64}
   def_y::Matrix{Float64}
   def_xx::Matrix{Float64}
   def_yy::Matrix{Float64}
   def_xy::Matrix{Float64}
end

function init_BestModel(D_d::Float64, 
                        grid_x::Matrix{Float64},
                        grid_y::Matrix{Float64},
                        pot::Matrix{Float64}, 
                        def_x::Matrix{Float64}, 
                        def_y::Matrix{Float64},
                        def_xx::Union{Nothing, Matrix{Float64}} = nothing,
                          def_yy::Union{Nothing, Matrix{Float64}} = nothing,
                          def_xy::Union{Nothing, Matrix{Float64}} = nothing)
   # Check if the sizes are consistent
   if (!isnothing(grid_x) && !isnothing(grid_y) && 
       !isnothing(pot) && 
       !isnothing(def_x) && !isnothing(def_y) && 
       !isnothing(def_xx) && !isnothing(def_yy) && !isnothing(def_xy))
      @assert all(size(grid_x) == size(grid_y) == size(pot) == size(def_x) == size(def_y) == size(def_xx) == size(def_yy) == size(def_xy))
   end

   obj = init_BestModel(D_d, grid_x, grid_y, pot, def_x, def_y, def_xx, def_yy, def_xy)
   return obj
end



end
