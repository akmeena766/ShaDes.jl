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
function _from_lens(lens::Lenses.AbstractLens, θx::Matrix{Float64}, θy::Matrix{Float64})
   # Get potential
   ψ = Lenses.get_potential(lens, θx, θy)

   # Get deflection
   ψx, ψy = Lenses.get_deflection(lens, θx, θy)

   # Get convergence
   κ, _, _ = Lenses.get_kappa_gamma(lens, θx, θy, 1.0)

   return ψ, ψx, ψy, κ
end

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
   kappa::Matrix{Float64}
end

function init_BestModel(D_d::Float64, grid_x::Matrix{Float64}, grid_y::Matrix{Float64};
                        lens::Union{Nothing, Lenses.AbstractLens} = nothing,
                        pot::Union{Nothing, Matrix{Float64}}      = nothing, 
                        def_x::Union{Nothing, Matrix{Float64}}    = nothing, 
                        def_y::Union{Nothing, Matrix{Float64}}    = nothing,
                        kappa::Union{Nothing, Matrix{Float64}}    = nothing)
   # Check if the lens is provided
   if !isnothing(lens)
      @assert size(grid_x) == size(grid_y)
      pot, def_x, def_y, kappa = _from_lens(lens, grid_x, grid_y)
   elseif !isnothing(pot) && !isnothing(def_x) && !isnothing(def_y) && !isnothing(kappa)
      @assert all(size(grid_x) == size(grid_y) == size(pot) == size(def_x) == size(def_y) == size(kappa))
   else
      error("Either provide lens or potential and deflection maps.")
   end

   # Create a struct
   obj = init_BestModel(D_d, grid_x, grid_y, pot, def_x, def_y, kappa)
   return obj
end


# --------------------------------------------------------------------------------------------------
# Perturbation basis
# --------------------------------------------------------------------------------------------------
struct init_PlummerBasic
   D_d::Float64
   x_c::Vector{Float64}
   y_c::Vector{Float64}
   x_s::Vector{Float64}
end


function init_PlummerBasis(; D_d::Real = Float64,
                           x_c::Vector{Float64} = Float64[],
                           y_c::Vector{Float64} = Float64[],
                           x_s::Vector{Float64} = Float64[])
   if !(length(x_c) == length(y_c) == length(x_s))
      throw(ArgumentError("x_c, y_c and x_s must have the same length (one entry per component);
            got $(length(x_c)), $(length(y_c)), $(length(x_s))."))
   end
   return init_PlummerBasis(D_d, x_c, y_c, x_s)
end


function init_PlummerBasic()
   
end
end
