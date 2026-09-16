module ShaDesV2


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
# ImageData.jl
export init_BestModel
export init_ImageSet
export knot_table
export positions
export positions_of
export rows_of
export n_sources
export n_knots
export source_positions

# Generate.jl
export shift_groups
export init_PlummerBasis
export critical_scale
export init_DegeneracySpace
export degeneracy_dimension
export sample_masses
export init_ShaDes
export shade_lens
export total_lens
export shade_kappa
export rescale
export total_mass
export net_mass

# Diagnostics.jl
export image_residuals
export shift_deflections
export source_shifts
export max_shift
export cap_positivity
export cap_multiplicity
export multiplicity
export magnification
export parity_flips
export check_shade
export is_degenerate
export failed_knots
export print_check
export realisation
export print_report


# --------------------------------------------------------------------------------------------------
# Plotting support, provided by PlotExt when Makie is loaded
# --------------------------------------------------------------------------------------------------
export plot_shade
export plot_caustic

function plot_shade end
function plot_caustic end


# --------------------------------------------------------------------------------------------------
# Implementation, in dependency order
# --------------------------------------------------------------------------------------------------
include("ImageData.jl")
include("Generate.jl")
include("Diagnostics.jl")


end