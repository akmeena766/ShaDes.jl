# --------------------------------------------------------------------------------------------------
# Best-model
# --------------------------------------------------------------------------------------------------
struct init_BestModel
   D_d::Float64
   lens::LensFactory.Lenses.AbstractLens
   grid_x::Matrix{Float64}
   grid_y::Matrix{Float64}
   kappa::Matrix{Float64}
end

"""
    init_BestModel(D_d::Float64, 
                   lens::LensFactory.Lenses.AbstractLens,
                   θx::Matrix{Float64},
                   θy::Matrix{Float64})
Store the best-fit lens model, its grid, and its convergence on that grid.

# Arguments
- `D_d` : ADD to the lens (in ``\\rm \\mathbf{meters}``).
- `lens`: The `LensFactory.jl` best-fit lens model.
- `θx`  : x-grid (in ``\\rm \\mathbf{arcseconds}``).
- `θy`  : y-grid (in ``\\rm \\mathbf{arcseconds}``).
"""
function init_BestModel(D_d::Float64, lens::Lenses.AbstractLens, θx::Matrix{Float64}, θy::Matrix{Float64})
   if size(θx) == size(θy)
      κ, _, _ = Lenses.get_kappa_gamma(lens, θx, θy, 1.0)
   else
      throw(ArgumentError("grid_x and grid_y must have the same shape; got $(size(θx)) and $(size(θy))."))
   end
   return init_BestModel(D_d, lens, θx, θy, κ)
end


# --------------------------------------------------------------------------------------------------
# Image constraints
# --------------------------------------------------------------------------------------------------
# How an `init_ImageSet` stores its rows internally, after extraction from a `LensFactory` table and
# conversion to tangent-plane arcseconds.  One row per observed image.  These index `imgs.data`; they
# are not an input format, and nothing outside the package produces a table in this order.
#
# There is deliberately no source position here: a source position is not data, it is whatever a
# given lens implies for these images, so it is computed on demand by `source_positions` from the
# lens you are asking about.  That removes a whole class of bug -- there is no stored source position
# that can go stale when the lens changes.
const COL_SRC  = 1
const COL_KNOT = 2
const COL_OBSX = 3
const COL_OBSY = 4
const COL_SIGX = 5
const COL_SIGY = 6
const COL_SIGT = 7


function _image_table(raw::AbstractMatrix{Float64}, reference::Tuple{Float64, Float64})
   if size(raw, 2) < 8
      throw(ArgumentError("a LensFactory image table needs at least 8 columns (src_id, knot_id, " *
                          "x, y, z_s, sig_x, sig_y, sig_theta); got $(size(raw, 2))."))
   end
 
    # skips z_s, and the flux and time-delay columns
   data = raw[:, [1, 2, 3, 4, 6, 7, 8]]
 
   # (RA, Dec) --> (arcsec, arcsec)
   if reference != (0.0, 0.0)
      x, y = LensFactory.LFUtils.gnomonic_offsets_arcsec(reference[1], reference[2], data[:, COL_OBSX], data[:, COL_OBSY])
      data[:, COL_OBSX] .= x
      data[:, COL_OBSY] .= y
   end
   return data
end


"""
    init_ImageSet(data::Matrix{Float64}, adis::Vector{Float64})
"""
struct init_ImageSet
   data::Matrix{Float64}
   adis::Vector{Float64}
 
   function init_ImageSet(raw::Matrix{Float64}, adis::Vector{Float64}; reference::Tuple{Float64, Float64} = (0.0, 0.0))
      data = _image_table(raw, reference)
      return new(data, copy(adis))
   end
end


function knot_table(imgs::init_ImageSet)
   return unique(imgs.data[:, [COL_SRC, COL_KNOT]], dims = 1)
end


function n_sources(imgs::init_ImageSet)
   return length(unique(@view imgs.data[:, COL_SRC]))
end


function n_knots(imgs::init_ImageSet)
   return size(knot_table(imgs), 1)
end


function positions(imgs::init_ImageSet)
   return imgs.data[:, COL_OBSX:COL_OBSY]
end


function positions_of(imgs::init_ImageSet, src_id::Int64, knot_id::Int64)
   rows = (imgs.data[:, COL_SRC] .== src_id) .& (imgs.data[:, COL_KNOT] .== knot_id)
   return imgs.data[rows, COL_OBSX:COL_OBSY]
end


function source_scatter(lens::Lenses.AbstractLens, imgs::init_ImageSet)
   knots = knot_table(imgs)
   beta, _ = source_positions(lens, imgs)
   out = Vector{Float64}(undef, size(knots, 1))
   @inbounds for j in axes(knots, 1)
      s_id, k_id = Int64(knots[j, 1]), Int64(knots[j, 2])
      rows = rows_of(imgs, s_id, k_id)
      adis = imgs.adis[s_id]
      x = imgs.data[rows, COL_OBSX]
      y = imgs.data[rows, COL_OBSY]
      ax, ay = Lenses.get_deflection(lens, x, y)
      out[j] = maximum(hypot(x[i] - adis * ax[i] - beta[j, 1], y[i] - adis * ay[i] - beta[j, 2]) for i in eachindex(rows))
   end
   return out
end