module PlotExt


# --------------------------------------------------------------------------------------------------
# Julia inbuilt functions to import
# --------------------------------------------------------------------------------------------------
using Makie
using Statistics


# --------------------------------------------------------------------------------------------------
# Modules to use
# --------------------------------------------------------------------------------------------------
using ShaDes
using LensFactory


# --------------------------------------------------------------------------------------------------
# This function is an internal function to set plot keywords.  It is a local copy of
# `LensFactory.Lenses.set_plotKws!` on purpose: LensFactory's own PlotExt only loads when Makie,
# KernelDensity *and* LaTeXStrings are all present, whereas this extension triggers on Makie alone.
# --------------------------------------------------------------------------------------------------
function _set_plotKws!(ax)
   ax.xtickalign = 1
   ax.xticksmirrored = true
   ax.ytickalign = 1
   ax.yticksmirrored = true

   ax.xminorticksvisible = true
   ax.xminortickalign = 1
   ax.xminorticksize = 6
   ax.xminorgridwidth = 2
   ax.yminorticksvisible = true
   ax.yminortickalign = 1
   ax.yminorticksize = 6
   ax.yminorgridwidth = 2

   ax.xticksize = 10
   ax.xtickwidth = 2
   ax.yticksize = 10
   ax.ytickwidth = 2

   # A heat map already carries the magnitude; a grid on top of it is noise.
   ax.xgridvisible = false
   ax.ygridvisible = false
end


"""
    ShaDes.plot_shade(shade::ShaDes.init_ShaDes, θx::Matrix{Float64}, θy::Matrix{Float64})

Plot the convergence perturbation of a single ShaDes realisation, with the image positions and the
loci where each component of the perturbing deflection vanishes.

The images are stored on `shade`, so they are not passed separately.  Each image must sit on an
intersection of an `α_x = 0` curve with an `α_y = 0` curve -- that intersection *is* the degeneracy,
and the plot is the visual form of `ShaDes.image_motion`.

# Arguments
- `shade::ShaDes.init_ShaDes` -- The realisation to plot
- `θx::Matrix{Float64}` -- x-coordinates grid (in arcseconds)
- `θy::Matrix{Float64}` -- y-coordinates grid (in arcseconds)

# Keyword arguments
- `kappa_M::Union{Nothing, Matrix{Float64}} = nothing` -- If given, the `κ_M + δκ = 0` locus is drawn
  and the binding pixel marked, showing where positivity sets the amplitude
- `plot_zero_loci::Bool = true` -- Draw the `α_x = 0` and `α_y = 0` curves
   - `contour_stride::Int64 = 4` -- Coarsening factor for the deflection grid.  The deflection costs
     one evaluation per component per pixel, and contour geometry does not need the full resolution
- `show_basis::Bool = false` -- Mark the Plummer component centres
- `show_ellipse::Bool = false` -- Outline the image-enclosing ellipse the basis was laid inside
   - `inflate::Float64 = 1.1` -- Must match the value `grid_centres` used
- `clip_quantile::Float64 = 0.99` -- Symmetric colour range is set at this quantile of `|δκ|` rather
  than at its maximum, so a single edge pixel cannot flatten the whole map
- `figure_size::NTuple{2, Real} = (560, 520)`
- `heatmap_kws::NamedTuple = (colormap = :vik,)` -- `colorrange` is set from `clip_quantile` and
  should be left out; any diverging map with a neutral midpoint is fine, a rainbow map is not
- `limits::Union{Nothing, NTuple{4, Real}} = nothing` -- `(xlo, xhi, ylo, yhi)`; defaults to the grid
- `save_plot::Bool = false`
   - `plot_name::String = "shade.png"`
   - `resolution::Int64 = 2`

# Returns
- `fig`: A Makie figure object containing the plot.
- `ax`: The axis object of the plot for further customization.
"""
function ShaDes.plot_shade(shade::ShaDes.init_ShaDes, θx::Matrix{Float64}, θy::Matrix{Float64};
                           kappa_M::Union{Nothing, Matrix{Float64}} = nothing,
                           plot_zero_loci::Bool                     = true,
                           contour_stride::Int64                    = 4,
                           show_basis::Bool                         = false,
                           show_ellipse::Bool                       = false,
                           inflate::Float64                         = 1.1,
                           clip_quantile::Float64                   = 0.99,
                           figure_size::NTuple{2, Real}             = (560, 520),
                           heatmap_kws::NamedTuple                  = (colormap = :vik,),
                           limits::Union{Nothing, NTuple{4, Real}}  = nothing,
                           save_plot::Bool                          = false,
                           plot_name::String                        = "shade.png",
                           resolution::Int64                        = 2)
   if size(θx) != size(θy)
      throw(ArgumentError("θx and θy must have the same shape; got $(size(θx)) and $(size(θy))."))
   end
   if kappa_M !== nothing && size(kappa_M) != size(θx)
      throw(ArgumentError("kappa_M must match the grid; got $(size(kappa_M)) and $(size(θx))."))
   end
   if !(0.0 < clip_quantile ≤ 1.0)
      throw(ArgumentError("clip_quantile must lie in (0, 1]; got $clip_quantile."))
   end

   # `get_meshgrid` fills θx[i, j] = -θx_half + (i-1)dθ and θy[i, j] = -θy_half + (j-1)dθ, so x runs
   # along dimension 1 and y along dimension 2 -- exactly Makie's heatmap(xs, ys, zs) layout.
   xs, ys = θx[:, 1], θy[1, :]

   # Perturbing convergence
   dk = ShaDes.shade_kappa(shade, θx, θy)

   # Diverging data needs a range symmetric about zero, or the visual midpoint stops meaning δκ = 0.
   lim = quantile(abs.(vec(dk)), clip_quantile)
   lim > 0 || throw(ErrorException("perturbation has zero convergence."))

   # Initialize empty figure
   fig = Figure(size = figure_size, figure_padding = 15, fontsize = 20,
                fonts = (; regular = "Times New Roman"))
   ax  = Axis(fig[1, 1])

   # Plot the convergence perturbation
   hm = heatmap!(ax, xs, ys, dk;
                 colorrange = (-lim, lim), lowclip = :black, highclip = :white, heatmap_kws...)

   # Colorbar specification
   Colorbar(fig[1, 2], hm; label = "δκ", labelpadding = 5, width = 20, tickalign = 1,
            ticksize = 10, tickwidth = 1.5, labelrotation = 3 * pi / 2)
   colgap!(fig.layout, 5)

   legend_marks  = Any[]
   legend_labels = String[]

   # Where positivity binds
   if kappa_M !== nothing
      total = kappa_M .+ dk
      contour!(ax, xs, ys, total; levels = [0.0], color = :firebrick, linewidth = 2.0)
      k = argmin(total)
      scatter!(ax, [θx[k]], [θy[k]]; color = :firebrick, marker = :xcross, markersize = 14)
      push!(legend_marks,  LineElement(color = :firebrick, linewidth = 2.0))
      push!(legend_labels, "κ_M + δκ = 0")
   end

   # Loci where the perturbing deflection vanishes.  Identity is carried by line style as well as
   # colour: two hues would have to compete with the diverging map across its whole range.
   if plot_zero_loci
      c  = max(1, contour_stride)
      cx = collect(@view θx[1:c:end, 1:c:end])
      cy = collect(@view θy[1:c:end, 1:c:end])
      αx, αy = LensFactory.Lenses.get_deflection(ShaDes.shade_lens(shade), cx, cy)

      contour!(ax, cx[:, 1], cy[1, :], αx; levels = [0.0], color = :black,
               linewidth = 1.4, linestyle = :solid)
      contour!(ax, cx[:, 1], cy[1, :], αy; levels = [0.0], color = :black,
               linewidth = 1.4, linestyle = :dash)

      push!(legend_marks,  LineElement(color = :black, linewidth = 1.4, linestyle = :solid))
      push!(legend_labels, "α_x = 0")
      push!(legend_marks,  LineElement(color = :black, linewidth = 1.4, linestyle = :dash))
      push!(legend_labels, "α_y = 0")
   end

   # The image-enclosing ellipse the basis was laid inside
   if show_ellipse
      centre, axes_, semi = ShaDes.enclosing_ellipse(shade.images; inflate = inflate)
      t  = range(0, 2pi; length = 361)
      ex = centre[1] .+ semi[1] .* cos.(t) .* axes_[1, 1] .+ semi[2] .* sin.(t) .* axes_[1, 2]
      ey = centre[2] .+ semi[1] .* cos.(t) .* axes_[2, 1] .+ semi[2] .* sin.(t) .* axes_[2, 2]
      lines!(ax, ex, ey; color = :gray40, linewidth = 1.5, linestyle = :dot)
      push!(legend_marks,  LineElement(color = :gray40, linewidth = 1.5, linestyle = :dot))
      push!(legend_labels, "basis ellipse")
   end

   # Plummer component centres
   if show_basis
      scatter!(ax, shade.basis.x_c, shade.basis.y_c;
               color = (:gray40, 0.55), marker = :cross, markersize = 4)
   end

   # Image positions
   scatter!(ax, shade.images[:, 1], shade.images[:, 2];
            color = :gold, marker = :circle, markersize = 10,
            strokecolor = :black, strokewidth = 0.8)
   push!(legend_marks,  MarkerElement(color = :gold, marker = :circle, markersize = 10,
                                      strokecolor = :black, strokewidth = 0.8))
   push!(legend_labels, "images")

   Legend(fig[2, 1], legend_marks, legend_labels;
          orientation = :horizontal, framevisible = false, tellheight = true, nbanks = 1)

   # Set plot keywords
   _set_plotKws!(ax)

   # Set axis labels and limits.  `gnomonic_offsets_arcsec` is north-up / east-left with x towards
   # west, so x increasing rightwards is already the conventional sky orientation -- do not flip it.
   ax.xlabel = "θ₁ (in arcseconds)"
   ax.ylabel = "θ₂ (in arcseconds)"
   ax.aspect = DataAspect()
   if limits === nothing
      xlims!(ax, minimum(θx), maximum(θx))
      ylims!(ax, minimum(θy), maximum(θy))
   else
      xlims!(ax, limits[1], limits[2])
      ylims!(ax, limits[3], limits[4])
   end

   if save_plot
      save(plot_name, fig, px_per_unit = resolution)
   end
   return fig, ax
end


end
