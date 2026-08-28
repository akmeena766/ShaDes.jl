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
- `show_basis::Bool = false` -- Mark the Plummer component centres
- `show_ellipse::Bool = false` -- Outline the image-enclosing ellipse the basis was laid inside
   - `inflate::Float64 = 1.1` -- Must match the value `grid_centres` used
- `clip_quantile::Float64 = 0.99` -- Symmetric colour range is set at this quantile of `|δκ|` rather
  than at its maximum, so a single edge pixel cannot flatten the whole map
- `figure_size::NTuple{2, Real} = (560, 520)`
- `heatmap_kws::NamedTuple = (colormap = :vik,)` -- `colorrange` is set from `clip_quantile` and
  should be left out; any diverging map with a neutral midpoint is fine, a rainbow map is not
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
                           show_basis::Bool                         = false,
                           show_ellipse::Bool                       = false,
                           inflate::Float64                         = 1.01,
                           clip_quantile::Float64                   = 0.99,
                           figure_size::NTuple{2, Real}             = (560, 520),
                           heatmap_kws::NamedTuple                  = (colormap = :vik,),
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
   if lim ≤ 0
      throw(ErrorException("perturbation has zero convergence."))
   end

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
      αx, αy = LensFactory.Lenses.get_deflection(ShaDes.shade_lens(shade), θx, θy)

      contour!(ax, xs, ys, αx; levels = [0.0], color = :black, linewidth = 1.4, linestyle = :solid)
      contour!(ax, xs, ys, αy; levels = [0.0], color = :black, linewidth = 1.4, linestyle = :dash)

      push!(legend_marks,  LineElement(color = :black, linewidth = 1.4, linestyle = :solid))
      push!(legend_labels, "α_x = 0")
      push!(legend_marks,  LineElement(color = :black, linewidth = 1.4, linestyle = :dash))
      push!(legend_labels, "α_y = 0")
   end

   # The image-enclosing ellipse the basis was laid inside
   if show_ellipse
      centre, axes_, semi = ShaDes.enclosing_ellipse(ShaDes.images(shade.sources); inflate = inflate)
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
   obs = ShaDes.images(shade.sources)
   scatter!(ax, obs[:, 1], obs[:, 2]; color = :gold, marker = :circle, markersize = 10,
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
   ax.xlabel = L"θ_x~\text{(in arcseconds)}"
   ax.ylabel = L"θ_y~\text{(in arcseconds)}"

   xlims!(ax, minimum(θx), maximum(θx))
   ylims!(ax, minimum(θy), maximum(θy))

   if save_plot
      save(plot_name, fig, px_per_unit = resolution)
   end
   return fig, ax
end


"""
    ShaDes.plot_caustics(shade::ShaDes.init_ShaDes, model::ShaDes.init_BestModel,
                         src_id::Int64, knot_id::Int64)
 
Plot the caustics of the best-fit model and of the perturbed model on top of each other, in the
source plane of one knot, with the critical curves alongside.
 
This is the picture behind `ShaDes.check_shade`.  The perturbation freezes the images but not the
deflection anywhere else, so it moves the caustics; whether that matters is entirely a question of
whether a caustic has swept across a source.  A realisation that passes the multiplicity check but
has a caustic lying a hair from `β` is one small change away from a different answer, and the count
alone does not show that.
 
Everything is drawn at the `adis` of the requested source, so caustics belonging to other source
planes are not shown.  Every knot sharing that redshift is marked, since they share these caustics.
 
# Arguments
- `shade::ShaDes.init_ShaDes` -- The realisation
- `model::ShaDes.init_BestModel` -- The best-fit model, used for its lens and its grid
- `src_id::Int64` -- Source, as numbered in the image table
- `knot_id::Int64` -- Knot within that source
 
# Keyword arguments
- `show_critical::Bool = true` -- Add a left-hand panel with the critical curves in the image plane
  and the observed images.  With `false` only the source plane is drawn
- `zoom::Float64 = NaN` -- Half-width of the source-plane panel about the knot, in arcseconds.  The
  default fits every caustic, which for a cluster usually leaves the knot invisible at the centre;
  a few arcseconds is generally what you want
- `figure_size::NTuple{2, Real} = (980, 520)` -- Reduced automatically when `show_critical = false`
- `model_kws::NamedTuple = (color = :black, linewidth = 1.6)` -- Best-fit curves
- `shade_kws::NamedTuple = (color = :firebrick, linewidth = 1.6, linestyle = :dash)` -- Perturbed
- `save_plot::Bool = false`
   - `plot_name::String = "caustics.png"`
   - `resolution::Int64 = 2`
 
# Returns
- `fig`: A Makie figure object containing the plot.
- `axes`: A tuple of the axes drawn, `(image_plane, source_plane)` or `(source_plane,)`.
"""
function ShaDes.plot_caustic(shade::ShaDes.init_ShaDes, model::ShaDes.init_BestModel,
                              src_id::Int64, knot_id::Int64;
                              show_critical::Bool          = true,
                              zoom::Float64                = NaN,
                              figure_size::NTuple{2, Real} = (980, 520),
                              model_kws::NamedTuple        = (color = :black, linewidth = 1.6),
                              shade_kws::NamedTuple        = (color = :firebrick, linewidth = 1.6, linestyle = :dash),
                              save_plot::Bool              = false,
                              plot_name::String            = "caustics.png",
                              resolution::Int64            = 2)
   sources = shade.sources
   knots   = ShaDes.knot_table(sources)
 
   # Locate the requested knot
   row = findfirst(j -> Int64(knots[j, 1]) == src_id && Int64(knots[j, 2]) == knot_id,
                   axes(knots, 1))
   if isnothing(row)
      throw(ArgumentError("source $(src_id), knot $(knot_id) is not in the set."))
   end
 
   adis = sources.adis[src_id]
   βx, βy = knots[row, 3], knots[row, 4]
 
   θx, θy = model.grid_x, model.grid_y
   lens_M = model.lens
   lens_P = ShaDes.total_lens(shade, model)
 
   # One deformation tensor per lens serves the critical curves and the caustics both
   ψM = LensFactory.Lenses.get_jacobian(lens_M, θx, θy)
   ψP = LensFactory.Lenses.get_jacobian(lens_P, θx, θy)
 
   caus_M = vcat(LensFactory.Lenses.get_caustic(lens_M, θx, θy, adis, ψM...)...)
   caus_P = vcat(LensFactory.Lenses.get_caustic(lens_P, θx, θy, adis, ψP...)...)
 
   # Initialize empty figure
   size_ = show_critical ? figure_size : (figure_size[1] ÷ 2, figure_size[2])
   fig = Figure(size = size_, figure_padding = 15, fontsize = 20, fonts = (; regular = "Times New Roman"))
 
   function draw!(ax, curves, kws)
      for c in curves
         lines!(ax, first.(c), last.(c); kws...)
      end
   end
 
   axs = []
 
   # Source plane on the left, image plane on the right: a lens is read source to image
   ax1 = Axis(fig[1, 1]; title = "source plane")
   draw!(ax1, caus_M, model_kws)
   draw!(ax1, caus_P, shade_kws)
 
   same = findall(j -> sources.adis[Int64(knots[j, 1])] == adis, axes(knots, 1))
   scatter!(ax1, knots[same, 3], knots[same, 4];
            color = (:gray40, 0.7), marker = :circle, markersize = 7)
   scatter!(ax1, [βx], [βy];
            color = :gold, marker = :star5, markersize = 18,
            strokecolor = :black, strokewidth = 0.8)
 
   _set_plotKws!(ax1)
   ax1.xlabel = L"β_x~\text{(in arcseconds)}"
   ax1.ylabel = L"β_y~\text{(in arcseconds)}"
 
   if isnan(zoom)
      allx = vcat([first.(c) for c in vcat(caus_M, caus_P)]...)
      ally = vcat([ last.(c) for c in vcat(caus_M, caus_P)]...)
      if !isempty(allx)
         xlims!(ax1, extrema(allx)...)
         ylims!(ax1, extrema(ally)...)
      end
   else
      xlims!(ax1, βx - zoom, βx + zoom)
      ylims!(ax1, βy - zoom, βy + zoom)
   end
   push!(axs, ax1)
 
   # Image plane: the critical curves the caustics came from
   if show_critical
      ax2 = Axis(fig[1, 2]; title = "image plane")
      crit_M = vcat(LensFactory.Lenses.get_critical_curve(θx, θy, adis, ψM...)...)
      crit_P = vcat(LensFactory.Lenses.get_critical_curve(θx, θy, adis, ψP...)...)
      draw!(ax2, crit_M, model_kws)
      draw!(ax2, crit_P, shade_kws)
 
      obs = ShaDes.images_of(sources, src_id, knot_id)
      scatter!(ax2, obs[:, 1], obs[:, 2];
               color = :gold, marker = :circle, markersize = 10,
               strokecolor = :black, strokewidth = 0.8)
 
      _set_plotKws!(ax2)
      ax2.xlabel = L"θ_x~\text{(in arcseconds)}"
      ax2.ylabel = L"θ_y~\text{(in arcseconds)}"
      xlims!(ax2, minimum(θx), maximum(θx))
      ylims!(ax2, minimum(θy), maximum(θy))
      push!(axs, ax2)
   end
 
   Legend(fig[2, 1:(show_critical ? 2 : 1)],
          [LineElement(; model_kws...), LineElement(; shade_kws...),
           MarkerElement(color = :gold, marker = :star5, markersize = 18,
                         strokecolor = :black, strokewidth = 0.8)],
          ["best-fit model", "perturbed model", "source $(src_id), knot $(knot_id)"];
          orientation = :horizontal, framevisible = false, tellheight = true, nbanks = 1)
 
   if save_plot
      save(plot_name, fig, px_per_unit = resolution)
   end
   return fig, Tuple(axs)
end


end
