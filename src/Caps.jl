# --------------------------------------------------------------------------------------------------
# Multiplicity check
# --------------------------------------------------------------------------------------------------
function _is_closed(curve)
   if length(curve) >= 4 && curve[1] == curve[end]
      return true
   else
      return false
   end
end

"""
    multiplicity(lens::Lenses.AbstractLens, model::init_BestModel, imgs::init_ImageSet;
                 n_far::Int64 = 1)
Number of images `lens` predicts for each knot, in the row order of `knot_table`.

The source position is taken from `lens` itself, not from any stored value, so the caustics and the
source always belong to the same lens.  That is the whole reason there is no source position in
`init_ImageSet`: handing the perturbed lens a source position derived from the unperturbed one is
the easiest mistake to make here, and this way it cannot be made.
"""
function multiplicity(lens::Lenses.AbstractLens, model::init_BestModel, imgs::init_ImageSet;
                      n_far::Int64 = 1)
   θx, θy = model.grid_x, model.grid_y
   ψxx, ψyy, ψxy = Lenses.get_jacobian(lens, θx, θy)

   knots   = knot_table(imgs)
   beta = source_positions(lens, imgs)
   k = size(knots, 1)
   adis = [imgs.adis[Int64(knots[j, 1])] for j in 1:k]
   N = Vector{Int64}(undef, k)
   open_total = 0

   for a in unique(adis)
      caustics_tan, caustics_rad = Lenses.get_caustic(lens, θx, θy, a, ψxx, ψyy, ψxy)

      curves = Vector{Vector{Vector{Float64}}}()
      for curve in vcat(caustics_tan, caustics_rad)
         if _is_closed(curve)
            push!(curves, curve)
         else
            open_total = open_total + 1
         end
      end

      for j in 1:k
         if adis[j] != a
            continue
         end
         N[j] = Lenses.get_image_multiplicity(curves, beta[j, 1], beta[j, 2]; n_far = n_far,
                                              verbose = false)
      end

      if open_total > 0
         @warn "Discarded $(open_total) open critical curve(s) and caustic(s)." maxlog=1
      end
   end
   return N
end


# --------------------------------------------------------------------------------------------------
# Positivity check
# --------------------------------------------------------------------------------------------------
"""
    cap_positivity(shade::init_ShaDes, 
                   model::init_BestModel; 
                   kappa_min::Float64 = 0.0,
                   safety::Float64    = 1.0)
Largest ``\\rm rms\\,\\Delta\\kappa`` at which this perturbation still keeps the total convergence
positive, ``\\kappa_M + \\Delta\\kappa > 0``, everywhere the model has mass.

This is the real bound on a shape degeneracy.  The images give none: the constraint matrix is rank
deficient, so a null-space member scales freely and reproduces every image at any amplitude.
Positivity is what stops it.

# Arguments
- `shade`   : The perturbation, at any amplitude (the cap is scale free).
- `kappa_M` : Convergence of the best-fit model on the same grid.
- `θx`      : x-component of the meshgrid (in ``\\rm \\mathbf{arcseconds}``).
- `θy`      : y-component of the meshgrid (in ``\\rm \\mathbf{arcseconds}``).

# Keyword Arguments
- `kappa_min`: Only pixels with `kappa_M > kappa_min` are tested, so that the empty outskirts --
               where any negative perturbation violates positivity trivially -- do not set it.

# Returns
- `cap`: Cap on ``\\rm rms\\,\\Delta\\kappa``, or `Inf` if the perturbation is nowhere negative.
"""
function cap_positivity(shade::init_ShaDes, model::init_BestModel; kappa_min::Float64 = 0.0, safety::Float64 = 1.0)
   if (safety <= 0.0 || safety > 1.0)
      throw(ArgumentError("safety must lie in (0, 1]; got $safety."))
   end

   dk = shade_kappa(shade, model.grid_x, model.grid_y)
   nx, ny = size(dk)
   rms = sqrt(mean(abs2, dk))
   if rms ≤ 0
      throw(ErrorException("perturbation has zero convergence."))
   end

   # Check if all pixels have κ ≥ 0. Otherwise throw a warning
   n_hole = 0
   @inbounds for j in 1:ny
      @inbounds for i in 1:nx
         if model.kappa[i, j] ≤ 0
            n_hole = n_hole + 1
         end
      end
   end
   if n_hole > 0
      @warn "the model has $(n_hole) pixel(s) with kappa <= 0, which are skipped: positivity " *
            "cannot be asked where there is no mass.  If this is most of the grid, check the " *
            "model."
   end

   # Calculate the amplitude cap
   factor, i_b, j_b = Inf, 0, 0
   @inbounds for j in 1:ny
      for i in 1:nx
         if model.kappa[i, j] > kappa_min && dk[i, j] < 0
            a = model.kappa[i, j] / -dk[i, j]
            if a < factor
               factor, i_b, j_b = a, i, j
            end
         end
      end
   end

   if !isfinite(factor)
      throw(ErrorException("positivity gives no bound. The model's convergence peaks " *
            "at $(round(maximum(model.kappa), digits = 3)); if that is ~0 the model " *
            "or the image coordinates are wrong."))
   end

   binding = (model.grid_x[i_b, j_b], model.grid_y[i_b, j_b], model.kappa[i_b, j_b])
   return rescale(shade, safety * factor), factor * rms, binding
end


function cap_multiplicity(shade::init_ShaDes, model::init_BestModel; n_scan::Int64 = 8,
                          iters::Int64 = 12, n_far::Int64 = 1)
   if n_scan < 1
      throw(ArgumentError("n_scan must be at least 1; got $n_scan."))
   end

   imgs = shade.imgs
   n_model = multiplicity(model.lens, model, imgs; n_far = n_far)

   function ok(f::Float64)
      n = multiplicity(total_lens(rescale(shade, f), model), model, imgs; n_far = n_far)
      return all(n .== n_model)
   end

   if ok(1.0)
      return shade, 1.0
   end

   f_pass, f_fail = 0.0, 1.0
   for i in (n_scan - 1):-1:1
      f = i / n_scan
      if ok(f)
         f_pass = f
         break
      end
      f_fail = f
   end
   if f_pass == 0.0
      @warn "no amplitude on the ladder preserves the multiplicities; this direction in the null " *
            "space is ruled out by the data at any amplitude worth having."
      return rescale(shade, 0.0), 0.0
   end

   for _ in 1:iters
      f = 0.5 * (f_pass + f_fail)
      ok(f) ? (f_pass = f) : (f_fail = f)
   end
   return rescale(shade, f_pass), f_pass
end
