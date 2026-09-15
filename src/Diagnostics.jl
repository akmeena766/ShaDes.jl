# --------------------------------------------------------------------------------------------------
# Diagnostics.jl -- is it really a degeneracy, and how far can it go
# --------------------------------------------------------------------------------------------------
# Everything asked of a realisation after it exists.
#
#   * `image_residuals`, `shift_deflections` -- the exact, estimator-free test: the perturbation's
#                               deflection is constant across every group, so no image moves.
#   * `source_shifts`, `max_shift` -- the unobservable source-plane move the constant implies.
#   * `cap_positivity`, `cap_multiplicity` -- what actually bounds the amplitude, since the images do
#                               not.
#   * `multiplicity`, `check_shade`, `parity_flips` -- image counts and parities, model vs perturbed.
#   * `realisation`, `print_report` -- one realisation end to end.
# --------------------------------------------------------------------------------------------------


"""
    image_residuals(shade::init_ShaDes)
Calculate the deflection that each perturbation adds at each constrained image, 
``N \\times 2`` (in ``\\rm \\mathbf{arcseconds}``).

This is the net deflection at each image position after all perturbations are added. 
At `tol = 0` this should be zero.  At `tol > 0` it is the budget that was asked for, and is 
the honest quantity to quote: a model fits images to a finite rms and anything inside it is 
an equally good fit.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.

# Returns
- `Matrix{Float64}`: An `N x 2` matrix where each row is the deflection `(ax, ay)` at the 
   corresponding image position.
"""
function image_residuals(shade::init_ShaDes)
   alpha, c = shift_deflections(shade)
   return alpha .- c
end


# --------------------------------------------------------------------------------------------------
# Source-plane shift bookkeeping
# --------------------------------------------------------------------------------------------------
"""
    shift_deflections(shade::init_ShaDes)
The deflection the perturbation adds at every image, and the constant it is allowed to be.

# Returns
- `alpha`: ``N \\times 2`` deflection at each image (in ``\\rm \\mathbf{arcseconds}``).
- `c`    : ``N \\times 2``, row `i` holding the constant for the group image `i` belongs to.  All
           zero when `shade.shift == :none`, so `alpha - c` is the honest residual in every mode.
"""
function shift_deflections(shade::init_ShaDes)
   lens = shade_lens(shade)
   obs  = positions(shade.imgs)
   n    = size(obs, 1)

   alpha = Matrix{Float64}(undef, n, 2)
   @inbounds for i in 1:n
      ax, ay = Lenses.get_deflection(lens, obs[i, 1], obs[i, 2])
      alpha[i, 1] = ax
      alpha[i, 2] = ay
   end

   c = zeros(Float64, n, 2)
   @inbounds for g in shift_groups(shade.imgs, shade.shift)
      mx = mean(@view alpha[g, 1])
      my = mean(@view alpha[g, 2])
      for i in g
         c[i, 1] = mx
         c[i, 2] = my
      end
   end
   return alpha, c
end


"""
    source_shifts(shade::init_ShaDes)
How far each knot's source moves, ``N_{\\rm knot} \\times 2`` (in ``\\rm \\mathbf{arcseconds}``),
in the row order of `knot_table`.  This is `-adis * c`, the shift the source takes if the images
map to it perfectly, and it is unobservable: it is the price the source plane pays so that every
image stays exactly where it was observed.

This is a diagnostic only.  Nothing downstream consumes it, because nothing stores a source
position -- `source_positions` re-derives the source from whichever lens it is handed, so the
perturbed lens automatically gets the perturbed source.

The two agree only in the limit of a perfect fit.  The weights in `source_positions` depend on `A_i`
at the images, which the perturbation changes, so a knot with any scatter has its weighted centroid
pulled slightly off the analytic shift:

```julia
beta_M = source_positions(model.lens, shade.imgs)
beta_P = source_positions(total_lens(shade, model), shade.imgs)
(beta_P - beta_M) - source_shifts(shade)      # zero only if the model fits perfectly
```

`multiplicity` uses the re-derived positions, not this, so it sees the perturbed source a modeller
would actually recover.
"""
function source_shifts(shade::init_ShaDes)
   imgs = shade.imgs
   knots   = knot_table(imgs)
   k       = size(knots, 1)
   delta   = zeros(Float64, k, 2)
   if shade.shift === :none
      return delta
   end

   _, c = shift_deflections(shade)
   @inbounds for j in 1:k
      s_id, k_id = Int64(knots[j, 1]), Int64(knots[j, 2])

      # Every image of a knot carries the same constant, so the first one is enough.
      row = 0
      for i in 1:size(imgs.data, 1)
         if imgs.data[i, COL_SRC] == s_id && imgs.data[i, COL_KNOT] == k_id
            row = i
            break
         end
      end

      adis = imgs.adis[s_id]
      delta[j, 1] = -adis * c[row, 1]
      delta[j, 2] = -adis * c[row, 2]
   end
   return delta
end


"""
    max_shift(shade::init_ShaDes)
Largest source-plane move any knot suffers (in ``\\rm \\mathbf{arcseconds}``).  Worth reading
before believing a realisation: a shift far larger than the source's own size, or than any prior on
the source position, is a degeneracy the data cannot see but a modeller would still reject.
"""
function max_shift(shade::init_ShaDes)
   d = source_shifts(shade)
   if size(d, 1) == 0
      return 0.0
   end
   return maximum(hypot(d[j, 1], d[j, 2]) for j in axes(d, 1))
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
# Checking one realisation
# --------------------------------------------------------------------------------------------------
struct init_ShaDesCheck
   knots::Matrix{Float64}        # (src_id, knot_id), row order of `knot_table`
   n_model::Vector{Int64}
   n_shade::Vector{Int64}   
end


function is_degenerate(check::init_ShaDesCheck)
   return all(check.n_shade .== check.n_model)
end


function failed_knots(check::init_ShaDesCheck)
   bad = Tuple{Int64, Int64, Int64, Int64}[]
   for j in eachindex(check.n_model)
      if check.n_shade[j] != check.n_model[j]
         push!(bad, (Int64(check.knots[j, 1]), Int64(check.knots[j, 2]), check.n_model[j], check.n_shade[j]))
      end
   end
   return bad
end


function print_check(c::init_ShaDesCheck; io::IO = stdout)
   println(io, "init_ShaDesCheck: ", is_degenerate(c) ? "degenerate" : "not a degeneracy")
   println(io, "    src  knot    N_M -> N_MP")
   for j in eachindex(c.n_model)
      flag = c.n_shade[j] != c.n_model[j] ? "  <-" : ""
      println(io, lpad(Int64(c.knots[j, 1]), 7), lpad(Int64(c.knots[j, 2]), 6),
                  lpad(c.n_model[j], 7), " -> ", rpad(c.n_shade[j], 8), flag)
   end
   return nothing
end


function magnification(lens::Lenses.AbstractLens, imgs::init_ImageSet)
   n = size(imgs.data, 1)
   mu = Vector{Float64}(undef, n)
   @inbounds for i in 1:n
      adis = imgs.adis[Int64(imgs.data[i, COL_SRC])]
      κ, γ1, γ2 = Lenses.get_kappa_gamma(lens, imgs.data[i, COL_OBSX], imgs.data[i, COL_OBSY], adis)
      mu[i] = 1.0 / ((1.0 - κ)^2 - γ1^2 - γ2^2)
   end
   return mu
end


function parity_flips(shade::init_ShaDes, model::init_BestModel)
   mu_model = magnification(model.lens, shade.imgs)
   mu_shade = magnification(total_lens(shade, model), shade.imgs)
   flipped = findall(i -> sign(mu_model[i]) != sign(mu_shade[i]), eachindex(mu_model))
   return flipped, mu_model, mu_shade
end


function check_shade(shade::init_ShaDes, model::init_BestModel; n_far::Int64 = 1)
   imgs = shade.imgs
   n_model = multiplicity(model.lens, model, imgs; n_far = n_far)
   n_shade = multiplicity(total_lens(shade, model), model, imgs; n_far = n_far)

   knots = knot_table(imgs)
   for j in eachindex(n_model)
      s_id, k_id = Int64(knots[j, 1]), Int64(knots[j, 2])
      n_obs = size(positions_of(imgs, s_id, k_id), 1)
      if n_model[j] < n_obs
         @warn "source $(s_id), knot $(k_id): the best-fit model predicts $(n_model[j]) " *
               "image(s) but $(n_obs) are given.  Enlarge the grid, or check `source_scatter`, " *
               "before reading anything into the comparison." maxlog=1
      end
   end
   return init_ShaDesCheck(knot_table(imgs), n_model, n_shade)
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

# --------------------------------------------------------------------------------------------------
# One realisation, end to end
# --------------------------------------------------------------------------------------------------
struct init_ShaDesReport
   scale::Float64
   core::Float64
   n_comp::Int64
   n_free::Int64
   positivity::Float64
   factor::Float64
   rms_kappa::Float64
   check::init_ShaDesCheck
   binding::NTuple{3, Float64}
   image_residual::Float64
   shift::Symbol
   max_shift::Float64
   flipped::Vector{Int64}
   mu_ratio::Float64
   total_mass::Float64
   net_mass::Float64
end


function print_report(r::init_ShaDesReport; io::IO = stdout)
   println(io, "init_ShaDesReport")
   println(io, "   basis           : scale ", round(r.scale, sigdigits = 3), ", core ",
                                     round(r.core, sigdigits = 3), " arcsec, ", r.n_comp,
                                     " components, ", r.n_free, " free")
   println(io, "   positivity cap  : ", round(r.positivity, sigdigits = 4), " rms kappa")
   println(io, "   multiplicity    : ", round(r.factor, sigdigits = 3), " of it")
   println(io, "   rms kappa used  : ", round(r.rms_kappa, sigdigits = 4))
   println(io, "   cap set at      : (", round(r.binding[1], digits = 2), ", ",
                                         round(r.binding[2], digits = 2), ") arcsec, kappa_M = ",
                                         round(r.binding[3], sigdigits = 3))
   println(io, "   image residual  : ", round(r.image_residual, sigdigits = 3), " arcsec")
   println(io, "   source shift    : ", r.shift === :none ? "none (alpha_P = 0 at images)" :
                                        string(r.shift, ", max ",
                                               round(r.max_shift, sigdigits = 3), " arcsec"))
   println(io, "   parity flips    : ", isempty(r.flipped) ? "none" : string(r.flipped))
   println(io, "   max |mu| change : x", round(r.mu_ratio, sigdigits = 3))
   println(io, "   mass moved      : ", round(r.total_mass, sigdigits = 4), " Msun (net ",
                                        round(r.net_mass, sigdigits = 4), ")")
   print_check(r.check; io = io)
   return nothing
end


function realisation(model::init_BestModel, space::init_DegeneracySpace; relax::Float64 = 0.0,
                     kappa_min::Float64 = 0.0, safety::Float64 = 1.0, n_far::Int64 = 1,
                     n_scan::Int64 = 8, iters::Int64 = 12,
                     rng::AbstractRNG = Random.default_rng())
   shade = init_ShaDes(space; relax = relax, rng = rng)
   shade, positivity, binding = cap_positivity(shade, model; kappa_min = kappa_min,
                                               safety = safety)

   shade, factor = cap_multiplicity(shade, model; n_scan = n_scan, iters = iters, n_far = n_far)

   check = check_shade(shade, model; n_far = n_far)
   flipped, mu_model, mu_shade = parity_flips(shade, model)
   mu_ratio = maximum(abs(mu_shade[i] / mu_model[i]) for i in eachindex(mu_model))

   res = image_residuals(shade)
   image_res = maximum(hypot(res[i, 1], res[i, 2]) for i in axes(res, 1))

   report = init_ShaDesReport(space.basis.scale, first(space.basis.x_s),
                              length(space.basis.x_c), degeneracy_dimension(space),
                              positivity, factor, factor * safety * positivity, check, binding,
                              image_res, shade.shift, max_shift(shade),
                              flipped, mu_ratio, total_mass(shade), net_mass(shade))
   return shade, report
end


function realisation(model::init_BestModel, imgs::init_ImageSet; scale::Float64 = NaN,
                     core::Float64 = NaN, rtol::Float64 = 1e-8, shift::Symbol = :none,
                     relax::Float64 = 0.0,
                     kappa_min::Float64 = 0.0, safety::Float64 = 1.0, n_far::Int64 = 1,
                     n_scan::Int64 = 8, iters::Int64 = 12,
                     rng::AbstractRNG = Random.default_rng())
   basis = init_PlummerBasis(model.D_d, imgs; scale = scale, core = core)
   space = init_DegeneracySpace(basis, imgs; rtol = rtol, shift = shift)

   return realisation(model, space; relax = relax, kappa_min = kappa_min, safety = safety,
                      n_far = n_far, n_scan = n_scan, iters = iters, rng = rng)
end