module ShaDes


# --------------------------------------------------------------------------------------------------
# Julia inbuilt functions to import
# --------------------------------------------------------------------------------------------------
using LinearAlgebra
using Random
using Statistics
using LensFactory


# --------------------------------------------------------------------------------------------------
# Local functions to import
# --------------------------------------------------------------------------------------------------
include("./ShaDesV2.jl")


# --------------------------------------------------------------------------------------------------
# Functions to export
# --------------------------------------------------------------------------------------------------
export init_BestModel
export init_SourceSet
export init_PlummerBasis
export init_DegeneracySpace
export shift_groups
export shift_deflections
export source_shifts
export shifted_sources


# --------------------------------------------------------------------------------------------------
# Plotting support
# --------------------------------------------------------------------------------------------------
export plot_shade
export plot_caustic

function plot_shade end
function plot_caustic end


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
# Column layout of the observation table.  One row per observed image.
const COL_SRC  = 1
const COL_KNOT = 2
const COL_OBSX = 3
const COL_OBSY = 4
const COL_SRCX = 5
const COL_SRCY = 6

"""
    init_SourceSet(data::Matrix{Float64}, adis::Vector{Float64})
Store the lensed image constraints. The input data is a ``N_{\\rm knot} \\times 6`` Matrix table 
with columns `src_id`, `knot_id`, `obs_x`, `obs_y`, `src_x`, `src_y` (in ``\\rm \\mathbf{arcseconds}``).

# Arguments
- `data` : ``N_{\\rm knot} \\times 6`` matrix.
- `adis` : ``N_{\\rm src}`` vector of distance ratios, indexed by `src_id`.
"""
struct init_SourceSet
   data::Matrix{Float64}
   adis::Vector{Float64}

   function init_SourceSet(data::Matrix{Float64}, adis::Vector{Float64})
      if size(data, 2) != 6
         throw(ArgumentError("data must have 6 columns " *"(src_id, knot_id, obs_x, obs_y, src_x, src_y); got $(size(data, 2))."))
      end

      if size(data, 1) < 1
         throw(ArgumentError("data has no rows."))
      end
 
      if !all(x -> x ≥ 1 && x == round(x), @view data[:, COL_SRC:COL_KNOT])
         throw(ArgumentError("src_id and knot_id must be positive whole numbers."))
      end
 
      n_src = Int64(maximum(@view data[:, COL_SRC]))
      if length(adis) != n_src
         throw(ArgumentError("adis must be equal to number of source; got $(length(adis)) entries but src_id runs up to $(n_src)."))
      end
      return new(copy(data), copy(adis))
   end
end


function knot_table(sources::init_SourceSet)
   return unique(sources.data[:, [COL_SRC, COL_KNOT, COL_SRCX, COL_SRCY]], dims = 1)
end


function n_sources(sources::init_SourceSet)
   return length(unique(@view sources.data[:, COL_SRC]))
end


function n_knots(sources::init_SourceSet)
   return size(knot_table(sources), 1)
end


function images(sources::init_SourceSet)
   return sources.data[:, COL_OBSX:COL_OBSY]
end


function images_of(sources::init_SourceSet, src_id::Int64, knot_id::Int64)
   rows = (sources.data[:, COL_SRC] .== src_id) .& (sources.data[:, COL_KNOT] .== knot_id)
   return sources.data[rows, COL_OBSX:COL_OBSY]
end


function beta_residual(model::init_BestModel, sources::init_SourceSet)
   knots = knot_table(sources)
   k = size(knots, 1)
   residual = Vector{Float64}(undef, k)
   scatter  = Vector{Float64}(undef, k)

   @inbounds for j in 1:k
      s_id, k_id = Int64(knots[j, 1]), Int64(knots[j, 2])
      adis = sources.adis[s_id]
      img  = images_of(sources, s_id, k_id)
      n = size(img, 1)
      bx = Vector{Float64}(undef, n)
      by = Vector{Float64}(undef, n)
      for i in 1:n
         ax, ay = Lenses.get_deflection(model.lens, img[i, 1], img[i, 2])
         bx[i] = img[i, 1] - adis * ax
         by[i] = img[i, 2] - adis * ay
      end
      mx, my = mean(bx), mean(by)
      residual[j] = hypot(mx - knots[j, 3], my - knots[j, 4])
      if n > 1
         scatter[j] = maximum(hypot(bx[i] - mx, by[i] - my) for i in 1:n)
      else
         scatter[j] = 0.0
      end
   end
   return residual, scatter
end


# --------------------------------------------------------------------------------------------------
# Perturbation basis
# --------------------------------------------------------------------------------------------------
struct init_PlummerBasis
   D_d::Float64
   x_c::Vector{Float64}
   y_c::Vector{Float64}
   x_s::Vector{Float64}
   scale::Float64
end

"""
    init_PlummerBasis(; D_d::Float64         = NaN, 
                        x_c::Vector{Float64} = Float64[], 
                        y_c::Vector{Float64} = Float64[], 
                        x_s::Vector{Float64} = Float64[])
"""
function init_PlummerBasis(; D_d::Float64         = NaN, 
                             x_c::Vector{Float64} = Float64[], 
                             y_c::Vector{Float64} = Float64[], 
                             x_s::Vector{Float64} = Float64[],
                             scale::Float64       = NaN)
   if !(length(x_c) == length(y_c) == length(x_s))
      throw(ArgumentError("x_c, y_c and x_s must have the same length (one entry per component)."))
   end
   return init_PlummerBasis(D_d, x_c, y_c, x_s, scale)
end

"""
    init_PlummerBasis(D_d::Float64, 
                      images::Matrix{Float64}; 
                      scale::Float64 = NaN, 
                      core::Float64  = NaN)
"""
function init_PlummerBasis(D_d::Float64, images::Matrix{Float64}; 
                           scale::Float64 = NaN, 
                           core::Float64  = NaN)
   if isnan(scale)
      scale = 0.5 * critical_scale(images)
   end
   
   if isnan(core)
      core = 1.5 * scale
   end
   
   centres = grid_centres(images, scale)
   m       = size(centres, 1)
   x_s     = fill(float(core), m)
   return init_PlummerBasis(D_d=D_d, x_c=centres[:, 1], y_c=centres[:, 2], x_s=x_s, scale=scale)
end

"""
    init_PlummerBasis(D_d::Float64, sources::init_SourceSet; scale::Float64=NaN, core::Float64=NaN)
"""
function init_PlummerBasis(D_d::Float64, sources::init_SourceSet; scale::Float64=NaN, core::Float64=NaN)
   return init_PlummerBasis(D_d, images(sources); scale = scale, core = core)
end



function enclosing_ellipse(images::Matrix{Float64}; inflate::Float64=1.01, tol::Float64=1E-7, maxiter::Int64=10_000)
   # Check if we have more than one image
   n, d = size(images, 1), size(images, 2)
   if n < 4
      throw(ArgumentError("Need at least four images."))
   end

   # Khachiyan's iteration on the lifted points
   Q = vcat(images', ones(1, n))
   u = fill(1 / n, n)
   for _ in 1:maxiter
      Xi = inv(Q * Diagonal(u) * Q')
      m  = [dot(view(Q, :, i), Xi * view(Q, :, i)) for i in 1:n]
      j  = argmax(m)
      step = (m[j] - d - 1) / ((d + 1) * (m[j] - 1))
      u_new = (1 - step) .* u
      u_new[j] += step
      if norm(u_new - u) < tol
         u = u_new
         break
      end
      u = u_new
   end

   # Ellipse in the form (x - c)' A (x - c) <= 1
   centre = images' * u
   A = inv(images' * Diagonal(u) * images - centre * centre') ./ d
   F = eigen(Symmetric(A))
   return centre, F.vectors, inflate ./ sqrt.(F.values)
end


function critical_scale(images::Matrix{Float64})
   _, _, semi = enclosing_ellipse(images)
   return sqrt(π * semi[1] * semi[2] / (2 * size(images, 1)))
end


function grid_centres(images::Matrix{Float64}, scale::Float64)
   if scale ≤ 0
      throw(ArgumentError("scale must be positive; got $scale."))
   end
   
   # Get image-enclosing ellipse parameter
   centre, axes_, semi = enclosing_ellipse(images)

   # Lay a square lattice over the ellipse in its own frame, then keep what falls inside
   nx = ceil(Int, semi[1] / scale)
   ny = ceil(Int, semi[2] / scale)
   out = Vector{Vector{Float64}}()
   for i in -nx:nx
      for j in -ny:ny
         z1, z2 = i * scale, j * scale
         if (z1 / semi[1])^2 + (z2 / semi[2])^2 <= 1.0
            push!(out, centre .+ axes_[:, 1] .* z1 .+ axes_[:, 2] .* z2)
         end
      end
   end

   # Throw error if zero basic functions
   if isempty(out)
      throw(ArgumentError("scale = $scale is larger than the image field."))
   end
   return permutedims(reduce(hcat, out))
end


# --------------------------------------------------------------------------------------------------
# Degenracy space
# --------------------------------------------------------------------------------------------------
struct init_DegeneracySpace
   basis::init_PlummerBasis
   sources::init_SourceSet
   U::Matrix{Float64}
   S::Vector{Float64}
   Vt::Matrix{Float64}
   rtol::Float64
end

"""
    init_DegeneracySpace(basis::init_PlummerBasis, 
                         images::Matrix{Float64}; 
                         rtol::Float64 = 1E-8)
"""
function init_DegeneracySpace(basis::init_PlummerBasis, sources::init_SourceSet; rtol::Float64=1e-8)
   A = constraint_matrix(basis, images(sources))
   F = svd(A; full=true)
   return init_DegeneracySpace(basis, sources, F.U, F.S, F.Vt, rtol)
end


function constraint_matrix(basis::init_PlummerBasis, images::Matrix{Float64})
   n = size(images, 1)
   m = length(basis.x_c)
   A = Matrix{Float64}(undef, 2n, m)
   @inbounds for j in 1:m
      lens = Lenses.init_PlummerLens(D_d=basis.D_d, x_c=basis.x_c[j], y_c=basis.y_c[j], mass=1.0, x_s=basis.x_s[j])
      for i in 1:n
         ax, ay = Lenses.get_deflection(lens, images[i, 1], images[i, 2])
         A[i, j]     = ax
         A[n + i, j] = ay
      end
   end
   return A
end


function degeneracy_dimension(space::init_DegeneracySpace)
   m = length(space.basis.x_c)
   return m - count(space.S .> space.rtol * first(space.S))
end


function _draw(space::init_DegeneracySpace, g::Vector{Float64}, relax::Float64)
   m = length(space.basis.x_c)
   ns = length(space.S)

   if relax <= 0
      keep = trues(m)
      keep[1:ns] .= space.S .< space.rtol * first(space.S)
      any(keep) || throw(ErrorException("no degeneracy at this scale; use a finer `scale` " *
                                        "(see `critical_scale`) or a non-zero `tol`."))
      return space.Vt'[:, keep] * g[1:count(keep)]
   end

   # Weight each direction by 1 / max(sigma, floor): small sigma -> free, large sigma -> suppressed
   floor_ = relax * first(space.S)
   w = fill(1 / floor_, m)
   @inbounds for i in 1:ns
      w[i] = 1 / max(space.S[i], floor_)
   end
   b = space.Vt' * (w .* g)
   return b ./ maximum(abs, b)
end


function sample_masses(space::init_DegeneracySpace; relax::Float64=0.0, rng::AbstractRNG=Random.default_rng())
   return _draw(space, randn(rng, length(space.basis.x_c)), relax)
end


# --------------------------------------------------------------------------------------------------
# Realization
# --------------------------------------------------------------------------------------------------
"""
    init_ShaDes(basis::init_PlummerBasis, 
                masses::Vector{Float64}, 
                sources::init_SourceSet)
"""
struct init_ShaDes
   basis::init_PlummerBasis
   masses::Vector{Float64}
   sources::init_SourceSet

   function init_ShaDes(basis::init_PlummerBasis, masses::Vector{Float64}, sources::init_SourceSet)
      if length(masses) != length(basis.x_c)
         throw(ArgumentError("need one mass per component; got $(length(masses)) for $(length(basis.x_c)) components."))
      end
      return new(basis, Vector{Float64}(masses), sources)
   end
end


function init_ShaDes(space::init_DegeneracySpace; 
                     relax::Float64   = 0.0, 
                     rng::AbstractRNG = Random.default_rng())
   return init_ShaDes(space.basis, sample_masses(space; relax = relax, rng = rng), space.sources)
end


"""
    shade_lens(shade::init_ShaDes)
Construct a `LensFactory.Lenses.MultiPlummerLens` from the ShaDes object.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.

# Returns
- `Lenses.MultiPlummerLens`: The `LensFactory.Lenses.init_MultiPlummerLens` object.
"""
function shade_lens(shade::init_ShaDes)
   return Lenses.init_MultiPlummerLens(D_d  = shade.basis.D_d, 
                                       x_c  = shade.basis.x_c,
                                       y_c  = shade.basis.y_c, 
                                       mass = shade.masses,
                                       x_s  = shade.basis.x_s)
end


"""
    total_lens(shade::init_ShaDes, model::init_BestModel)
Construct the perturbed lens, ``M + P``, as a `LensFactory.Lenses.init_CompositeLens`.  A composite
best-fit model is flattened rather than nested, since `LensFactory` walks `_components_` only one
level deep.
"""
function total_lens(shade::init_ShaDes, model::init_BestModel)
   # Initialize the parts array using a standard if condition
   if model.lens._lens_ == :CompositeLens
      parts = copy(model.lens._components_)
   else
      parts = Lenses.AbstractLens[model.lens]
   end

   # Add the shade lens
   push!(parts, shade_lens(shade))

   # Return the updated composite lens
   return Lenses.init_CompositeLens(_components_ = parts)
end


"""
    shade_kappa(shade::init_ShaDes, θx::AbstractMatrix, θy::AbstractMatrix)
Convergence of the perturbation on a meshgrid, ``\\Delta\\kappa``.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.
- `θx::AbstractMatrix`: x-coordinates of the grid points (in arcseconds).
- `θy::AbstractMatrix`: y-coordinates of the grid points (in arcseconds).

# Returns
- `Matrix{Float64}`: Convergence values on the grid.
"""
function shade_kappa(shade::init_ShaDes, θx::Matrix{Float64}, θy::Matrix{Float64})
   κ, _, _ = Lenses.get_kappa_gamma(shade_lens(shade), θx, θy, 1.0)
   return κ
end


"""
    total_mass(shade::init_ShaDes)
Mass moved around by the ShaDes perturbations, ``\\sum_j |m_j|`` (in ``\\rm \\mathbf{M_\\odot}``).

# Arguments
- `shade::init_ShaDes`: The ShaDes object.

# Returns
- `Float64`: The total mass moved around by the ShaDes perturbations.
"""
function total_mass(shade::init_ShaDes)
   return sum(abs, shade.masses)
end


"""
    net_mass(shade::init_ShaDes)
Mass added on balance, ``\\sum_j m_j`` (in ``\\rm \\mathbf{M_\\odot}``).  Much smaller than
`total_mass` as the degeneracy mostly rearranges mass rather than adding it.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.

# Returns
- `Float64`: The net mass added by the ShaDes perturbations.
"""
function net_mass(shade::init_ShaDes)
   return sum(shade.masses)
end


"""
    rescale(shade::init_ShaDes, factor::Real)
Scale the whole perturbation. Still an exact degeneracy as the constraint is linear, so any
multiple of a solution is a solution.  The amplitude is fixed by `amplitude_cap`, not by the 
images.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.
- `factor::Float64`: The scaling factor.

# Returns
- `init_ShaDes`: The rescaled ShaDes object.
"""
function rescale(shade::init_ShaDes, factor::Float64)
   return init_ShaDes(shade.basis, factor .* shade.masses, shade.sources)
end


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
   lens = shade_lens(shade)
   obs  = images(shade.sources)
   n    = size(obs, 1)
   r    = Matrix{Float64}(undef, n, 2)
   @inbounds for i in 1:n
      ax, ay = Lenses.get_deflection(lens, obs[i, 1], obs[i, 2])
      r[i, 1] = ax
      r[i, 2] = ay
   end
   return r
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

function multiplicity(lens::Lenses.AbstractLens, model::init_BestModel, sources::init_SourceSet;
                      n_far::Int64 = 1)
   θx, θy = model.grid_x, model.grid_y
   ψxx, ψyy, ψxy = Lenses.get_jacobian(lens, θx, θy)

   knots = knot_table(sources)
   k = size(knots, 1)
   adis = [sources.adis[Int64(knots[j, 1])] for j in 1:k]
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
         N[j] = Lenses.get_image_multiplicity(curves, knots[j, 3], knots[j, 4]; n_far = n_far, verbose = false)
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
   knots::Matrix{Float64}
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


function magnification(lens::Lenses.AbstractLens, sources::init_SourceSet)
   n = size(sources.data, 1)
   mu = Vector{Float64}(undef, n)
   @inbounds for i in 1:n
      adis = sources.adis[Int64(sources.data[i, COL_SRC])]
      κ, γ1, γ2 = Lenses.get_kappa_gamma(lens, sources.data[i, COL_OBSX], sources.data[i, COL_OBSY], adis)
      mu[i] = 1.0 / ((1.0 - κ)^2 - γ1^2 - γ2^2)
   end
   return mu
end


function parity_flips(shade::init_ShaDes, model::init_BestModel)
   mu_model = magnification(model.lens, shade.sources)
   mu_shade = magnification(total_lens(shade, model), shade.sources)
   flipped = findall(i -> sign(mu_model[i]) != sign(mu_shade[i]), eachindex(mu_model))
   return flipped, mu_model, mu_shade
end


function check_shade(shade::init_ShaDes, model::init_BestModel; n_far::Int64 = 1)
   sources = shade.sources
   n_model = multiplicity(model.lens, model, sources; n_far = n_far)
   n_shade = multiplicity(total_lens(shade, model), model, sources; n_far = n_far)

   knots = knot_table(sources)
   for j in eachindex(n_model)
      s_id, k_id = Int64(knots[j, 1]), Int64(knots[j, 2])
      n_obs = size(images_of(sources, s_id, k_id), 1)
      if n_model[j] < n_obs
         @warn "source $(s_id), knot $(k_id): the best-fit model predicts $(n_model[j]) " *
               "image(s) but $(n_obs) are given.  Enlarge the grid, or check `beta_residual`, " *
               "before reading anything into the comparison." maxlog=1
      end
   end
   return init_ShaDesCheck(knot_table(sources), n_model, n_shade)
end


function cap_multiplicity(shade::init_ShaDes, model::init_BestModel; n_scan::Int64 = 8,
                          iters::Int64 = 12, n_far::Int64 = 1)
   if n_scan < 1
      throw(ArgumentError("n_scan must be at least 1; got $n_scan."))
   end

   sources = shade.sources
   n_model = multiplicity(model.lens, model, sources; n_far = n_far)

   function ok(f::Float64)
      n = multiplicity(total_lens(rescale(shade, f), model), model, sources; n_far = n_far)
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
   beta_residual::Float64
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
   println(io, "   beta residual   : ", round(r.beta_residual, sigdigits = 3), " arcsec")
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
   beta_res, _ = beta_residual(model, shade.sources)

   report = init_ShaDesReport(space.basis.scale, first(space.basis.x_s),
                              length(space.basis.x_c), degeneracy_dimension(space),
                              positivity, factor, factor * safety * positivity, check, binding,
                              image_res, maximum(beta_res), flipped, mu_ratio,
                              total_mass(shade), net_mass(shade))
   return shade, report
end


function realisation(model::init_BestModel, sources::init_SourceSet; scale::Float64 = NaN,
                     core::Float64 = NaN, rtol::Float64 = 1e-8, relax::Float64 = 0.0,
                     kappa_min::Float64 = 0.0, safety::Float64 = 1.0, n_far::Int64 = 1,
                     n_scan::Int64 = 8, iters::Int64 = 12,
                     rng::AbstractRNG = Random.default_rng())
   basis = init_PlummerBasis(model.D_d, sources; scale = scale, core = core)
   space = init_DegeneracySpace(basis, sources; rtol = rtol)

   return realisation(model, space; relax = relax, kappa_min = kappa_min, safety = safety,
                      n_far = n_far, n_scan = n_scan, iters = iters, rng = rng)
end


end