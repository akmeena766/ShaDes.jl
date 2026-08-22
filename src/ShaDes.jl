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
# Best-model
# --------------------------------------------------------------------------------------------------
struct init_BestModel
   D_d::Float64
   grid_x::Matrix{Float64}
   grid_y::Matrix{Float64}
   kappa::Matrix{Float64}
end

function init_BestModel(D_d::Float64, lens::Lenses.AbstractLens, θx::Matrix{Float64}, θy::Matrix{Float64})
   if size(θx) == size(θy)
      κ, _, _ = Lenses.get_kappa_gamma(lens, θx, θy, 1.0)
   else
      throw(ArgumentError("grid_x and grid_y must have the same shape; got $(size(θx)) and $(size(θy))."))
   end
   return init_BestModel(D_d, θx, θy, kappa)
end


# --------------------------------------------------------------------------------------------------
# Perturbation basis
# --------------------------------------------------------------------------------------------------
struct init_PlummerBasis
   D_d::Float64
   x_c::Vector{Float64}
   y_c::Vector{Float64}
   x_s::Vector{Float64}
end

function init_PlummerBasis(; D_d::Float64 = NaN, x_c::Vector{Float64}=Float64[], y_c::Vector{Float64}=Float64[], x_s::Vector{Float64}=Float64[])
   if !(length(x_c) == length(y_c) == length(x_s))
      throw(ArgumentError("x_c, y_c and x_s must have the same length (one entry per component)."))
   end
   return init_PlummerBasis(D_d, x_c, y_c, x_s)
end

function init_PlummerBasis(D_d::Float64, images::Matrix{Float64}; scale::Float64=NaN, core::Float64=NaN)
   scale   = isnan(scale) ? 0.5 * critical_scale(images) : scale
   core    = isnan(core) ? 1.5 * scale : core
   centres = grid_centres(images, scale)
   mass    = size(centres, 1)
   x_s     = fill(float(core), mass)
   return init_PlummerBasis(D_d=D_d, x_c=centres[:, 1], y_c=centres[:, 2], x_s=x_s)
end



function enclosing_ellipse(images::Matrix{Float64}; inflate::Float64 = 1.1, 
                                                    tol::Float64     = 1E-7,
                                                    maxiter::Int64   = 10_000)
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
   images::Matrix{Float64}
   U::Matrix{Float64}
   S::Vector{Float64}
   Vt::Matrix{Float64}
   rtol::Float64
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


function init_DegeneracySpace(basis::init_PlummerBasis, images::Matrix{Float64}; rtol::Float64=1e-8)
   A = constraint_matrix(basis, images)
   F = svd(A; full=true)
   return init_DegeneracySpace(basis, images, F.U, F.S, F.Vt, rtol)
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


function image_motion(space::init_DegeneracySpace, masses::Vector{Float64})
   n = size(space.images, 1)
   ns = length(space.S)
   c = (space.Vt * masses)[1:ns] .* space.S
   d = space.U[:, 1:ns] * c
   return maximum(hypot(d[i], d[n + i]) for i in 1:n)
end


# --------------------------------------------------------------------------------------------------
# Realization
# --------------------------------------------------------------------------------------------------
struct init_ShaDes
   basis::init_PlummerBasis
   masses::Vector{Float64}
   images::Matrix{Float64}

   function init_ShaDes(basis::init_PlummerBasis, masses, images)
      length(masses) == length(basis.x_c) ||
         throw(ArgumentError("need one mass per component; got $(length(masses)) for " *
                             "$(length(basis.x_c)) components."))
      return new(basis, Vector{Float64}(masses), Matrix{Float64}(images))
   end
end


function shade_lens(shade::init_ShaDes)
   return Lenses.init_MultiPlummerLens(D_d = shade.basis.D_d, 
                                       x_c = shade.basis.x_c,
                                       y_c = shade.basis.y_c, 
                                       mass = shade.masses,
                                       x_s = shade.basis.x_s)
end

function total_mass(shade::init_ShaDes)
   return sum(abs, shade.masses)
end

function net_mass(shade::init_ShaDes)
   return sum(shade.masses)
end

function rescale(shade::init_ShaDes, factor::Float64)
   return init_ShaDes(shade.basis, factor .* shade.masses, shade.images)
end

function image_residuals(shade::init_ShaDes)
   lens = shade_lens(shade)
   n = size(shade.images, 1)
   r = Matrix{Float64}(undef, n, 2)
   @inbounds for i in 1:n
      ax, ay = Lenses.get_deflection(lens, shade.images[i, 1], shade.images[i, 2])
      r[i, 1] = ax
      r[i, 2] = ay
   end
   return r
end

function shade_kappa(shade::init_ShaDes, X::Matrix{Float64}, Y::Matrix{Float64})
   κ, _, _ = Lenses.get_kappa_gamma(shade_lens(shade), X, Y, 1.0)
   return κ
end

function amplitude_cap(shade::init_ShaDes, kappa_M::Matrix{Float64}, X::Matrix{Float64}, Y::Matrix{Float64};
                       kappa_min::Float64 = 0.1)
   dk = shade_kappa(shade, X, Y)
   rms = sqrt(mean(abs2, dk))
   if rms ≤ 0
      throw(ErrorException("perturbation has zero convergence."))
   end
   dk ./= rms

   cap = Inf
   @inbounds for k in eachindex(dk)
      if (kappa_M[k] > kappa_min && dk[k] < 0)
         cap = min(cap, kappa_M[k] / -dk[k])
      end
   end
   return cap
end


# --------------------------------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------------------------------
struct init_ShaDesEnsemble
   space::init_DegeneracySpace
   shades::Vector{init_ShaDes}
   cap::Float64
   x::Vector{Float64}
   y::Vector{Float64}
   kappa_M::Matrix{Float64}
end


function _ensemble_cap(shades::Vector{<:init_ShaDes}, kappa_M::Matrix, X::Matrix{Float64},
                       Y::Matrix{Float64}, kappa_min::Float64)
   caps = filter(isfinite, [amplitude_cap(s, kappa_M, X, Y; kappa_min = kappa_min) for s in shades])
   if isempty(caps)
      throw(ErrorException("positivity gives no bound: no realisation lowers kappa anywhere " *
            "with kappa_M > $(kappa_min).  The model's convergence peaks at " *
            "$(round(maximum(kappa_M), digits = 3)) on a grid spanning " *
            "$(round(extrema(X)[2] - extrema(X)[1], digits = 1)) x " *
            "$(round(extrema(Y)[2] - extrema(Y)[1], digits = 1)) arcsec.  If that peak is ~0 the " *
            "model or the image coordinates are wrong; if the grid is not tens of arcsec across, " *
            "the image positions are."))
   end
   return median(caps)
end


function explore(model, images::Matrix{Float64}; scale::Float64     = NaN, 
                                                 core::Float64      = NaN, 
                                                 tol::Float64       = 0.0, 
                                                 n::Int64           = 32,
                                                 kappa_min::Float64 = 0.1, 
                                                 rtol::Float64      = 1e-8,
                                                 rng::AbstractRNG   = Random.default_rng())
   if n ≤ 0
      throw(ArgumentError("need at least one realisation; got n = $n."))
   end

   if isnan(scale) 
      scale = 0.5 * critical_scale(images)
   end

   basis = init_PlummerBasis(model.D_d, images; scale=scale, core=core)
   
   space = init_DegeneracySpace(basis, images; rtol = rtol)

   X, Y = model.grid_x, model.grid_y
   kappa_M = model.kappa

   # The exact null space sets the amplitude, because positivity -- not the images -- is what
   # bounds it, and because the cap is what makes `tol` an arcsecond rather than a bare number.
   gs = [randn(rng, length(basis.x_c)) for _ in 1:n]
   raw = [init_ShaDes(basis, _draw(space, g, 0.0), images) for g in gs]
   cap = _ensemble_cap(raw, kappa_M, X, Y, kappa_min)

   # Relaxing the constraint changes which directions are drawn, which changes the cap, which
   # changes what `tol` arcsec means.  Two passes are enough: the cap moves by ~15 % on the first
   # and by under a per cent on the second.
   if tol > 0
      throw(ArgumentError("tol > 0 not implemented in this version; use tol = 0."))
   end

   shades = [rescale(s, cap / sqrt(mean(abs2, shade_kappa(s, X, Y)))) for s in raw]
   return init_ShaDesEnsemble(space, shades, cap, x, y, kappa_M)
end


end