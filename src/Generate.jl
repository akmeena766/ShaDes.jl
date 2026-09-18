# --------------------------------------------------------------------------------------------------
# Source-plane shift modes
# --------------------------------------------------------------------------------------------------
const SHIFT_MODES = (:none, :knot, :source, :global)


function _group_key(row::AbstractVector{Float64}, shift::Symbol)
   if shift === :knot
      return (Int64(row[COL_SRC]), Int64(row[COL_KNOT]))
   elseif shift === :source
      return (Int64(row[COL_SRC]), 0)
   elseif shift === :global
      return (0, 0)
   else
      throw(ArgumentError("shift must be one of $(SHIFT_MODES); got $shift."))
   end
end


"""
    shift_groups(imgs::init_ImageSet, shift::Symbol)
Partition the rows of `imgs.data` into the sets of images that must share one constant
deflection.

# Arguments
- `imgs` : ImageSet containing the image positions.
- `shift` : Shift mode to use. The allowed shift modes are 
   - `:none`: No constraint on the deflection.
   - `:knot`: Images with the same knot share the same constant deflection.
   - `:source`: Images with the same source share the same constant deflection.
   - `:global`: All images share the same constant deflection.

# Returns
- Vector of image indices that share the same constant deflection.
"""
function shift_groups(imgs::init_ImageSet, shift::Symbol)
   if !(shift in SHIFT_MODES)
      throw(ArgumentError("shift must be one of $(SHIFT_MODES); got $shift."))
   end
   if shift === :none
      return Vector{Int64}[]
   end

   n    = size(imgs.data, 1)
   keys = [_group_key(view(imgs.data, i, :), shift) for i in 1:n]
   return [findall(isequal(k), keys) for k in unique(keys)]
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
                      obs::Matrix{Float64}; 
                      scale::Float64 = NaN, 
                      core::Float64  = NaN)
"""
function init_PlummerBasis(D_d::Float64, obs::Matrix{Float64}; 
                           scale::Float64 = NaN, 
                           core::Float64  = NaN)
   if isnan(scale)
      scale = 0.5 * critical_scale(obs)
   end
   
   if isnan(core)
      core = 1.5 * scale
   end
   
   centres = grid_centres(obs, scale)
   m       = size(centres, 1)
   x_s     = fill(float(core), m)
   return init_PlummerBasis(D_d=D_d, x_c=centres[:, 1], y_c=centres[:, 2], x_s=x_s, scale=scale)
end

"""
    init_PlummerBasis(D_d::Float64, imgs::init_ImageSet; scale::Float64=NaN, core::Float64=NaN)
"""
function init_PlummerBasis(D_d::Float64, imgs::init_ImageSet; scale::Float64=NaN, core::Float64=NaN)
   return init_PlummerBasis(D_d, positions(imgs); scale = scale, core = core)
end


"""
    enclosing_ellipse(obs::Matrix{Float64}; inflate::Float64=1.01, tol::Float64=1E-7, maxiter::Int64=10_000)
"""
function enclosing_ellipse(obs::Matrix{Float64}; inflate::Float64=1.01, tol::Float64=1E-7, maxiter::Int64=10_000)
   # Check if we have more than one image
   n, d = size(obs, 1), size(obs, 2)
   if n < 4
      throw(ArgumentError("Need at least four images."))
   end

   # Khachiyan's iteration on the lifted points
   Q = vcat(obs', ones(1, n))
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
   centre = obs' * u
   A = inv(obs' * Diagonal(u) * obs - centre * centre') ./ d
   F = eigen(Symmetric(A))
   return centre, F.vectors, inflate ./ sqrt.(F.values)
end


"""
    critical_scale(obs::Matrix{Float64})
Estimate a characteristic scale (i.e., seperation between basis componenets) for the image 
configuration based on the enclosing ellipse.

# Arguments
- `obs::Matrix{Float64}`: Matrix of observed image positions (n images × 2 coordinates).

# Returns
- `Float64`: The estimated scale.
"""
function critical_scale(obs::Matrix{Float64})
   _, _, semi = enclosing_ellipse(obs)
   return sqrt(π * semi[1] * semi[2] / (2 * size(obs, 1)))
end


"""
    grid_centres(obs::Matrix{Float64}, scale::Float64)
"""
function grid_centres(obs::Matrix{Float64}, scale::Float64)
   if scale ≤ 0
      throw(ArgumentError("scale must be positive; got $scale."))
   end
   
   # Get image-enclosing ellipse parameter
   centre, axes_, semi = enclosing_ellipse(obs)

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
   imgs::init_ImageSet
   U::Matrix{Float64}
   S::Vector{Float64}
   Vt::Matrix{Float64}
   rtol::Float64
   shift::Symbol
end

"""
    init_DegeneracySpace(basis::init_PlummerBasis, 
                         obs::Matrix{Float64}; 
                         rtol::Float64 = 1E-8)
"""
function init_DegeneracySpace(basis::init_PlummerBasis, imgs::init_ImageSet;
                              rtol::Float64 = 1e-8, shift::Symbol = :none)
   groups = shift_groups(imgs, shift)

   # A group of one image constrains nothing at all once its source is free to move: the perturbation
   # can deflect that image anywhere and the shift absorbs it.
   lonely = count(g -> length(g) == 1, groups)
   if lonely > 0
      @warn "shift = $(shift): $(lonely) group(s) contain a single image and so place no " *
            "constraint on the perturbation.  Their source positions absorb whatever deflection " *
            "the perturbation happens to produce."
   end

   A = constraint_matrix(basis, positions(imgs), groups)
   F = svd(A; full=true)
   return init_DegeneracySpace(basis, imgs, F.U, F.S, F.Vt, rtol, shift)
end


"""
    deflection_table(basis::init_PlummerBasis, obs::Matrix{Float64})
"""
function deflection_table(basis::init_PlummerBasis, obs::Matrix{Float64})
   n = size(obs, 1)
   m = length(basis.x_c)
   Ax = Matrix{Float64}(undef, n, m)
   Ay = Matrix{Float64}(undef, n, m)
   @inbounds for j in 1:m
      lens = Lenses.init_PlummerLens(D_d=basis.D_d, x_c=basis.x_c[j], y_c=basis.y_c[j], mass=1.0, x_s=basis.x_s[j])
      for i in 1:n
         ax, ay = Lenses.get_deflection(lens, obs[i, 1], obs[i, 2])
         Ax[i, j] = ax
         Ay[i, j] = ay
      end
   end
   return Ax, Ay
end


"""
    constraint_matrix(basis::init_PlummerBasis, obs::Matrix{Float64})
"""
function constraint_matrix(basis::init_PlummerBasis, obs::Matrix{Float64})
   Ax, Ay = deflection_table(basis, obs)
   return vcat(Ax, Ay)
end


"""
    constraint_matrix(basis::init_PlummerBasis, obs::Matrix{Float64}, groups::Vector{Vector{Int64}})
Constraint rows for the constant-deflection degeneracy.  Within each group the rows are the
deflection minus the group mean, so the null space is exactly the set of mass vectors whose
deflection is constant across every group.  Centring on the mean rather than differencing against a
reference image keeps the rows symmetric in the images; it makes the matrix rank deficient by
``2G`` on purpose, which the SVD absorbs.
"""
function constraint_matrix(basis::init_PlummerBasis, obs::Matrix{Float64},
                           groups::Vector{Vector{Int64}})
   if isempty(groups)
      return constraint_matrix(basis, obs)
   end

   Ax, Ay = deflection_table(basis, obs)
   n, m   = size(Ax)
   C      = Matrix{Float64}(undef, 2n, m)
   @inbounds for g in groups
      ng = length(g)
      for j in 1:m
         mx, my = 0.0, 0.0
         for i in g
            mx += Ax[i, j]
            my += Ay[i, j]
         end
         mx /= ng
         my /= ng
         for i in g
            C[i, j]     = Ax[i, j] - mx
            C[n + i, j] = Ay[i, j] - my
         end
      end
   end
   return C
end


"""
    degeneracy_dimension(space::init_DegeneracySpace)
"""
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


"""
    sample_masses(space::init_DegeneracySpace; relax::Float64=0.0, rng::AbstractRNG=Random.default_rng())
Draw a random mass vector from the degeneracy space.  The masses are drawn from a Gaussian
distribution in the SVD basis, with the standard deviation of each component set by the singular 
value.  The `relax` parameter sets a floor on the singular values, so that directions with very 
small singular values are not over-weighted.  The resulting mass vector is normalized to have a 
maximum absolute value of 1.0.  The `rng` parameter allows for specifying a random number generator 
for reproducibility.

# Arguments
- `space::init_DegeneracySpace`: The degeneracy space object.

# Keyword Arguments
- `relax = 0.0`: A non-negative value that sets a floor on the singular values.  Directions with 
   singular values below `relax * first(space.S)` are treated as free directions. Default is 0.0 
   (i.e., no relaxation).
- `rng = Random.default_rng()`: A random number generator for reproducibility.

# Returns
- `Vector{Float64}`: A random mass vector sampled from the degeneracy space, normalized to have a maximum absolute value of 1.0.
"""
function sample_masses(space::init_DegeneracySpace; relax::Float64=0.0, rng::AbstractRNG=Random.default_rng())
   return _draw(space, randn(rng, length(space.basis.x_c)), relax)
end


# --------------------------------------------------------------------------------------------------
# Realization
# --------------------------------------------------------------------------------------------------
"""
    init_ShaDes(basis::init_PlummerBasis, 
                masses::Vector{Float64}, 
                imgs::init_ImageSet)
"""
struct init_ShaDes
   basis::init_PlummerBasis
   masses::Vector{Float64}
   imgs::init_ImageSet
   shift::Symbol

   function init_ShaDes(basis::init_PlummerBasis, masses::Vector{Float64},
                        imgs::init_ImageSet, shift::Symbol)
      if length(masses) != length(basis.x_c)
         throw(ArgumentError("need one mass per component; got $(length(masses)) for $(length(basis.x_c)) components."))
      end
      if !(shift in SHIFT_MODES)
         throw(ArgumentError("shift must be one of $(SHIFT_MODES); got $shift."))
      end
      return new(basis, Vector{Float64}(masses), imgs, shift)
   end
end


"""
    init_ShaDes(basis::init_PlummerBasis, 
                masses::Vector{Float64}, 
                imgs::init_ImageSet)
"""
function init_ShaDes(basis::init_PlummerBasis, masses::Vector{Float64}, imgs::init_ImageSet)
   return init_ShaDes(basis, masses, imgs, :none)
end


"""
    init_ShaDes(space::init_DegeneracySpace; 
                relax::Float64   = 0.0, 
                rng::AbstractRNG = Random.default_rng())
"""
function init_ShaDes(space::init_DegeneracySpace; 
                     relax::Float64   = 0.0, 
                     rng::AbstractRNG = Random.default_rng())
   return init_ShaDes(space.basis, sample_masses(space; relax = relax, rng = rng), space.imgs,
                      space.shift)
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
multiple of a solution is a solution.  The amplitude is fixed by `cap_positivity` and
`cap_multiplicity`, not by the images.

# Arguments
- `shade::init_ShaDes`: The ShaDes object.
- `factor::Float64`: The scaling factor.

# Returns
- `init_ShaDes`: The rescaled ShaDes object.
"""
function rescale(shade::init_ShaDes, factor::Float64)
   return init_ShaDes(shade.basis, factor .* shade.masses, shade.imgs, shade.shift)
end