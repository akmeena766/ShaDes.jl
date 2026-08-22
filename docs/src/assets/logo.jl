# docs/src/assets/logo.jl
#
# ShaDes.jl logo — a fan of conics through three fixed points.
#
# The three dots are the images.  Every curve drawn passes through all three of
# them exactly, so the mark states the package's premise: the data pins the
# model at the images and nowhere else.
#
# Construction.  Conics through three points form a two-parameter linear system,
# spanned by the three degenerate line-pair conics
#
#     A = L₁₂ · L₁₃,    B = L₁₂ · L₂₃,    C = L₁₃ · L₂₃
#
# (each vanishes on two lines that between them contain all three points).
# Normalise each to 1 at the centroid and A + B + C is the Steiner
# circumellipse — the smallest ellipse through the three points, drawn solid.
# Then
#
#     C(θ) = A + B + C + r ( cos(θ) A + cos(θ - 2π/3) B + cos(θ + 2π/3) C )
#
# sweeps a closed one-parameter family around it.  r ≲ 0.35 keeps every member a
# bounded ellipse; beyond that members escape to hyperbolas and leave the frame.
#
# Note there is no fan of *circles* here and there cannot be: exactly one circle
# passes through three given points.  Only the conics form a family.
#
#     julia logo.jl        -->  logo.png
#
# Needs Luxor and Colors; the rest is LinearAlgebra from stdlib.

using Luxor
using Colors
using LinearAlgebra


# --------------------------------------------------------------------------------------------------
# Julia's exact palette — the same RGB triples as LensFactory.jl's logo.jl,
# and the ones Luxor's own `juliacircles` uses.
# --------------------------------------------------------------------------------------------------
const JULIA_GREEN  = RGB(0.22,  0.596, 0.149)   # #389826
const JULIA_PURPLE = RGB(0.584, 0.345, 0.698)   # #9558B2
const JULIA_RED    = RGB(0.796, 0.235, 0.2)     # #CB3C33
const JULIA_BLUE   = RGB(0.251, 0.388, 0.847)   # #4063D8

# The three image positions, in a y-down frame.  Scalene on purpose: no mirror
# axis, no two sides equal, no two points at the same height.  A symmetric
# triangle makes the family symmetric too, and a symmetric family around three
# dots is a shape every reader has already filed under "orbits".  Real image
# configurations are not equilateral either.
const IMAGES     = [(-70.0, 18.0), (58.0, 42.0), (6.0, -58.0)]
const DOTCOLOURS = [JULIA_GREEN, JULIA_RED, JULIA_PURPLE]


# --------------------------------------------------------------------------------------------------
# Conics
# --------------------------------------------------------------------------------------------------
"Homogeneous coefficients `(a, b, c)` of the line `a x + b y + c = 0` through `p` and `q`."
line_through(p, q) = (p[2] - q[2], q[1] - p[1], p[1] * q[2] - q[1] * p[2])

"The degenerate conic `L₁ · L₂`, as coefficients `(a, b, c, d, e, f)` of `a x² + b x y + c y² + d x + e y + f`."
function line_pair(L1, L2)
   a1, b1, c1 = L1
   a2, b2, c2 = L2
   return (a1 * a2, a1 * b2 + a2 * b1, b1 * b2,
           a1 * c2 + a2 * c1, b1 * c2 + b2 * c1, c1 * c2)
end

"Evaluate a conic at a point."
function conic_at(co, p)
   a, b, c, d, e, f = co
   x, y = p
   return a * x^2 + b * x * y + c * y^2 + d * x + e * y + f
end

"""
    conic_basis(P)

The three degenerate conics through the three points `P`, each normalised to 1
at their centroid.  Any combination of them passes through all three points.
"""
function conic_basis(P)
   g = (sum(p -> p[1], P) / 3, sum(p -> p[2], P) / 3)
   A = line_pair(line_through(P[1], P[2]), line_through(P[1], P[3]))
   B = line_pair(line_through(P[1], P[2]), line_through(P[2], P[3]))
   C = line_pair(line_through(P[1], P[3]), line_through(P[2], P[3]))
   return A ./ conic_at(A, g), B ./ conic_at(B, g), C ./ conic_at(C, g)
end

"""
    ellipse_params(co)

`(cx, cy, rx, ry, θ)` for a conic, or `nothing` if it is not a real ellipse —
which is how members that have escaped to hyperbolas are dropped.
"""
function ellipse_params(co)
   a, b, c, d, e, f = co
   4a * c - b^2 > 1e-9 || return nothing
   M = [2a b; b 2c]
   abs(det(M)) > 1e-12 || return nothing
   h, k = M \ [-d, -e]
   fc = a * h^2 + b * h * k + c * k^2 + d * h + e * k + f
   abs(fc) > 1e-12 || return nothing
   F = eigen(Symmetric([a b/2; b/2 c] ./ (-fc)))
   all(>(0), F.values) || return nothing
   return (h, k, 1 / sqrt(F.values[1]), 1 / sqrt(F.values[2]),
           atan(F.vectors[2, 1], F.vectors[1, 1]))
end

"""
    fan(P; r = 0.30, n = 5)

`(steiner, members)` — the Steiner circumellipse of `P` and `n` further conics
around it.  Every curve returned passes exactly through all three points.
"""
function fan(P; r = 0.30, n = 5)
   A, B, C = conic_basis(P)
   base = A .+ B .+ C
   members = NTuple{5,Float64}[]
   for θ in range(0, 2π, length = n + 1)[1:end-1]
      co = base .+ r .* (cos(θ) .* A .+ cos(θ - 2π/3) .* B .+ cos(θ + 2π/3) .* C)
      p = ellipse_params(co)
      p === nothing || push!(members, p)
   end
   steiner = ellipse_params(base)
   steiner === nothing && error("the Steiner circumellipse is degenerate — are the points collinear?")
   return steiner, members
end

"Scale and centre that fit every ellipse in `es` inside a `box`-square with `pad` to spare."
function fit_transform(es; box = 512.0, pad = 28.0)
   xs = Float64[]
   ys = Float64[]
   for (h, k, rx, ry, θ) in es
      ex = hypot(rx * cos(θ), ry * sin(θ))      # half-width of the axis-aligned bounding box
      ey = hypot(rx * sin(θ), ry * cos(θ))
      append!(xs, (h - ex, h + ex))
      append!(ys, (k - ey, k + ey))
   end
   x0, x1 = extrema(xs)
   y0, y1 = extrema(ys)
   s = (box - 2pad) / max(x1 - x0, y1 - y0)
   return s, (x0 + x1) / 2, (y0 + y1) / 2
end


# --------------------------------------------------------------------------------------------------
# A check worth running: every curve must pass through every image
# --------------------------------------------------------------------------------------------------
"""
    incidence_error(images = IMAGES; r = 0.30, n = 5)

The worst `|f - 1|` over every curve and every image, where `f = 1` is the
ellipse equation.  Should be at the level of floating-point round-off; anything
larger means the construction is wrong, not merely ugly.
"""
function incidence_error(images = IMAGES; r = 0.30, n = 5)
   steiner, members = fan(images; r = r, n = n)
   worst = 0.0
   for (h, k, rx, ry, θ) in vcat(members, [steiner]), p in images
      u =  cos(θ) * (p[1] - h) + sin(θ) * (p[2] - k)
      v = -sin(θ) * (p[1] - h) + cos(θ) * (p[2] - k)
      worst = max(worst, abs((u / rx)^2 + (v / ry)^2 - 1))
   end
   return worst
end


# --------------------------------------------------------------------------------------------------
# The drawing
# --------------------------------------------------------------------------------------------------
"""
    create_shades_logo(filename = "logo.png"; box = 512, kwargs...)

Draw the mark and write it to `filename`.  The background is left transparent,
so the one file works on a light or a dark page.

`pad`, `linewidth` and `dotradius` are given for a 256-pixel box and scale with
`box`, so changing the output size changes nothing but the resolution.
"""
function create_shades_logo(filename = "logo.png";
                            box         = 512,          # output size, pixels square
                            images      = IMAGES,
                            dotcolours  = DOTCOLOURS,
                            n           = 5,            # fan members, besides the Steiner ellipse
                            r           = 0.30,         # how far the family drifts
                            pad         = 14,
                            linewidth   = 4.4,
                            dotradius   = 13.5,
                            fanopacity  = 0.34,
                            fillopacity = 0.05,
                            curve       = JULIA_BLUE,   # the Steiner ellipse
                            fancurve    = nothing)      # the family; defaults to `curve`

   u = box / 256                       # the numbers above are for a 256-pixel box
   fancurve = something(fancurve, curve)

   steiner, members = fan(images; r = r, n = n)
   s, cx, cy = fit_transform(vcat(members, [steiner]); box = float(box), pad = pad * u)
   place(h, k) = Point((h - cx) * s + box / 2, (k - cy) * s + box / 2)

   Drawing(box, box, filename)         # origin at the top-left corner, y downwards
                                       # no background() call: the PNG stays transparent

   # the family
   sethue(fancurve)
   setopacity(fanopacity)
   setline(linewidth * u)
   for (h, k, rx, ry, θ) in members
      @layer begin
         translate(place(h, k))
         rotate(θ)
         ellipse(O, 2rx * s, 2ry * s, :stroke)
      end
   end

   # the Steiner circumellipse — the smallest of them, drawn as the model
   bh, bk, brx, bry, bθ = steiner
   @layer begin
      translate(place(bh, bk))
      rotate(bθ)
      sethue(curve)
      setopacity(fillopacity)
      ellipse(O, 2brx * s, 2bry * s, :fill)
      setopacity(1.0)
      setline((linewidth + 1.6) * u)
      ellipse(O, 2brx * s, 2bry * s, :stroke)
   end

   # the images
   setopacity(1.0)
   for (p, col) in zip(images, dotcolours)
      sethue(col)
      circle(place(p[1], p[2]), dotradius * u, :fill)
   end

   finish()
   return filename
end


@info "worst incidence error over every curve and every image" incidence_error()
create_shades_logo(joinpath(@__DIR__, "logo.png"))