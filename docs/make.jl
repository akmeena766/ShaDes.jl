using Documenter
using LensFactory
using ShaDes
using Makie

makedocs(
   sitename = "ShaDes.jl",
   modules  = [ShaDes,
               Base.get_extension(ShaDes, :PlotExt)],
   format   = Documenter.HTML(; collapselevel = 1, 
                                assets = ["assets/custom.css"],
                                prettyurls = get(ENV, "CI", nothing) == "true"),
   pages = [
         "Home" => "index.md",
         "Plot Extension" => "plot.md"
      ]
)

deploydocs(
   repo = "github.com/akmeena766/ShaDes.jl.git",
   devbranch = "main",
   branch = "gh-pages"

)
