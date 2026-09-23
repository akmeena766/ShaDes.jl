using Documenter
using DocumenterCitations
using LensFactory
using ShaDes
using Makie

bib = CitationBibliography(joinpath(@__DIR__, "src", "References.bib"), style = :authoryear)

makedocs(
   sitename = "ShaDes.jl",
   plugins  = [bib],
   modules  = [ShaDes,
               Base.get_extension(ShaDes, :PlotExt)],
   format   = Documenter.HTML(; collapselevel = 1, 
                                assets = ["assets/custom.css"],
                                prettyurls = get(ENV, "CI", nothing) == "true"),
   pages = [
         "Introduction"   => "index.md",
         "Input"          => "Input.md",
         "Generation"     => "Generation.md",
         "Physical caps"  => "Caps.md",
         "Diagnostics"    => "Diagnostics.md",
         "Plot Extension" => "plot.md",
         "Bibliography"   => "References.md"
      ]
)

deploydocs(
   repo = "github.com/akmeena766/ShaDes.jl.git",
   devbranch = "main",
   branch = "gh-pages"

)
