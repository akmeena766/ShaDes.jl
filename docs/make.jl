using Documenter
using ShaDes

makedocs(
   sitename = "ShaDes.jl",
   modules = [ShaDes],
   format = Documenter.HTML(; collapselevel = 1, 
                              assets = ["assets/custom.css"],
                              prettyurls = get(ENV, "CI", nothing) == "true"),
   pages = [
         "Home" => "index.md",
      ]
)

deploydocs(
   repo = "github.com/akmeena766/ShaDes.jl.git",
   devbranch = "main",
   branch = "gh-pages"

)
