using Documenter
using ShaDes

makedocs(
    sitename = "ShaDes.jl",
    format = Documenter.HTML(),
    modules = [ShaDes],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/akmeena766/ShaDes.jl.git",
    devbranch = "main",
    branch = "gh-pages"

)
