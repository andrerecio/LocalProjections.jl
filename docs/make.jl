using Documenter
using LocalProjections

makedocs(
    sitename = "LocalProjections.jl",
    modules = [LocalProjections],
    checkdocs = :exports,  # Only check exported functions
    pagesonly = true,      # skip .md files under docs/src that are not in `pages`
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/basic.md",
            "Transformations" => "tutorials/transformations.md",
            "Inference and Plotting" => "tutorials/inference.md",
            "Plotting" => "tutorials/plotting.md",
            "Ramey-Zubairy Multipliers" => "tutorials/ramey_zubairy.md",
            "FEVD (Gorodnichenko-Lee)" => "tutorials/fevd.md",
        ],
        "API Reference" => "api.md",
    ]
)

# Uncomment for GitHub Pages deployment
# deploydocs(
#     repo = "github.com/YOUR_USERNAME/LocalProjections.jl.git",
#     devbranch = "main",
# )
