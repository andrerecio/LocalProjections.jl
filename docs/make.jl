using Documenter
using LocalProjections

makedocs(
    sitename = "LocalProjections.jl",
    modules = [LocalProjections],
    checkdocs = :exports,  # Only check exported functions
    pagesonly = true,      # skip stray .md files under docs/src (third-party reference suite)
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/basic.md",
            "Transformations" => "tutorials/transformations.md",
            "Inference and Plotting" => "tutorials/inference.md",
            "Ramey-Zubairy Multipliers" => "tutorials/ramey_zubairy.md"
        ],
        "Inference Procedures Guide" => "inference_procedures_guide.md",
        "API Reference" => "api.md"
    ]
)

# Uncomment for GitHub Pages deployment
# deploydocs(
#     repo = "github.com/YOUR_USERNAME/LocalProjections.jl.git",
#     devbranch = "main",
# )
