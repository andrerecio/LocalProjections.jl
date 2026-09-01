#!/usr/bin/env julia
"""
Format the package with the SciML style, from the local `dev` environment.

CI auto-commits sciml formatting on every push, so running this first keeps the
diff clean.

    julia --project=dev dev/format.jl

Only the repository's own Julia sources are touched. `docs/src/code/` is skipped
explicitly: it holds the untracked third-party MATLAB replication suite, which
is read-only reference material and must never be reformatted.
"""

using JuliaFormatter

const ROOT = dirname(@__DIR__)
const TARGETS = ["src", "ext", "test", "benchmark", "dev",
    "docs/make.jl", "docs/make_rz_figure.jl"]

function main()
    allclean = true
    for t in TARGETS
        path = joinpath(ROOT, t)
        ispath(path) || continue
        clean = format(path)
        allclean &= clean
        println(clean ? "  ok       " : "  reformatted ", t)
    end
    if allclean
        println("\nAll files already formatted.")
    else
        println("\nSome files were reformatted — review `git diff` before committing.")
    end
    return allclean
end

if abspath(PROGRAM_FILE) == @__FILE__
    main() || exit(0)   # reformatting is not a failure; CI would do it anyway
end
