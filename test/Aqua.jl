using Test
using Aqua
using LocalProjections

@testset "Aqua.jl" begin
    Aqua.test_all(
        LocalProjections;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        stale_deps = true,
        deps_compat = true,
        # StatsModels methods (apply_schema, termvars) are intentional for custom
        # formula terms like leads(), cumul(), anchor(). This is standard practice
        # for extending StatsModels with new term types.
        piracies = false
    )
end
