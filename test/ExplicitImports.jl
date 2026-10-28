@testitem "ExplicitImports" tags = [:explicitimports] begin
    using LocalProjections
    using ExplicitImports
    # Upstream packages document these names but do not declare them `public`:
    # `lag`/`lead` (ShiftedArrays, re-exported here), and the StatsModels/Tables
    # names used to extend `apply_schema`, `termvars`, and `modelcols`, and
    # the Tables.jl interface functions `istable`, `columnaccess`, and `columns`.
    test_explicit_imports(
        LocalProjections;
        all_explicit_imports_are_public = (; ignore = (:lag, :lead)),
        all_qualified_accesses_are_public = (;
            ignore = (:Schema, :termvars, :ColumnTable, :istable, :columnaccess, :columns),
        ),
    )
end
