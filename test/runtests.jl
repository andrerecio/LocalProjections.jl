using TestItemRunner

# The Gorodnichenko–Lee fixtures (test/fixtures/goro_lee/*.csv) are generated
# locally and not committed; without them the :octave items are skipped.
const GL_FIXTURES = isfile(joinpath(@__DIR__, "fixtures", "goro_lee", "app_tfp_y_r2.csv"))
GL_FIXTURES || @info "Octave fixtures absent: skipping the :octave test items"

@run_package_tests filter = ti -> GL_FIXTURES || !(:octave in ti.tags)
