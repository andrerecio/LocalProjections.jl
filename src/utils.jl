# Run `run_draw(b)` for b = 1:nboot, on threads or serially. Each draw seeds
# its own RNG and writes only its own slots, so the two give identical results.
function _run_draws!(run_draw, nboot::Int, threaded::Bool)
    if threaded
        Threads.@threads for b in 1:nboot
            run_draw(b)
        end
    else
        for b in 1:nboot
            run_draw(b)
        end
    end
    return nothing
end

function _check_level(level::Real)
    level_f = Float64(level)
    (0 < level_f < 1) ||
        throw(ArgumentError("`level` must be in (0, 1) (got $level_f)"))
    return level_f
end
