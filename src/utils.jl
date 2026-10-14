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

function _check_sample(sample::Symbol)
    sample in (:horizon, :common) ||
        throw(ArgumentError("`sample` must be :horizon or :common (got :$sample)"))
    return sample
end

# The rows of `sample = :common`: the intersection of the horizon-specific
# complete-case masks `rowsof(h)`, h = 0:horizon. Without internal gaps this is
# the mask of the longest horizon; a gap removes every row it hits at any h.
function _common_rows(rowsof, horizon::Integer)
    rows = rowsof(0)::BitVector
    for h in 1:horizon
        rows .&= rowsof(h)::BitVector
    end
    any(rows) || throw(ArgumentError(
        "No observation is complete at every horizon 0:$horizon " *
        "(sample = :common); reduce `horizon` or use sample = :horizon"))
    return rows
end
