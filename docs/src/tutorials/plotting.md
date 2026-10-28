# Plotting Impulse Response Functions

LocalProjections.jl supports two plotting backends: **Plots.jl** and **Makie.jl**. Both draw the `LocalProjectionIRFResult` that [`as_irf_result`](@ref) returns, using the plotting methods of MacroEconometricTools.jl, so plots show the same estimates and bands as `summarize` and `as_irf_result`.

## Setup

All examples below assume you have estimated a local projection. Plots use the covariance estimator stored in the result (HC1 unless changed with `lp_result + vcov(estimator)`); passing an estimator as the second argument overrides it.

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices

df = DataFrame(y = cumsum(randn(200)), x = randn(200))
lp_result = lp(@formula(leads(y) ~ x), df; horizon = 12)
```

## Plots.jl

`plot(lp_result)` works as soon as Plots.jl is loaded.

### Basic Plot

```julia
using Plots

plot(lp_result)                                 # stored estimator
plot(lp_result + vcov(Bartlett{NeweyWest}()))   # HAC
plot(lp_result, HC3())                          # override
```

### Customization

Standard Plots.jl keyword arguments work alongside IRF-specific options:

```julia
plot(lp_result, HC1();
    term = :x,
    levels = [0.68, 0.95],
    title = "Impulse Response",
    xlabel = "Horizon",
    ylabel = "Response",
    legend = :topright)
```

### Comparing Estimators

```julia
p1 = plot(lp_result, HC1(); title = "HC1", legend = false)
p2 = plot(lp_result, HC3(); title = "HC3", legend = false)
p3 = plot(lp_result, Bartlett{NeweyWest}(); title = "Newey-West", legend = false)

plot(p1, p2, p3, layout = (1, 3), size = (900, 300))
```

## Makie.jl

`irfplot` and `irfplot!` have methods for local projections once a Makie backend (CairoMakie, GLMakie, WGLMakie) is loaded.

### Basic Plot

`irfplot` returns a `Figure`:

```julia
using CairoMakie

irfplot(lp_result)          # stored estimator
irfplot(lp_result, HC3())   # override
```

### Adding to an Existing Axis

`irfplot!` draws one panel onto an `Axis` and returns the axis:

```julia
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Horizon", ylabel = "Response", title = "Shock → y")
irfplot!(ax, lp_result)
fig
```

### Multi-Panel Figures

```julia
fig = Figure(size = (900, 300))
for (j, (name, est)) in enumerate(["HC1" => HC1(), "HC3" => HC3(), "Newey-West" => Bartlett{NeweyWest}()])
    ax = Axis(fig[1, j]; title = name, xlabel = "Horizon")
    irfplot!(ax, lp_result, est)
end
fig
```

### Customization

Keywords other than `term` and `levels` are passed to MacroEconometricTools' `irfplot`/`irfplot!`:

| Keyword | Default | Description |
|---------|---------|-------------|
| `bandcolor` | `:steelblue` | Color of confidence bands |
| `bandalpha` | `0.2` | Opacity of the widest band (narrower bands are darker) |
| `linecolor` | `:black` | Color of the point estimate line |
| `linewidth` | `2.0` | Width of the point estimate line |
| `drawzero` | `true` | Whether to draw a dashed zero line |
| `zerolinecolor` | `:gray60` | Color of the zero line |
| `zerolinestyle` | `:dash` | Line style of the zero line |
| `xtickstep` | `4` | X-axis tick spacing (0 to disable) |

`irfplot` also accepts layout keywords such as `title` and `size`; see `?irfplot`.

```julia
irfplot(lp_result;
    levels = [0.68, 0.90, 0.95],
    bandcolor = :steelblue,
    bandalpha = 0.3,
    linecolor = :darkblue,
    linewidth = 2.5)
```

## IRF-Specific Options

These options are common to both backends:

- **`term`**: Which coefficient's impulse response to plot. Defaults to the shock variable specified in `lp()` / `lpiv()`.
- **`levels`**: A vector of confidence levels. Multiple levels produce nested bands with decreasing opacity.

To scale IRFs (e.g., to percentage points), convert with `as_irf_result()`, apply `rescale` from MacroEconometricTools.jl, and plot the rescaled object with `irfplot` or `plot`.
