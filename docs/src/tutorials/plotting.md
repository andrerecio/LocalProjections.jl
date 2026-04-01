# Plotting Impulse Response Functions

LocalProjections.jl supports two plotting backends: **Plots.jl** (via RecipesBase) and **Makie.jl** (via a package extension).

## Setup

All examples below assume you have estimated a local projection and have a variance estimator ready:

```julia
using LocalProjections, DataFrames, StatsModels, CovarianceMatrices

df = DataFrame(y = cumsum(randn(200)), x = randn(200))
lp_result = lp(@formula(leads(y) ~ x), df; horizon = 12)
```

## Plots.jl

Plots.jl integration works out of the box via RecipesBase recipes.

### Basic Plot

```julia
using Plots

plot(lp_result, HC1())
```

You can pass a pre-computed covariance object instead of an estimator:

```julia
cov_result = vcov(HC1(), lp_result)
plot(lp_result, cov_result)
```

### Customization

Standard Plots.jl keyword arguments work alongside IRF-specific options:

```julia
plot(lp_result, HC1();
    term = :x,
    levels = [0.68, 0.95],
    irf_scale = 100.0,
    title = "Impulse Response",
    xlabel = "Horizon",
    ylabel = "Response (%)",
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

The Makie extension loads automatically when any Makie backend (CairoMakie, GLMakie, WGLMakie) is available. It provides three functions: `irfplot`, `irfplot!`, and `irfplot_axis`.

### Basic Plot

```julia
using CairoMakie

irfplot(lp_result, HC1())
```

### Adding to an Existing Axis

```julia
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "Horizon", ylabel = "Response")
irfplot!(ax, lp_result, HC1())
fig
```

### Complete Panel with `irfplot_axis`

`irfplot_axis` creates an `Axis` with labels and returns `(subfig, axis, plot)`:

```julia
fig = Figure()
irfplot_axis(fig[1, 1], lp_result, HC1(); title = "Shock → y")
fig
```

### Multi-Panel Figures

```julia
fig = Figure(size = (900, 300))
irfplot_axis(fig[1, 1], lp_result, HC1(); title = "HC1")
irfplot_axis(fig[1, 2], lp_result, HC3(); title = "HC3")
irfplot_axis(fig[1, 3], lp_result, Bartlett{NeweyWest}(); title = "Newey-West")
fig
```

### Customization

The Makie recipe supports these attributes:

| Attribute | Default | Description |
|-----------|---------|-------------|
| `term` | shock variable | Which coefficient to plot |
| `levels` | `[0.95]` | Confidence levels for bands |
| `irf_scale` | `1.0` | Scaling factor for the IRF |
| `bandcolor` | `:blue` | Color of confidence bands |
| `bandalpha` | `0.25` | Base opacity of bands |
| `linecolor` | `:black` | Color of the point estimate line |
| `linewidth` | `2.0` | Width of the point estimate line |
| `drawzero` | `true` | Whether to draw a dashed zero line |
| `zerolinecolor` | `:gray70` | Color of the zero line |

```julia
irfplot(lp_result, HC1();
    levels = [0.68, 0.90, 0.95],
    bandcolor = :steelblue,
    bandalpha = 0.3,
    linecolor = :darkblue,
    linewidth = 2.5,
    irf_scale = 100.0)
```

## IRF-Specific Options

These options are common to both backends:

- **`term`**: Which coefficient's impulse response to plot. Defaults to the shock variable specified in `lp()` / `lpiv()`.
- **`levels`**: A vector of confidence levels. Multiple levels produce nested bands with decreasing opacity.
- **`irf_scale`**: Multiply the IRF and standard errors by this factor (useful for converting to percentage points).
