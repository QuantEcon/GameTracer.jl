# GameTracer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantEcon.github.io/GameTracer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/GameTracer.jl/dev/)
[![Build Status](https://github.com/QuantEcon/GameTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/GameTracer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/QuantEcon/GameTracer.jl/graph/badge.svg)](https://codecov.io/gh/QuantEcon/GameTracer.jl)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Julia wrapper for [gametracer](https://github.com/QuantEcon/gametracer), providing IPA and GNM Nash equilibrium solvers for [GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl) normal-form games.

## Example usage

Consider the following 2x2x2 game with 9 Nash equilibria from McKelvey and McLennan (1996):

```julia
using GameTheory, GameTracer, Random
seed = 1234

g = NormalFormGame((2, 2, 2))
g[1, 1, 1] = 9, 8, 12
g[2, 2, 1] = 9, 8, 2
g[1, 2, 2] = 3, 4, 6
g[2, 1, 2] = 3, 4, 4
println(g)
```
```
2×2×2 NormalFormGame{3, Float64}:
[:, :, 1] =
 [9.0, 8.0, 12.0]  [0.0, 0.0, 0.0]
 [0.0, 0.0, 0.0]   [9.0, 8.0, 2.0]

[:, :, 2] =
 [0.0, 0.0, 0.0]  [3.0, 4.0, 6.0]
 [3.0, 4.0, 4.0]  [0.0, 0.0, 0.0]
```

`ipa_solve` computes one mixed-action Nash equilibrium with the iterated polymatrix approximation (IPA) algorithm (Govindan and Wilson, 2004):

```julia
rng = MersenneTwister(seed)
res = ipa_solve(rng, g)
res.NE
```
```
([0.2500000414734812, 0.7499999585265188], [0.49999980486854023, 0.5000001951314598], [0.33333361994793975, 0.6666663800520602])
```

`gnm_solve` computes mixed-action Nash equilibria with the global Newton method (GNM) algorithm (Govindan and Wilson, 2003):

```julia
rng = MersenneTwister(seed)
res1 = gnm_solve(rng, g)
println(length(res1.NEs))
```
```
7
```

When `ray` is omitted, `GameTracer.jl` generates it internally with `rand(rng, sum(g.nums_actions))`. Therefore, repeated calls with the same `rng` object (the same RNG instance) use different rays as the RNG state advances.

```julia
res2 = gnm_solve(rng, g);
println(length(res2.NEs))
```
```
2
```

In the example above, the second call finds 2 equilibria while the first call finds 7. Different rays may yield different equilibria, or different numbers of equilibria, if found. `gnm_solve` is not guaranteed to find an equilibrium on an arbitrary run.

## License

This package is licensed under the GNU General Public License, version 3 or (at 
    your option) any later version. See [LICENSE](LICENSE).

It wraps [gametracer](https://github.com/QuantEcon/gametracer),
originally released under the GNU General Public License, version 2 or
(at your option) any later version.
