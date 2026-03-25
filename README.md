# GameTracer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mnshkw.github.io/GameTracer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mnshkw.github.io/GameTracer.jl/dev/)
[![Build Status](https://github.com/mnshkw/GameTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mnshkw/GameTracer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Julia wrapper for [gametracer](https://github.com/QuantEcon/gametracer), 
    providing IPA and GNM Nash equilibrium solvers for 
    [GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl) normal-form 
    games.

## Example usage

Consider the following 2x2x2 game with 9 Nash equilibria from McKelvey and 
    McLennan (1996):

```julia
using GameTheory, GameTracer, Random
g = NormalFormGame((2, 2, 2))
g[1, 1, 1] = 9, 8, 12
g[2, 2, 1] = 9, 8, 2
g[1, 2, 2] = 3, 4, 6
g[2, 1, 2] = 3, 4, 4
printin(g)
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

`ipa_solve` computes one mixed-action Nash equilibrium with the the iterated 
    polymatrix approximation (IPA) algorithm (Govindan and Wilson, 2004):
```julia
ray = [0.11, 0.37, 0.73, 0.19, 0.59, 0.83];
res = ipa_solve(g; ray=ray);
```
```
GameTracer.IPAResult{3}(([1.0, 0.0], [1.0, 0.0], [1.0, 0.0]), 1)
```

`gnm_solve` computes mixed-action Nash equilibria with the the global Newton 
    method (GNM) algorithm (Govindan and Wilson, 2003):

```julia
res = gnm_solve(g, ray=ray)
```
```
GameTracer.GNMResult{3}([([0.0, 1.0], [1.0, 0.0], [0.0, 1.0]), 
    ([0.2500000000000622, 0.7499999999999374], [1.0, 0.0], [0.25000000000008304,
     0.7499999999999165]), ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])], 3)
```

## License

This package is licensed under the GNU General Public License, version 3 or (at 
    your option) any later version. See [LICENSE](LICENSE).

It wraps [gametracer](https://github.com/QuantEcon/gametracer),
originally released under the GNU General Public License, version 2 or
(at your option) any later version.