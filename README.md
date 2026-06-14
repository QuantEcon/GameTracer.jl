# GameTracer.jl

[![Build Status](https://github.com/QuantEcon/GameTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/GameTracer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/QuantEcon/GameTracer.jl/graph/badge.svg)](https://codecov.io/gh/QuantEcon/GameTracer.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantEcon.github.io/GameTracer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/GameTracer.jl/dev/)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Julia wrapper for [gametracer](https://github.com/QuantEcon/gametracer),
exposing its Nash equilibrium solvers for `NormalFormGame` from
[GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl)


## Solvers

- [`ipa_solve`](https://quantecon.github.io/GameTracer.jl/stable/#GameTracer.ipa_solve):
  Compute one mixed-action approximate Nash equilibrium of a normal form game
  with the iterated polymatrix approximation (IPA) algorithm (Govindan and Wilson, 2004).
  It approximates the game by a sequence of polymatrix games, each solved by a
  variant of the Lemke-Howson algorithm.

- [`gnm_solve`](https://quantecon.github.io/GameTracer.jl/stable/#GameTracer.gnm_solve):
  Compute mixed-action Nash equilibria of a normal form game with the global
  Newton method (GNM) algorithm (Govindan and Wilson, 2003). It follows a
  homotopy path perturbing the game along a ray, starting from a game with a
  simple known equilibrium and tracing the path back to the original game to
  return equilibria encountered along that path.


## Example usage

A 2x2x2 example:

```julia
using GameTheory, GameTracer, Random

g = NormalFormGame((2, 2, 2))
g[1, 1, 1] = 9, 8, 12
g[2, 2, 1] = 9, 8, 2
g[1, 2, 2] = 3, 4, 6
g[2, 1, 2] = 3, 4, 4
display(g)
```
```
2×2×2 NormalFormGame{3, Float64}:
[:, :, 1] =
 (9.0, 8.0, 12.0)  (0.0, 0.0, 0.0)
 (0.0, 0.0, 0.0)   (9.0, 8.0, 2.0)

[:, :, 2] =
 (0.0, 0.0, 0.0)  (3.0, 4.0, 6.0)
 (3.0, 4.0, 4.0)  (0.0, 0.0, 0.0)
```

### `ipa_solve`

```julia
rng = MersenneTwister(1)
res = ipa_solve(rng, g)
res.NE
```
```
([0.25, 0.75], [0.5, 0.5], [0.333333, 0.666667])
```

### `gnm_solve`

```julia
rng = MersenneTwister(23)
res = gnm_solve(rng, g)
res.NEs
```
```
9-element Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}:
 ([1.0, 0.0], [0.0, 1.0], [0.0, 1.0])
 ([0.5, 0.5], [0.333333, 0.666667], [0.25, 0.75])
 ([0.0, 1.0], [0.0, 1.0], [1.0, 0.0])
 ([0.0, 1.0], [0.333333, 0.666667], [0.333333, 0.666667])
 ([0.25, 0.75], [0.5, 0.5], [0.333333, 0.666667])
 ([0.5, 0.5], [0.5, 0.5], [1.0, 0.0])
 ([1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
 ([0.25, 0.75], [1.0, 0.0], [0.25, 0.75])
 ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0])
```


## References

- S. Govindan and R. Wilson, "A global Newton method to compute Nash
  equilibria," Journal of Economic Theory, 110 (2003), 65-86.
- S. Govindan and R. Wilson, "Computing Nash equilibria by iterated polymatrix
  approximation," Journal of Economic Dynamics and Control, 28 (2004),
  1229-1241.


## License

This package is licensed under the GNU General Public License, version 3
or (at your option) any later version. See [LICENSE](LICENSE).

It wraps [gametracer](https://github.com/QuantEcon/gametracer),
originally released under the GNU General Public License, version 2 or
(at your option) any later version.
