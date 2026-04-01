```@meta
CurrentModule = GameTracer
```

# GameTracer.jl

Julia wrapper for [gametracer](https://github.com/QuantEcon/gametracer),
exposing its Nash equilibrium solvers for `NormalFormGame` from
[GameTheory.jl](https://github.com/QuantEcon/GameTheory.jl)

## API Reference

### Iterated Polymatrix Approximation (IPA)

```@docs
ipa_solve
IPAResult
```

### Global Newton Method (GNM)

```@docs
gnm_solve
GNMResult
```
