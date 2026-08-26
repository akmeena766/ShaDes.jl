# ShaDes.jl
[![Build status (Github Actions)](https://github.com/akmeena766/ShaDes.jl/workflows/CI/badge.svg)](https://github.com/akmeena766/ShaDes.jl/actions)
[![codecov.io](http://codecov.io/github/akmeena766/ShaDes.jl/coverage.svg?branch=main)](http://codecov.io/github/akmeena766/ShaDes.jl?branch=main)

---

## Introduction
Given a best-fit model and the observed image positions, find the perturbations the data cannot
see. A perturbation $P$ leaves every image exactly where it is when its deflection vanishes there,
i.e.,
```math
\begin{equation}
\boldsymbol{\alpha}_P(\boldsymbol{\theta}_i) = 0,  \qquad i = 1, 2, \dots, N,
\end{equation}
```
where $\boldsymbol{\alpha}_P$ is the deflection angle field corresponding to perturbation $P$ and 
$\boldsymbol{\theta}_i$ is the position of the $i$-th image. 


---
## Best-fit model
```@docs
ShaDes.init_BestModel
```

---
## Lensed image constraints
```@docs
ShaDes.init_SourceSet
```

---
## Basis functions
```@docs
ShaDes.init_PlummerBasis
```

---
## Degeneracy space
```@docs
ShaDes.init_DegeneracySpace
```

---
## Degeneracy realization
```@docs
ShaDes.init_ShaDes
ShaDes.shade_lens
ShaDes.total_lens
ShaDes.shade_kappa
ShaDes.total_mass
ShaDes.net_mass
ShaDes.rescale
ShaDes.image_residuals
ShaDes.amplitude_cap
```