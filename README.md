# ShaDes
[![Build status (Github Actions)](https://github.com/akmeena766/ShaDes.jl/workflows/CI/badge.svg)](https://github.com/akmeena766/ShaDes.jl/actions)
[![codecov.io](http://codecov.io/github/akmeena766/ShaDes.jl/coverage.svg?branch=main)](http://codecov.io/github/akmeena766/ShaDes.jl?branch=main)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://akmeena766.github.io/ShaDes.jl/stable)

---

# ShaDes.jl - Shape Degeneracies of a strong-lensing mass model

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
