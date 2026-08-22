# ShaDes
[![Build Status](https://github.com/akmeena766/ShaDes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/akmeena766/ShaDes.jl/actions/workflows/CI.yml?query=branch%3Amain)
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
