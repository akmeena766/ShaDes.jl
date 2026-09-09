# ShaDes.jl
[![Build status (Github Actions)](https://github.com/akmeena766/ShaDes.jl/workflows/CI/badge.svg)](https://github.com/akmeena766/ShaDes.jl/actions)
[![codecov.io](http://codecov.io/github/akmeena766/ShaDes.jl/coverage.svg?branch=main)](http://codecov.io/github/akmeena766/ShaDes.jl?branch=main)

---

In strong gravitational lensing, one of the fundamental quantity is the arrival time delay surface,
given as [1992grle.book.....S](@cite),
```math
\begin{equation}
    t_d(\pmb{\theta}, \pmb{\beta}) = \frac{1+z_d}{\rm c} \frac{D_d D_s}{D_{ds}} 
    \left[ \frac{1}{2} |\pmb{\theta} - \pmb{\beta}|^2 - \frac{D_{ds}}{D_s}\psi(\pmb{\theta}, \pmb{\beta}) \right],
\end{equation}
```
where $\pmb{\beta}$ is the source position, $\pmb{\theta}$ represents position in the image plane. 
$D_d$, $D_s$, and $D_{ds}$ are the angular diameter distances from observer to lens, observer to
source, and lens to source, respectively. $\psi$ is the projected lensing potential. Very often
(or at least I do it), $D_{ds}/D_s$ is referred to as the *distance ratio*, $a_{\rm dis}$. 

Degeneracies are transformations that leave the observables unchanged. For example, the well known
mass sheet degeneracy (MSD), re-scales the lensing potential,
```math
\begin{equation}
    \psi_\lambda(\theta) = \lambda \psi(\theta) + (1-\lambda) \frac{|\pmb{\theta}|^2}{2},
\end{equation}
```
such that the $t_d$ is scaled by a constant factor $\lambda$. Hence, all time delay and 
magnification are re-scaled by the same factor but leaves the observed image positions and 
relative magnifications unchanged. Similarly, one can concoct other transformations such that 
the observables at hand remains invariant.

Here, we focus on a specific class of degeneracies that occurs in gravitational lensing, namely, 
*shape degeneracies* (**ShaDes**). ShaDes are defined as transformations that leave the observed
image positions unchanged for all sources. In other words, given a best-fit model ($\mathcal{M}$) 
and the observed image positions ($\pmb{\theta}_i$), introduce a perturbation ($\mathcal{P}$) in 
mass distribution such that the image positions remains unchanged.