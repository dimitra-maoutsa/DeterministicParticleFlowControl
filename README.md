[![PyPI version](https://badge.fury.io/py/DeterministicParticleFlowControl.svg)](https://badge.fury.io/py/DeterministicParticleFlowControl)
[![Documentation Status](https://readthedocs.org/projects/deterministicparticleflowcontrol/badge/?version=latest)](https://deterministicparticleflowcontrol.readthedocs.io/en/latest/?badge=latest)
[![PyPI format](https://img.shields.io/pypi/format/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/releases/)

# Deterministic Particle Flow Control

Repository for the **Deterministic Particle Flow Control framework**


Link to extended preprint: [http://arxiv.org/abs/2112.05735](http://arxiv.org/abs/2112.05735)

Link to poster: [poster](https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/blob/main/Poster_Maoutsa_Deterministic_Particle_Flows_Neurips_3scaled.png)

Link to documendation: [readthedocs](http://deterministicparticleflowcontrol.readthedocs.io/)

[ This is work together with Manfred Opper funded by the [SFB 1294](https://www.sfb1294.de/) ]

<p align="center">
<img src="https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/blob/main/waterfall_plot_cmap25b.png" width=50% height=50%>
</p>

## Description:

Computing optimal interventions for stochastic nonlinear systems is a computationally demanding process requiring the solution of nonlinear partial differential equations. Here we build on the Path Integral control formalism to derive a *noniterative* framework that represents the solutions of the underlying partial differential equations in terms of *deterministic* particle flows.   

## Setting:

We consider a stochastic system described by a stochastic differential equation (SDE) 

<img src="https://render.githubusercontent.com/render/math?math=dX_t = f(X_t,t) dt %2B \sigma(x,t) dW_t">

with drift `f(x,t)` and diffusion `σ(x,t)`.

We want to constrain the dynamics of this system for a time interval <img src="https://render.githubusercontent.com/render/math?math=[0,\,T]"> 
- either to reach some target state <img src="https://render.githubusercontent.com/render/math?math=x^*"> at time T   (_**terminal** constraint_), 
- and/or to visit/avoid specific regions of the state space (_**path** constraints_).  

We implement the constraints in terms a time- and state-dependent perturbation of the deterministic part of the dynamics, i.e. we apply interventions `u(x,t)` and the controlled system dynamics become

<img src="https://render.githubusercontent.com/render/math?math=dX_t = f(X_t,t) dt %2B u(x,t) dt %2B \sigma(x,t) dW_t">.

Following the assumptions of the Path Integral control formalism, i.e. assuming that control costs are inversely proportional of noise variance (see [paper](http://arxiv.org/abs/2112.05735) for more details), we can show that the interventions `u(x,t)` can be obtained from the **logarithmic gradient of the solution of a backward partial differential equation** rescaled by the noise variance. 

Here, instead of solving the backward PDE to obtain the optimal drift adjustment that implements the constraints, we express the optimal interventions as the **difference of the logarithmic gradient of two probability flows**, <img src="https://render.githubusercontent.com/render/math?math=\rho_t(x)"> and <img src="https://render.githubusercontent.com/render/math?math=\q_t(x)">. The probability flow or density <img src="https://render.githubusercontent.com/render/math?math=\rho_t(x)"> satisfies the forward filtering equation, a forward PDE that in the absence of path constraints is the Fokker--Planck equation of the uncontrolled dynamics, while <img src="https://render.githubusercontent.com/render/math?math=\q_t(x)"> is the marginal constrained density that in turn satisfies the Fokker--Planck equation of the optimally  controlled dynamics.


## Instalation

To install the install the software run the following commands

```python

# Create and activate a new virtualenvironment
python -m venv venv
source venv/bin/activate
# install the depndencies
pip install -r requirements.txt
pip install -e ./
```


## Main functionality: `DPFC`

To obtain the time dependent control functions create an instance of `DPFC`
```python
from DeterministicParticleFlowControl import DPFC
control_flows = DPFC(t1,t2,y1,y2,f,g,N,M,reweight, U,dens_est,reject,kern='RBF',f_true=None,brown_bridge=False)
```
where
- `t1`, `t2` : are define the timeinterval [t1,t2] within which the constraints are imposed onto the system,
- `y1`, `y2` : are the initial and target state,
-  `f`       : is the drift function of the uncontrolled system,
-  `g`       : is the diffusion coefficient (can be scalar or matrix),
-  `N`       : the number of particles that will be employed for sampling the flows,
-  `M`       : inducing point number for the sparse kernel estimation of the logarithmic gradients,
-  `reweight`: boolean indicating whether reweighting of forward trajectories is necessary, i.e. when:
   - path constraints are relevant for the problem,
   - the target is a non typical system state and the forward flow will be sampled as a reweighted Brownian bridge,
- `U`        : function handle representing the path constraints, i.e. <img src="https://render.githubusercontent.com/render/math?math=U(x,t) = ( x - sin(t) )^2"> , for                      pushing the controlled trajectories towards the sin(t) line. Expects two arguments, so even if path constraint is non timedependent add an extra dummy variable                in the argument list, 
- `dens_est` : idicator determining the method to be employed for the logarithmic gradient (score) estimation. Currently supported:
  - `nonparametric` : for nonparametric sparse kernel estimation, 
- `reject`   : boolean variable determining whether rejection of backward trajectories failing to reach the initial condition will take place 
               (this is mostly used as an indicator of numerical instabilities - if everything runs smoothly 1 or 2 trajectories need to be deleted, but often none of them
               if more trajectories get deleted, then rerun the computation with more particles N)
- `kern`     : the kernel that will be employed for the nonparametric logarithmic gradient estimation of the sampled densities. Currently supported:
   - `RBF` : radial basis function kernel with lengthscale estimated at every time step from the samples,

- `f_true`   : function handler of system drift function when  the reweighted Brownian bridge functionality for forward sampling is used. In that case, `f` should be supplied with the drift function of the brownian bridge, `U` should be set to the necessary path constraint (see [paper](http://arxiv.org/abs/2112.05735)), and `reweight` and `brown_bridge` should be turned to True,
- `brown_bridge`: boolean variable indicating whether the sampling of the forward flow will happen with reweighted Brownian bridge dynamics.


This returns an object that contains the sampled flows and control functions. To compute the controls for the timestep `ti` when the system is at state `x` call
```python
u = control_flows.calculate_u(x,ti)
```






# References

\[1\] Dimitra Maoutsa, Manfred Opper. "Deterministic particle flows for constraining stochastic nonlinear systems". 2021. [[arXiv]](http://arxiv.org/abs/2112.05735)

\[2\] Dimitra Maoutsa, Manfred Opper. "Deterministic particle flows for constraining SDEs". 2021. [[Machine Learning and the Physical Sciences, Workshop at the 35th Conference on Neural Information Processing Systems (NeurIPS)]](https://arxiv.org/pdf/2110.13020)

\[3\] Dimitra Maoutsa, Sebastian Reich, Manfred Opper. "Interacting particle solutions of Fokker–Planck equations through gradient–log–density estimation". 2021. [[Entropy]](https://www.mdpi.com/1099-4300/22/8/802/htm)
