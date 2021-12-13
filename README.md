# Deterministic Particle Flow Control
Repository for the Deterministic Particle Flow Control framework


Link to extended preprint: [http://arxiv.org/abs/2112.05735](http://arxiv.org/abs/2112.05735)

[ This is work together with Manfred Opper funded by the [SFB 1294](https://www.sfb1294.de/) ]

<p align="center">
<img src="https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/blob/main/waterfall_plot_cmap25b.png" width=50% height=50%>
</p>


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
