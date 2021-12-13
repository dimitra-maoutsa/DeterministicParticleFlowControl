# Deterministic Particle Flow Control
Repository for Deterministic Particle Flow Control framework


Link to extended preprint: [http://arxiv.org/abs/2112.05735](http://arxiv.org/abs/2112.05735)

[ This is work together with Manfred Opper funded by the [SFB 1294](https://www.sfb1294.de/) ]

<p align="center">
<img src="https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/blob/main/waterfall_plot_cmap25b.png" width=50% height=50%>
</p>


## The main functionality: `DPFC`

To obtain the time dependent control functions create an instance of `DPFC`
```python
from DeterministicParticleFlowControl import DPFC
control_flows = DPFC(t1,t2,y1,y2,f,g,N,M,reweight, U,dens_est='nonparametric',reject=True,plotting=True,kern='RBF',f_true=None,brown_bridge=False)
```
where
- `t1`, `t2` : are define the timeinterval [t1,t2] within which the constraints are imposed onto the system,
- `y1`, `y2` : are the initial and target state,
-  `f`       : is the drift function of the uncontrolled system,
-  `g`       : is the diffusion coefficient (can be scalar or matrix),
-  `N`       : the number of particles that will be employed for sampling the flows,
-  `M`       : inducing point number for the sparse kernel estimation of the logarithmic gradients,
-  `reweight`: boolean indicating whether reweighting of forward trajectories is necessary, i.e. when:
   - path constraints are relevant for the problem
   - the target is a non typical system state and the forward flow will be sampled as a reweighted Brownian bridge

-                
