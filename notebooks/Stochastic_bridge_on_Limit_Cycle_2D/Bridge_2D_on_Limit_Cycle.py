# -*- coding: utf-8 -*-

"""
Stochastic bridge on a Limit Cycle
===================================
This example demonstrates how to call the framework for a multidimensional
system with only terminal constraint.
"""

#%%
# Here we consider a two dimensional system with a limit cycle, and we will
# create a stochastic bridge between two points on the limit cycle, i.e. we will
# impose a terminal constraint onto the dynamics.
#
# For sanity check we simulate a long trajectory of the uncontrolled system
# stored in F. We create an instance of DPFC with proper attributes (i.e. initial
# and terminal state and time, drift and diffusion of the uncontrolled process).


from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import DeterministicParticleFlowControl as dpfc
#from DeterministicParticleFlowControl import DPFC

### ploting parameters
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.rcParams["axes.edgecolor"] = "1.0"
plt.rcParams["axes.linewidth"] = 2


# Drift of the system
def f(x, t=0):#LC
    x0 = -x[1] + x[0]*(1-x[0]**2 -x[1]**2)
    x1 = x[0] + x[1]*(1-x[0]**2 -x[1]**2)
    return np.array([x0, x1])

# noise amplitude
g = 0.1

#simulation_precision
dt = 0.001

t_start = 0.
T = 500.
#%%
# We simulate first a very long trajectory of the uncontrolled system.

x_0 = np.array([-0., -1.0])

timegridall = np.arange(0, T, dt)
F = np.zeros((2, timegridall.size))

for ti, _ in enumerate(timegridall):
    if ti == 0:
        F[:, 0] = x_0
    else:
        F[:, ti] = F[:, ti-1]+ dt* f(F[:, ti-1])+\
            (g)*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(2, ))

#%%
# Set initial and terminal conditions for the bridge and
# create a DPFC object that samples the two probability flows.
#


steps = 4000 #steps between initial and terminal points

# number of particles
N = 200
# number of sparce/inducing points for the sparse kernel logarithmic gradient
# estimations
M = 40
# initial time
t1 = timegridall[100]
# terminal time
t2 = timegridall[100+steps]
# initial state
y1 = F[:, 100]
# terminal state
y2 = F[:, 100+steps]

print('Starting sampling...')
##create object bridg2d that contains the sampled flows
bridg2d = dpfc.DPFC(t1, t2, y1, y2, f, g, N, M, dens_est='nonparametric', deterministic=True)
print('Sampling done!')
#%%
# Plot of the invariant density of the limit cycle as approximated by the
# long simulation, and the sampled backward probability flow (maroon).
# The sampled time-reversed flow already represents the constrained (marginal)
# density.

############## sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_thumbnail_path = '_figs/bridge_on_LC.png'

plt.figure(figsize=(10, 10))
plt.plot(F[0], F[1], '.')
plt.plot(bridg2d.B[0].T, bridg2d.B[1].T, alpha=0.5, c='maroon')
plt.plot(y1[0], y1[1], 'g.', markersize=16)
plt.plot(y2[0], y2[1], '*', c='yellow', markersize=16)
plt.title('Invariant density and time reversed flow', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
ax = plt.gca()
ax.annotate("target", xy=y2, xycoords='data',
            xytext=(y2[0]-0.5, y2[1]+0.3), textcoords='data', size=18,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-.3", color='k', lw=2.5),
            )
plt.show()
#plt.savefig('bridge_with_correct_drift.png')
#plt.figure(),plt.plot(bridg2d.B[0].T,alpha=0.3)

#%%
# .. image:: ../../../bridge_on_LC.png
#    :scale: 70%
#    :align: center
#    :alt: Some Text

#%%
# Plot of the sampled constrained flow across each dimension.

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(bridg2d.timegrid, bridg2d.B[0, :, :].T, 'maroon', alpha=0.5)
plt.plot(bridg2d.timegrid[-1], y2[0], '*', c='yellow', markersize=10)
plt.plot(bridg2d.timegrid[0], y1[0], '.g')
plt.xlabel('time')
plt.ylabel('x')
#plt.ylim(-2,2)

plt.subplot(1, 2, 2)
plt.plot(bridg2d.timegrid, bridg2d.B[1].T, 'maroon', alpha=0.5)
plt.plot(bridg2d.timegrid[-1], y2[1], '*', c='yellow', markersize=10)
plt.plot(bridg2d.timegrid[0], y1[1], '.g')
plt.xlabel('time')
plt.ylabel('y')
plt.suptitle('Zoomed in each dimension seperately')


#%%
# Set the controls for the heart of the star.
# Simulate an ensemble of controlled and an ensemble of uncontrolled
# trajectories.

dim = 2
reps = 30
### storage for controlled trajectories
Fcont = np.zeros((dim, bridg2d.timegrid.size-1, reps))
### storagefor uncontrolled trajectories
Fnon =  np.zeros((dim, bridg2d.timegrid.size-1, reps))
### storage for controls
used_u =  np.zeros((dim, bridg2d.timegrid.size, reps))
for ti, tt in enumerate(bridg2d.timegrid[:-1]):


    if ti == 0:
        Fcont[:,ti] = y1.reshape(dim, -1)
        Fnon[:,ti] = y1.reshape(dim, -1)

    else:
        
        uu = bridg2d.calculate_u(np.atleast_2d(Fcont[:, ti-1]).T, ti)



        used_u[:, ti] = uu

        Fcont[:, ti] =  (Fcont[:, ti-1]+ dt* f(Fcont[:, ti-1])+dt*g**2 *uu+\
                        (g)*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(dim, reps)))
        Fnon[:, ti] =  (Fnon[:, ti-1]+ dt* f(Fnon[:, ti-1])+\
                       (g)*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(dim, reps)))

#%%
#

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(F[0], F[1], '.')
plt.plot(bridg2d.B[0].T, bridg2d.B[1].T, alpha=0.5, c='maroon')
ax = plt.gca()
orange = next(ax._get_lines.prop_cycler)['color']
plt.plot(Fcont[0], Fcont[1], c=orange, alpha=0.8)
plt.plot(y1[0], y1[1], 'g.', markersize=16)
plt.plot(y2[0], y2[1], '*', c='yellow', markersize=16)
plt.title('Controlled trajectories', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)

plt.subplot(1, 2, 2)
plt.plot(F[0], F[1], '.')
plt.plot(bridg2d.B[0].T, bridg2d.B[1].T, alpha=0.5, c='maroon')
plt.plot(Fnon[0], Fnon[1], c='silver', alpha=0.8)
plt.plot(y1[0], y1[1], 'g.', markersize=16)
plt.plot(y2[0], y2[1], '*', c='yellow', markersize=16)
plt.title('Uncontrolled trajectories', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
#plt.savefig('Controlled_and_uncontrolled_LC.png')
plt.show()