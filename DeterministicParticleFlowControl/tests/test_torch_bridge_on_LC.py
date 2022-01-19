# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 02:20:34 2022

@author: maout
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
from DeterministicParticleFlowControl import torched_DPFC
#import DeterministicParticleFlowControl as dpfc
from utils.utils_pytorch import set_device

###Limit cycle function and analytic gradient for passing for comparison calculations
def f(x,t=0):#LC

    x0 = -x[1] + x[0]*(1-x[0]**2 -x[1]**2)
    x1 = x[0] + x[1]*(1-x[0]**2 -x[1]**2)
    
    return torch.cat((x0.view(1, -1) ,x1.view(1, -1) ), dim=0)

def f_numpy(x,t=0):#LC
    x0 = -x[1] + x[0]*(1-x[0]**2 -x[1]**2)
    x1 = x[0] + x[1]*(1-x[0]**2 -x[1]**2)
    return np.array([x0,x1])

def glnfss(x,sigma):
    x0 = - x[0]*(x[0]**2 + x[1]**2 - 1)/(0.5*sigma**2)
    x1 = - x[1]*(x[0]**2 + x[1]**2 - 1)/(0.5*sigma**2)
    return np.array([x0,x1])

DEVICE = set_device()
#simulation_precision
dt = 0.001

t_start = 0.
T = 50#0.
#x0 = np.array([1.81, -1.41])
x0 = torch.tensor([-0., -1.0], dtype=torch.float64, device=DEVICE )

timegridall = np.arange(0,T,dt)
F = np.zeros((2,timegridall.size))
#noise amplitude
g = 0.1    
for ti,t in enumerate(timegridall):
    if ti==0:
        F[:,0] = x0.cpu()
    else:
        F[:,ti] = F[:,ti-1]+ dt* f_numpy(F[:,ti-1])+(g)*np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(2,))  

steps = 500 #steps between initial and terminal points
obs_dens = steps
N = 200
M = 40
t1 = timegridall[100]
t2 = timegridall[100+steps]
y1 = torch.tensor(F[:,100], dtype=torch.float64, device=DEVICE)
y2 = torch.tensor(F[:,100+steps], dtype=torch.float64, device=DEVICE)
    
    
##create object bridg2d that contains the simulated flows
bridg2d = torched_DPFC(t1,t2,y1,y2,f,g,N,M,dens_est='nonparametric', deterministic=True, device=DEVICE)


plt.figure(figsize=(10,10)),
plt.plot(F[0],F[1],'.', alpha=0.05);
if DEVICE=='cpu':
    #plt.plot(bridg2d.Z[0].detach().numpy().T,bridg2d.Z[1].detach().numpy().T,alpha=0.5,c='grey');
    plt.plot(bridg2d.B[0].detach().numpy().T,bridg2d.B[1].detach().numpy().T,alpha=0.5,c='grey');
    plt.plot(y1[0].detach().numpy(),y1[1].detach().numpy(),'g.',markersize=16);
    plt.plot(y2[0].detach().numpy(),y2[1].detach().numpy(),'d',c='maroon',markersize=16);
    plt.xlim(-0.5,1.5)
    plt.ylim(-1.5,0)
else:
    plt.plot(bridg2d.B[0].cpu().detach().numpy().T,bridg2d.B[1].cpu().detach().numpy().T,alpha=0.5,c='grey');
    plt.plot(y1[0].cpu().detach().numpy(),y1[1].cpu().detach().numpy(),'g.',markersize=16);
    plt.plot(y2[0].cpu().detach().numpy(),y2[1].cpu().detach().numpy(),'d',c='maroon',markersize=16);
plt.title('Invariant density of the limit cycle and backwad flow');