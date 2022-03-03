# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 01:13:00 2022

@author: maout
"""



import numpy as np
from matplotlib import pyplot as plt
#import seaborn as sns
from scipy.integrate import odeint
from odeintw import odeintw
#import joblib
#save_file='C:/Users/maout/Data_Assimilation_stuff/codes/results_otto/'
from scipy.linalg import sqrtm
## covariance matrix
K = np.zeros((2, 2))
kx = 0.2
ky = 0.5
K[0, 0] = kx
K[1, 1] = ky
###correlation between the two traits
sigma1 = np.sqrt(kx)
sigma2 = np.sqrt(ky)
rho_xy = 0.25
kxy = rho_xy* sigma1 *sigma2
K[0, 1] = kxy
K[1, 0] = kxy
K0 = np.diag(np.diag(K))
##standard deviations of each variable 
#Dinv = np.diag(1 / np.sqrt(np.diag(K))) 
#R = Dinv @ K @ Dinv

sigma = sqrtm(K)  ##noise
sigma0 = sqrtm(K0)  ##noise
x0 = np.array([0,0]) #1
y1 = x0
y2 = np.array([0.5, 0])


C = np.eye(2)
C[0, 0] = 2 #
C[1, 1] = 4
def f(x,t=0):  
    return -2*K @ C@x

def f0(x,t=0): 
    return -2*K0 @ C@x
    

h = 0.001
t1=0
t2 = 1
T = t2-t1
timegrid = np.arange(0,T+h/2,h) 

g= sigma
g0 = sigma0


def f_var(C,t):    
    A = -2*K @ C    
    return A@C + C@A.T + sigma@sigma.T   #this should have been be A@C + C@A.T + A@np.eye(2,2)*sigma**2

def f_var0(C,t):    
    A = -2*K0 @ C    
    return A@C + C@A.T + sigma0@sigma.T

C_init = np.array([[0.0**2,0],[0,0.0**2]])

m_t = odeint(f, x0, timegrid)
C_t = odeintw(f_var, C_init,timegrid)

m_t0 = odeint(f0, x0, timegrid)
C_t0 = odeintw(f_var0, C_init,timegrid)


m_tb = odeint(f, y2, timegrid)
C_tb = odeintw(f_var, C_init,timegrid)

m_t0b = odeint(f0, y2, timegrid)
C_t0b = odeintw(f_var0, C_init,timegrid)

#%%
def grad_log_p_Gaussold(x,ti, m_t, C_t): ##pass all times
    return - np.linalg.inv( (C_t[ti,:,:]).T ) @ ( np.atleast_2d(x) - m_t[ti,:] ).T 


def grad_log_p_Gauss(x, m_t, C_t): ##pass only a single time array
    #print(np.linalg.inv( (C_t[:,:]).T ).shape)
    #print( ( x - np.atleast_2d(m_t).T ).shape)
    return   -np.linalg.inv( (C_t[:,:]).T + 1e-6 * np.eye(C_t.shape[0]))  @ ( x - np.atleast_2d(m_t).T )


grad_log_q = lambda x, ti: grad_log_p_Gauss(x, m_tb[timegrid.size-ti-1], C_tb[timegrid.size-ti-1])
grad_log_p = lambda x, ti: grad_log_p_Gauss(x, m_t[ti], C_t[ti])

grad_log_q0 = lambda x, ti: grad_log_p_Gauss(x, m_t0b[timegrid.size-ti-1], C_t0b[timegrid.size-ti-1])
grad_log_p0 = lambda x, ti: grad_log_p_Gauss(x, m_t0[ti], C_t0[ti])


u = lambda x, ti: sigma@sigma.T@(grad_log_q(x,ti) - grad_log_p(x,ti))


u0 = lambda x, ti: sigma0@sigma0.T@(grad_log_q0(x,ti) - grad_log_p0(x,ti))

#%%
reps = 1000
dim = 2
AFcont = np.zeros((2,reps, timegrid.size))
AFcont0 = np.zeros((2,reps, timegrid.size))
for ti,tt in enumerate(timegrid[:]):
    if ti==0:
        AFcont[0,:,ti] = y1[0] 
        AFcont0[0,:,ti] = y1[0] 
        
        
        AFcont[1,:,ti] = y1[1] 
        AFcont0[1,:,ti] = y1[1] 
    else:
        AFcont0[:,:,ti] =  ( AFcont0[:,:,ti-1]+ h* f(AFcont0[:,:,ti-1])+h*u0(AFcont0[:,:,ti-1], ti-1)+(g)@np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,reps)) )
        AFcont[:,:,ti] =   AFcont[:,:,ti-1]+ h* f(AFcont[:,:,ti-1])+h* u(AFcont[:,:,ti-1], ti-1)+(g)@np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,reps)) 

#%%
        
plt.figure()
#plt.figure(figsize=(5,8)),
plt.subplot(2,1,1),

plt.plot(timegrid[:-1],AFcont[0,:,:-1].T,'-')
plt.subplot(2,1,2),

plt.plot(timegrid[:-1],AFcont[1,:,:-1].T,'-')


#%%
        
plt.figure()
#plt.figure(figsize=(5,8)),
plt.subplot(2,1,1),

plt.plot(timegrid[:-1],AFcont0[0,:,:-1].T,'-')
plt.subplot(2,1,2),

plt.plot(timegrid[:-1],AFcont0[1,:,:-1].T,'-')
   
#%%        
plt.figure()
#plt.figure(figsize=(5,8)),
plt.subplot(2,1,1),

plt.plot(timegrid[:-1],np.mean(AFcont[0,:,:-1],axis=0),'-')
plt.plot(timegrid[:-1],np.mean(AFcont0[0,:,:-1],axis=0),'--')
plt.title(r'$\rho_{xy} = %.2f$'%rho_xy)
plt.subplot(2,1,2),

plt.plot(timegrid[:-1],np.mean(AFcont[1,:,:-1],axis=0),'-')
plt.plot(timegrid[:-1],np.mean(AFcont0[1,:,:-1],axis=0),'--')



#%%


#%%        
plt.figure()
#plt.figure(figsize=(5,8)),
#plt.subplot(2,1,1),

plt.plot(np.mean(AFcont[0,:,:-1],axis=0),np.mean(AFcont[1,:,:-1],axis=0),'-')
#plt.plot(np.mean(AFcont0[0,:,:-1],axis=0),np.mean(AFcont0[0,:,:-1],axis=0),'--')
#%%

plt.figure(),
plt.subplot(2,1,1)
plt.plot(timegrid, m_t[:,0],'k')
plt.plot(timegrid, m_t[:,0] + np.sqrt( C_t[:,0,0]) ,'r--')
plt.plot(timegrid, m_t[:,0] - np.sqrt( C_t[:,0,0]) ,'r--')

plt.plot(timegrid, m_t0[:,0] + np.sqrt( C_t0[:,0,0]) ,'g--')
plt.plot(timegrid, m_t0[:,0] - np.sqrt( C_t0[:,0,0]) ,'g--')

plt.subplot(2,1,2)
plt.plot(timegrid, m_tb[:,1],'k')
plt.plot(timegrid, m_tb[:,1] + np.sqrt( C_tb[:,1,1]),'r--' )
plt.plot(timegrid, m_tb[:,1] - np.sqrt( C_tb[:,1,1]),'r--' )


#%%

# n_sampls = 5000#2000

# AF = np.zeros((2,n_sampls,timegrid.size))
# for ti,t in enumerate(timegrid):
#     # Define epsilon.
#     epsilon = 0.0001
#     # Add small pertturbation. 
#     K = C_t[ti] + epsilon*np.identity(2)  
#     AF[:,:,ti] = np.random.multivariate_normal(mean=m_t[ti].reshape(2,), cov=K, size=n_sampls).T
    
#joblib.dump(AF,filename=save_file+'OU_2D_samples_from_analytic_trajectories_fortiming_N_%d'%(5000)) 
