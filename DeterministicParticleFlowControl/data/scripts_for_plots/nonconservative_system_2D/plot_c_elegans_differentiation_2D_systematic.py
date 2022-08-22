# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:45:34 2021

@author: maout
"""


#@title score function multidimensional

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import pinv
from functools import reduce
from scipy.stats import gamma,norm,dweibull,tukeylambda,skewnorm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import time
### calculate score function from empirical distribution
### uses RBF kernel
### follows description of [Batz, Ruttor, Opper, 2016]

#Ktestsp = pdist2(xtrain',xsparse');
#Ktestsp= Ktestsp.^2/L^2;
#Ktestsp = exp(-Ktestsp);

def score_function_multid_seperate(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
    
    """
    returns function psi(z)
    Input: X: N observations
           Z: sparse points
           func_out : Boolean, True returns function if False return grad-log-p on data points                    
           l: lengthscale of rbf kernel
           C: weighting constant           
           which: return 1: grad log p(x) 
           which_dim: which gradient of log density we want to compute (starts from 1 for the 0-th dimension)
    Output: psi: array with density along the given dimension N or N_s x 1
    
    """
    if kern=='RBF':
        #l = 1 # lengthscale of RBF kernel
        
        def K(x,y,l,multil=False):
            if multil:                
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
            #return np.exp(-(x-y.T)**2/(2*l*l))
            #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape            
            diffs = x[:,None]-y   
            #print(diffs.shape)
            redifs = np.zeros((1*N,N))
            ii = which_dim -1
            #print(ii)
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
            #return -(1./(l*l))*(x-y.T)*K(x,y)
     
        def grdy_K(x,y): # gradient with respect to the second argument
            N,dim = x.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            ii = which_dim -1              
            redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
            return -redifs
            #return (1./(l*l))*(x-y.T)*K(x,y)
                
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs
            #return np.multiply((K(x,y)),(np.power(x[:,None]-y,2)-l**2))/l**4
     
    if isinstance(l, (list, tuple, np.ndarray)):
       ### for different lengthscales for each dimension 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       multil = True
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True) 
       # if not(Test_p == 'None'):
       #     K_sz = K(Test_p,Z,l,multil=True)
        
    else:
        multil = False
        K_xz = K(X,Z,l,multil=False) 
        Ks = K(Z,Z,l,multil=False)    
        
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=False)
    sumgradx_K = np.sum(gradx_K ,axis=0)
    if func_out==False: #if output wanted is evaluation at data points
        ### evaluatiion at data points
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:           
        #### for function output 
        if multil:                
            #res = np.ones((x.shape[0],y.shape[0]))                
            #for ii in range(len(l)): 
            K_sz = lambda x: np.multiply(np.exp(-cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1),'sqeuclidean')/(2*l[0]*l[0])),np.exp(-cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(2*l[1]*l[1])))
            #return K_sz
        else:
            K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            #return K_sz

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K


    
    return res1


#%%title Bridge Multi D

from matplotlib import pyplot as plt
from copy import deepcopy
import seaborn as sns
from scipy.stats import skew,kurtosis
import numpy as np

### ploting parameters
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.rcParams["axes.edgecolor"] = "1.0"
plt.rcParams["axes.linewidth"]  = 2  




class BRIDGE_ND_reweight:
    def __init__(self,t1,t2,y1,y2,f,g,N,k,M,reweight=False, U=1,dens_est='nonparametric',reject=True,plotting=True,kern='RBF'):
        """
        Bridge initialising function
        t1: starting time point
        t2: end time point
        y1: initial observation/position
        y2: end observation/position
        f: drift function handler
        g: diffusion coefficient or function handler 
        N: number of particles/trajectories
        k: discretisation steps within bridge 
        M: number of sparse points for grad log density estimation
        reweight: boolean - determines if reweighting will follow
        U: function, reweighting function to be employed during reweighting: dim_y1 \to 1
        dens_est: density estimation function
                  > 'nonparametric' : non parametric density estimation
                  > 'hermit1' : parametic density estimation empoying hermite polynomials (physiscist's)
                  > 'hermit2' : parametic density estimation empoying hermite polynomials (probabilists's)
                  > 'poly' : parametic density estimation empoying simple polynomials
                  > 'rbf' : parametric density estimation employing radial basis functions
        kern: type of kernel: 'RBF' or 'periodic'
        reject: boolean parameter indicating whether non valid bridge trajectories will be rejected
        plotting: boolean parameter indicating whether bridge statistics will be plotted
        
        """
        self.dim = y1.size # dimensionality of the problem
        self.t1 = t1
        self.t2 = t2
        self.y1 = y1
        self.y2 = y2

        
        ##density estimation stuff
        self.kern = kern
        # DRIFT /DIFFUSION
        self.f = f
        self.g = g #scalar or array
        
        ### PARTICLES DISCRETISATION
        self.N = N        
        
        self.N_sparse = M
        
        self.dt = 0.001 #((t2-t1)/k)
        
        ### reweighting
        self.reweight = reweight
        if self.reweight:
          self.U = U

        ### reject
        self.reject = reject

        
        self.finer = 1#200 #discetasation ratio between numerical BW solution and particle bridge solution
        self.timegrid = np.arange(self.t1,self.t2+self.dt/2,self.dt)
        self.k = self.timegrid.size
        #self.timegrid_fine = np.arange(self.t1, self.t2+self.dt*(1./self.finer)/2, self.dt*(1./self.finer) )
        
        # print(self.k == self.timegrid.size)
        # print(self.timegrid)
        
        self.Z = np.zeros((self.dim,self.N,self.k)) #storage for forward trajectories
        self.B = np.zeros((self.dim,self.N,self.k)) #storage for backward trajectories
        self.ln_roD = [] 
        self.BPWE = np.zeros((self.dim,self.N,self.timegrid.size))
        self.BPWEmean = np.zeros((self.dim,self.k*self.finer))
        self.BPWEstd = np.zeros((self.dim,self.k*self.finer))
        self.BPWEskew = np.zeros((self.dim,self.k*self.finer))
        self.BPWEkurt = np.zeros((self.dim,self.k*self.finer))
        
        
        #self.forward_sampling()
        self.forward_sampling_Otto()
        # plt.figure(figsize=(6,4)),plt.plot(self.Z[0].T,self.Z[1].T,alpha=0.3);
        # plt.plot(self.y1[0],self.y1[1],'go')
        # plt.plot(self.y2[0],self.y2[1],'ro')
        # plt.show()

           
        #self.density_estimation()
        self.backward_simulation()
        self.reject_trajectories() 
        #self.calculate_true_statistics()
        #if plotting:
        #    self.plot_statistics()
        
    def forward_sampling(self): 
        print('Sampling forward...')
        W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):

            if ti == 0:
                self.Z[0,:,0] = self.y1[0]
                self.Z[1,:,0] = self.y1[1]
            else:
                for i in range(self.N):
                    #self.Z[:,i,:] = sdeint.itoint(self.f, self.g, self.Z[i,0], self.timegrid)[:,0] 
                    self.Z[:,i,ti] = ( self.Z[:,i,ti-1] + self.dt* self.f(self.Z[:,i,ti-1]) + \
                                      (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,)) )
        
                ###WEIGHT
                if self.reweight == True:
                  if ti>0:
                      W[:,0] = np.exp(1*self.dt*self.U(self.Z[:,:,ti]))                    
                      W = W/np.sum(W)
                      
                      ###REWEIGHT                    
                      Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                      #P = Tstar *N
                      # print(Tstar.shape)
                      # print(X.shape)
                      self.Z[:,:,ti] = (  (self.Z[:,:,ti])@Tstar  )
                
        for di in range(self.dim):
          self.Z[di,:,-1] = self.y2[di]
        print('Forward sampling done!')
        return 0
    
    ### effective forward drift - estimated seperatelly for each dimension
    def f_seperate(self,x,t):#plain GP prior
        
        dimi, N = x.shape        
        bnds = np.zeros((dimi,2))
        for ii in range(dimi):
            bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
        sum_bnds = np.sum(bnds)
        if np.isnan(sum_bnds) or np.isinf(sum_bnds):
          plt.figure(figsize=(6,4)),plt.plot(self.Z[0].T,self.Z[1].T,alpha=0.3);
          plt.show()

        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )        
        gpsi = np.zeros((dimi, N))
        lnthsc = 2*np.std(x,axis=1)    
           
        for ii in range(dimi):            
            gpsi[ii,:]= score_function_multid_seperate(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc,which_dim=ii+1, kern=self.kern)     
        
        return (self.f(x,t)-0.5* self.g**2* gpsi)
    
    
    def forward_sampling_Otto(self):
        print('Sampling forward with deterministic particles...')
        W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):  
            #print(ti)          
            if ti == 0:
                for di in range(self.dim):
                    self.Z[di,:,0] = self.y1[di]
                    self.Z[di,:,-1] = self.y2[di]   
                    #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
            elif ti==1: #propagate one step with stochastic to avoid the delta function
                #for i in range(self.N):                            #substract dt because I want the time at t-1
                self.Z[:,:,ti] = (self.Z[:,:,ti-1] + self.dt*self.f(self.Z[:,:,ti-1],tt-self.dt)+\
                                 (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
            else:
                
                self.Z[:,:,ti] = ( self.Z[:,:,ti-1] + self.dt* self.f_seperate(self.Z[:,:,ti-1],tt-self.dt) )
                
                ###WEIGHT
            if self.reweight == True:
              if ti>0:
                  W[:,0] = np.exp(self.U(self.Z[:,:,ti])*self.dt) #-1 
                  
                  W = W/np.sum(W)       
                  
                  ###REWEIGHT    
                  start = time.time()
                  Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                  #print(Tstar)
                  if ti ==3:
                      stop = time.time()
                      print('Timepoint: %d needed '%ti, stop-start)
                  
                  self.Z[:,:,ti] = ((self.Z[:,:,ti])@Tstar ) ##### 
        
        print('Forward sampling with Otto is ready!')
        
        return 0
    
    def density_estimation(self, ti,rev_ti):
        rev_t = rev_ti-1#########################################################-1
        grad_ln_ro = np.zeros((self.dim,self.N))
        lnthsc = 2*np.std(self.Z[:,:,rev_t],axis=1)
        
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,rev_t]),np.min(self.B[ii,:,rev_ti])),min(np.max(self.Z[ii,:,rev_t]),np.max(self.B[ii,:,rev_ti]))]
        sum_bnds = np.sum(bnds)
        
        if np.isnan(sum_bnds) or np.isinf(sum_bnds):
          
          plt.figure(figsize=(6,4)),plt.plot(self.B[0].T,self.B[1].T,alpha=0.3);
          plt.plot(self.y1[0],self.y1[1],'go')
          
          plt.show()
        #sparse points
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
        
        for di in range(self.dim):     
            #estimate density from forward (Z) and evaluate at current postitions of backward particles (B)       
            grad_ln_ro[di,:] = score_function_multid_seperate(self.Z[:,:,rev_t].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(self.B[:,:,rev_ti].T)
                     
        
        return grad_ln_ro 


    def bw_density_estimation(self, ti, rev_ti):
        grad_ln_b = np.zeros((self.dim,self.N))
        lnthsc = 2*np.std(self.B[:,:,rev_ti],axis=1)
        #print(ti, rev_ti, rev_ti-1)
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,rev_ti-1]),np.min(self.B[ii,:,rev_ti])),min(np.max(self.Z[ii,:,rev_ti-1]),np.max(self.B[ii,:,rev_ti]))]
        #sparse points
        #print(bnds)
        sum_bnds = np.sum(bnds)
        
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
        
        for di in range(self.dim):            
            grad_ln_b[di,:] = score_function_multid_seperate(self.B[:,:,rev_ti].T,Sxx.T,func_out= False,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)#(self.B[:,:,-ti].T)
        
        return grad_ln_b # this should be function
    
    
    def backward_simulation(self):   
        
        for ti,tt in enumerate(self.timegrid[:-1]): 
            W = np.ones((N,1))/N           
            if ti==0:
                for di in range(self.dim):
                    self.B[di,:,-1] = self.y2[di]                
            else:
                
                Ti = self.timegrid.size
                rev_ti = Ti- ti     
                
                grad_ln_ro = self.density_estimation(ti,rev_ti) #density estimation of forward particles  
                
                if ti==1: 
                  print(rev_ti,rev_ti-1)
                  self.B[:,:,rev_ti-1] = (self.B[:,:,rev_ti] - self.f(self.B[:,:,rev_ti], self.timegrid[rev_ti])*self.dt + self.dt*self.g**2*grad_ln_ro \
                                         + (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
                else:
                  grad_ln_b = self.bw_density_estimation(ti,rev_ti)
                  self.B[:,:,rev_ti-1] = (self.B[:,:,rev_ti] -\
                                        ( self.f(self.B[:,:,rev_ti], self.timegrid[rev_ti])- self.g**2*grad_ln_ro +0.5*self.g**2 * grad_ln_b )*self.dt)
                
        for di in range(self.dim):
            self.B[di,:,0] = self.y1[di]
            
        return 0 



    def reject_trajectories(self):
      fplus = self.y1+self.f(self.y1,self.t1)*self.dt+4*self.g**2 *np.sqrt(self.dt)
      fminus = self.y1+self.f(self.y1,self.t1) *self.dt-4*self.g**2 *np.sqrt(self.dt)
      for iii in range(2):
        if fplus[iii] < fminus[iii]:
          temp = fminus[iii]
          fminus[iii] = fplus[iii]
          fplus[iii] = temp

      sinx = np.where( np.logical_or(np.logical_not(np.logical_and( self.B[0,:,1]<fplus[0],self.B[0,:,1]>fminus[0])) , np.logical_not( np.logical_and(self.B[0,:,1]<fplus[0],self.B[0,:,1]>fminus[0])) ) )[0]
                           #((self.B[1,:,-2]<fplus[1]))  ) & ( & (self.B[1,:,-2]>fminus[1]) )  ))[0]
      print(sinx)
      temp = len(sinx)
      print("Identified %d invalid bridge trajectories "%len(sinx))
      if self.reject:
          print("Deleting invalid trajectories...")
          sinx = sinx[::-1]
          for element in sinx:
              self.B = np.delete(self.B, element, axis=1)
      return 0

    def calculate_u(self,grid_x,ti):
        """
        

        Parameters
        ----------
        grid_x : array of size d x number of points on the grid
        ti     : time index in timegrid for the computation of u
            Computes the control u on the grid or on a the point .
        

        Returns
        -------
        The control u(grid_x, t), where t=timegrid[ti].

        """
        #a = 0.001
        #grad_dirac = lambda x,di: - 2*(x[di] -self.y2[di])*np.exp(- (1/a**2)* (x[0]- self.y2[0])**2)/(a**3 *np.sqrt(np.pi))                 
        u_t = np.zeros(grid_x.shape)
        lnthsc = 2*np.std(self.B[:,:,ti],axis=1)
  
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
     
        #sum_bnds = np.sum(bnds)
      
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
      
        for di in range(self.dim):  
            u_t[di] = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T) \
                     - score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T)
      
        return u_t
    
    
    def check_if_covered(self, X, ti):
        """
        Checks if test point X falls within forward and backward densities at timepoint timegrid[ti]

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        ti : TYPE
            DESCRIPTION.

        Returns
        -------
        Boolean variable - True if the text point X falls within the densities.

        """
        covered = True
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
            #bnds[ii] = [np.min(self.B[ii,:,ti]),np.max(self.B[ii,:,ti])]
        
            covered = covered * ( (X[ii] >= bnds[ii][0]) and (X[ii] <= bnds[ii][1]) )
            
        return covered    

#%%
        

#figure and plotting settings
import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = (1,1,1,0)
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.size'] = 10*1.2
# plt.rcParams['font.family'] = 'sans-serif'     # not available in Colab
# plt.rcParams['font.sans-serif'] = 'Helvetica'  # not available in Colab
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
#plt.rcParams['axes.Axes.tick_params'] = True
#plt.rcParams['axes.solid_capstyle'] = 'round'
plt.rc('axes',edgecolor='#4f4949')
plt.rcParams['figure.frameon'] = False
plt.rcParams['figure.subplot.hspace'] = 0.51
plt.rcParams['figure.subplot.wspace'] = 0.51
plt.rcParams['figure.subplot.left'] = 0.1
plt.rcParams['figure.subplot.right'] = 0.9
plt.rcParams['figure.subplot.top'] = 0.9
plt.rcParams['figure.subplot.bottom'] = 0.1
plt.rcParams['lines.solid_capstyle'] = 'round'
plt.rcParams['lines.solid_joinstyle'] = 'round'
#plt.rcParams['xtick.major.size'] = 20
#plt.rcParams['xtick.major.width'] = 4
plt.rcParams['text.usetex'] = True
#%%
        
import numpy as np
import matplotlib.pyplot as plt

# def f(x,mu=1):#Van der Pol oscillator
#     x0 = mu*(x[0] - (1/3)*x[0]**3-x[1])
#     x1 = (1/mu)*x[0]
#     return np.array([x0,x1])

# def f(x):
#   x0 = -2*x[0]*x[0]*x[0] +6*x[0]- x[1]
#   x1 = -2*x[1]*x[1]*x[1] +6*x[1]- x[0]
#   return np.array([x0,x1])


p1 = 0.5
p2= 0.5
p3 = 0.5
p4 = 0.5

def f(x,t=0):  
  x0 =  ( x[0]**4/ (p1**4 + x[0]**4)) + ( p2**4  / (p2**4 + x[1]**4))- x[0]
  x1 =  ( x[1]**4/ (p3**4 + x[1]**4)) + ( p2**4  / (p4**4 + x[0]**4))- x[1]
  return np.array([x0,x1])

def divf(x):
  dFx_dx = -4*x[0]**7/( p1**4 + x[0]**4 ) + 4* x[0]**3 / ( p1**4 + x[0]**4) -1
  dFy_dy = -4*x[1]**7/( p3**4 + x[1]**4 ) + 4* x[1]**3 / ( p3**4 + x[1]**4) -1
  return dFx_dx + dFy_dy

def gradfs(x):
  dFx_dx = -4*x[0]**7/( p1**4 + x[0]**4 ) + 4* x[0]**3 / ( p1**4 + x[0]**4) -1
  dFx_dy = -4*p2**4*x[1]**3 / (p2**4 + x[1]**4)**2
  dFy_dy = -4*x[1]**7/( p3**4 + x[1]**4 ) + 4* x[1]**3 / ( p3**4 + x[1]**4) -1
  dFy_dx = -4*p4**4*x[0]**3 / (p4**4 + x[0]**4)**2
  return [dFx_dx , dFx_dy, dFy_dx, dFy_dy ]

#def Lnr(xo,x):
    ##computes the linearisation of f around point xo

  

fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

h = 0.001 #sim_prec
t_start = 0.

#x0 = np.array([1.81, -1.41])
x0 = fixed_points[:,0]#np.array([2., 0])
y1 = x0
y2 = fixed_points[:,2]

def Uc(x):
      return 0.5*(1/sigma**2)*( (f(x).T@f(x))+sigma**2*divf(x))

t1=0
t2 = 0.5#0.35
T = t2-t1
timegrid = np.arange(0,T+h,h)
N = 400
g = 1
sigma = g
k = timegrid.size+1
M = 20
gs = [0.75, 1, 1.25,1.5]
reps = 1000
################################################
##computing evolution of grad log density for linearized drift around y2
import odeintw
aa,bb,cc,dd = gradfs(y2)
def flin(x):
    A1 = np.array( [[aa,bb], [cc,dd]])    
    return A1@x + f(y2) - A1@y2
#xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=N) 
def flin_var(C,t):    
    A  = np.array( [[aa,bb], [cc,dd]])    
    return A@C + C@A.T + 1*np.eye(2,2)


C_0 = np.array([[0,0],[0,0.0]])
m_t = np.zeros((2,6))

small_timegrid = np.array([0.00, 0.001, 0.002, 0.003, 0.004, 0.005])
for ti in range(6):
    if ti==0:
        m_t[:,0] = y2
    else:
        m_t[:,ti] = m_t[:,ti-1] + flin( m_t[:,ti-1]) *0.001

##has dimensions (timepoints, 2,2)        
C_t = odeintw.odeintw(flin_var, C_0,small_timegrid)
def grad_log_p_Gauss(x,to):
    return  -0.5* ( np.linalg.inv(C_t[to,:, :] ) + np.linalg.inv( (C_t[to,:,:]).T ) )@ ( x.T - m_t[:,to] ).T 


# xs = np.linspace(0,2,50)
# xs = np.array([xs,xs])

# for ti in range(1,6):
#     plt.plot(xs[1],grad_log_p_Gauss(xs,ti)[1])

###############################################  

#timegrid2 = np.arange(0,2000,h)


def f1(x,t):
    if t==T:
        return np.array([0,0])
    else:
        return np.array([(y2[0]-x[0])/(T-t),  (y2[1]-x[1])/(T-t)  ]) 

#%% simulate deterministic particle control
FcontALL = np.zeros((len(gs),2,reps,timegrid.size))
FnonALL = np.zeros((len(gs),2,reps,timegrid.size))
UcontALL = np.zeros((len(gs),2,reps,timegrid.size))
for gi,g in enumerate(gs):

    bridg2d = BRIDGE_ND_reweight(t1,t2,y1,y2,f,g,N,k,M,dens_est='nonparametric',reject=True,plotting=True,reweight=False, U=0)
    print('simulate controlled for g:',g)
    
    
    dim = 2
    
    for repi in range(reps):
        Fcont = np.zeros((2,bridg2d.timegrid.size))
        Fnon =  np.zeros((2,bridg2d.timegrid.size))
        used_u =  np.zeros((2,bridg2d.timegrid.size))
        for ti,tt in enumerate(bridg2d.timegrid[:]):
            ### ti is local time, tti is global time - both are time indices
              ## index of timepoint in the initial timegrid- the real time axis
            
            if ti==0:
                Fcont[:,ti] = y1    
                Fnon[:,ti] = y1      
            # elif ti >= bridg2d.timegrid.size-5:
            #     uu = -grad_log_p_Gauss(np.atleast_2d(Fcont[:,ti-1]),bridg2d.timegrid.size-ti)
            else:        
                ###use previous grad log for current step
                if False:#ti>= bridg2d.timegrid.size-2:
                    uu = bridg2d.calculate_u(np.atleast_2d(Fcont[:,-2]).T,ti-1)
                elif ti>= bridg2d.timegrid.size-4:
                    uu = grad_log_p_Gauss( np.atleast_2d(Fcont[:,ti-1]).T, bridg2d.timegrid.size -ti)
                else:
                    uu = bridg2d.calculate_u(np.atleast_2d(Fcont[:,ti-1]).T,ti)
                    
                
                
                used_u[:,ti] = uu.T
                
                Fcont[:,ti] =  ( Fcont[:,ti-1]+ h* f(Fcont[:,ti-1])+h*g**2 *uu.T+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,)) )
                Fnon[:,ti] =  ( Fnon[:,ti-1]+ h* f(Fnon[:,ti-1])+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,)) )
        
        UcontALL[gi,:,repi] = used_u
        FcontALL[gi,:,repi] = Fcont
        FnonALL[gi,:,repi] = Fnon


#%%
        
import pickle

to_save = dict()
to_save['UcontALL'] = UcontALL
to_save['FcontALL'] = FcontALL
to_save['timegrid'] = timegrid
to_save['FnonALL'] = FnonALL
#to_save['Rtt'] = Rtt
#to_save['K'] = K
to_save['t2'] = t2
to_save['gs'] = gs
to_save['N'] = N
to_save['M'] = M

#pickle.dump(to_save, open('Nonconservative_systematic_N_%d_M_%d'%(N,M), "wb"))      



#%% systematic pice



x1 = y1
x2 = y2
dt=h
R = 1
nu = 1
Q = 2

def bas(x,ord=1):  
  if ord==1:
    return x[0]
  elif ord==2:
    return x[1]
  elif ord ==3:
    return x[0]*x[1]
  elif ord ==4:
    return x[0]**2
  elif ord ==5:     ###up to power2
    return x[1]**2
  elif ord ==6:
    return (x[0]**2)*x[1]
  elif ord ==7:
    return (x[1]**2)*x[0]
  elif ord ==8:
    return x[0]**3
  elif ord ==9:
    return x[1]**3
  elif ord ==10:
    return (x[0]**2)*x[1]**2
  elif ord ==11:
    return (x[0]**2)*x[1]**2
  elif ord ==12:
    return (x[0]**3)*x[1]
  elif ord ==13:
    return (x[1]**3)*x[0]
  elif ord ==14:
    return x[1]**4
  elif ord ==15:
    return x[1]**4
  elif ord==0:
    return x[0]*0+1
eta = 0.05#0.1 #learning rate
Npice= 600#N#2000#particles

nu = 0.1
beta = 2 #annealing factor
gamma = 0.1 ## annealing threshold 

bas_order = 5 #max order of basis function
def phi(x,alph=10): #end cost
    epsi = 0.0001
    po = x2  ###x2
    # if (x>= po-epsi)  and (x<= po+epsi):
    #   return 0
    # elif (x>= -po-epsi)  and (x<= -po+epsi):
    #   return 0
    # else:
    return np.sum(alph @ np.sqrt((x-po)**2) ,axis=0)

def phi2(x,alph):
    ph = lambda xx: phi(xx,alph)
    return -np.vectorize(ph)(x)


def Path_cost(x):
  
   return np.sum((Q/2)*x**2,axis= 1)*dt
dim = 2 
#def PICE(x0):

max_iter = 1500 #max_iterations
ESS = 0.0001
FpiceALL = np.zeros((len(gs),2,reps,timegrid.size))
UpiceALL = np.zeros((len(gs),2,reps,timegrid.size))
for gi,g in enumerate(gs):
    sigma = g
    theta = 0.001 #effective sample size threshold 0.01
    At = np.zeros(( dim,timegrid.size, bas_order+1, max_iter))  ###timedependent coefficints for u: timepoints x k (order of basis)
    Ht = np.zeros(( timegrid.size, bas_order+1,bas_order+1)) 
    dQ = np.zeros(( timegrid.size,bas_order+1))         ### timegrid x iter x basis
    nos = np.zeros((dim,Npice, timegrid.size))  ##noise realisations   ##### dim xN x iter
    us = np.zeros((dim, Npice,timegrid.size)) #control value per path and timepoints  ### N x timegrid
    alph= np.ones(dim)
    endpoints1 = np.zeros(max_iter)
    endpoints2 = np.zeros(max_iter)
    def uf_pice(x,ti,iteri):
        ui = []
        for di in range(dim):
            ui.append([])
            # print( At[di,ti,:,iteri])
            # print( [bas(x,l) for l in range(bas_order+1)])
            # print( np.array([bas(x,l) for l in range(bas_order+1)]).shape )
            # print(  (  At[di,ti,:,iteri]@ np.array([bas(x,l) for l in range(bas_order+1)])  ).shape  )
            # print('-----')
            ui[di] = At[di,ti,:,iteri]@ np.array([bas(x,l) for l in range(bas_order+1)])
        return ui
    
    #plt.figure(),
    Z = np.zeros((dim,Npice,timegrid.size))
    just_changed1 = 0
    just_changed2 = 0
    for iteri in range(0,max_iter-1):
      print("Iteration: %d"%iteri)
      if True:#ESS< theta:  ##while the sampling is not good enough    
        for di in range(dim):
          Z[di,:,0] = x1[di]
    
        for ti,tt in enumerate(timegrid):
            if ti>0:
              nos[:,:,ti-1] = sigma* np.random.normal(loc = 0.0, scale = np.sqrt(dt),size=(dim,Npice))
              us[:,:,ti-1] = uf_pice(Z[:,:,ti-1],ti-1, iteri) ###this should have size dim x N
              Z[:,:,ti] = Z[:,:,ti-1] + dt * f(Z[:,:,ti-1] ) + dt* sigma* us[:,:,ti-1] + nos[:,:,ti-1]
    
        #plt.figure(),
        # plt.subplot(1,2,1)
        # plt.plot(timegrid, np.mean(Z[0],axis=0), alpha=0.5);
        # plt.subplot(1,2,2)
        # plt.plot(timegrid, np.mean(Z[1],axis=0), alpha=0.5);
        # plt.ylim(-4,4)
        # plt.show()
        Su =  np.nansum( (R/2)*us**2*dt,axis=(0,2)) +np.nansum( us* nos,axis=(0,2))   - np.nansum(phi2(Z[:,:,-1],alph) , axis=0)#+ Path_cost(Z)
        wu = np.exp(- Su )
        nullwu = np.where(wu==0)[0]
        # for entry in nullwu:
        #     print('he')
        #     wu[entry] = np.abs(np.random.normal(loc=0, scale=0.00001))
        wu = wu/np.nansum(wu)
        #print(wu)
        #ESS = -1/(np.log(Npice))* wu @np.log(wu).T  #1/(1+np.std(wu)**2)
        endpoints1[iteri] = np.nanmean(Z[0,:,-1]) 
        endpoints2[iteri] =  np.nanmean(Z[1,:,-1])
        print(endpoints1[iteri])
        print(endpoints2[iteri])
        print('------')
        theta_0 = 0.01
        if ( np.abs(endpoints1[iteri] - x2[0]) <= theta_0 ) and ( np.abs(endpoints2[iteri] - x2[1]) <= theta_0 ):
            print('Done!')
            break;
        else:
            if (iteri>50) and (np.abs(np.nanmean(np.diff(endpoints1[iteri-50:iteri]))) < 0.001 ) and just_changed1<=0:
                ##here I check if the optimisation gets stuck but the end point is not reachable yet, 
                ## and I increase the relative weight of the end condition in the cost fucntion
                just_changed1 =50
                alph[0] = alph[0] *1.25
                print(alph)
            elif (iteri>50) and (np.abs(np.mean(np.diff(endpoints2[iteri-50:iteri]))) < 0.001 ) and just_changed2<=0:
                ##here I check if the optimisation gets stuck but the end point is not reachable yet, 
                ## and I increase the relative weight of the end condition in the cost fucntion
                just_changed2 =50
                alph[1] = alph[1] *1.25
                print(alph)
                
            else:
                just_changed1 = just_changed1 -1
                just_changed2 = just_changed2 -1
        
    
    
        ###estimating H(t) and dQ(ht)
        for di in range(dim):
            hXt = lambda x: np.array([bas(x,l) for l in range(bas_order+1)])
            for ti,tt in enumerate(timegrid):
              hX = hXt(Z[:,:,ti])   ## N x k      
              Ht[ti] =  np.nansum(np.einsum('in,jn->nij',wu*hX, hX) , axis=0)   #basxN   , basxN  ---> N xbas x bas 
              dQ[ti] = np.nansum(np.einsum('n,kn->nk',   wu*nos[di,:,ti] , hX ), axis=0)
        
        
              At[di,ti, :, iteri+1] = At[di,ti, :, iteri] + eta *dQ[ti]/dt @np.linalg.pinv(Ht[ti])   
      ###end of estimation of u
      
    iterat = iteri
    dim = 2    
    Fcont_pice = np.zeros((2,reps,timegrid.size))
    used_upice =  np.zeros((2,reps,timegrid.size))
    
    for ti,tt in enumerate(timegrid[:]):
        ### ti is local time, tti is global time - both are time indices
          ## index of timepoint in the initial timegrid- the real time axis
        
        if ti==0:
            
            Fcont_pice[0,:,ti] = y1[0] 
            
            Fcont_pice[1,:,ti] = y1[1]        
        
        else:        
            
            uu3 = np.array( uf_pice(Fcont_pice[:,:,ti-1],ti-1,iterat) )
            
            used_upice[:,:,ti] = uu3
            
            
            Fcont_pice[:,:,ti] =  ( Fcont_pice[:,:,ti-1]+ h* f(Fcont_pice[:,:,ti-1])+h*g**2 *uu3+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,reps)) )
            
    FpiceALL[gi] = Fcont_pice 
    UpiceALL[gi] = used_upice         

#FpiceALLB = UpiceALL
#%%
import pickle

to_save = dict()
to_save['UpiceALL'] = UpiceALL
to_save['FpiceALL'] = FpiceALL
to_save['timegrid'] = timegrid

#to_save['Rtt'] = Rtt
#to_save['K'] = K
to_save['t2'] = t2
to_save['gs'] = gs
to_save['N'] = N
to_save['bas_order'] = bas_order
to_save['eta'] = eta

#pickle.dump(to_save, open('Nonconservative_systematic_N_%d_pice_bas_order_%d.dat'%(N,bas_order), "wb"))      
#%%

to_save = dict()
to_save['Fcont'] = Fcont
to_save['Fnon'] = Fnon
to_save['timegrid'] = timegrid



#pickle.dump(to_save, open('Nonconservative_systematic_N_%d_example_trajectories.dat'%(N), "wb"))      
#%% load file
import pickle
naming = 'Nonconservative_systematic_N_600_M_20'
to_save  =  pickle.load(open(naming, "rb") )

UcontALL = to_save['UcontALL'] 
FcontALL = to_save['FcontALL'] 
timegrid = to_save['timegrid'] 
FnonALL = to_save['FnonALL'] 

t2 = to_save['t2'] 
gs = to_save['gs'] 
N= to_save['N'] 
M = to_save['M'] 
T=t2
#%%
naming = 'Nonconservative_systematic_N_600_pice_bas_order_5.dat'
to_save  =  pickle.load(open(naming, "rb") )
UpiceALL = to_save['UpiceALL'] 
FpiceALL = to_save['FpiceALL'] 

#%%
naming = 'Nonconservative_systematic_N_600_example_trajectories.dat'
to_save  =  pickle.load(open(naming, "rb") )


Fcont = to_save['Fcont'] 
Fnon = to_save['Fnon'] 
#to_save['timegrid'] = timegrid



 

#%%plot means


plt.figure(figsize=(8,10)),
for gi in range(4):
    plt.subplot(2,2, gi+1)

    mnstdlc = np.nanmean(FcontALL[gi,0] ,axis=0)-np.nanstd(FcontALL[gi,0],axis=0)
    mnstdupc = np.nanmean(FcontALL[gi,0],axis=0)+np.nanstd(FcontALL[gi,0],axis=0)
    #plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )
    
    plt.plot(timegrid,np.nanmean(FcontALL[gi,0],axis=0),linestyle=(0, (3, 3)), c='r', lw=4, zorder=2)
    plt.plot(timegrid,np.mean(FpiceALL[gi,0,:],axis=0),'-', c='grey',lw=4.,zorder=0)
    #plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
    plt.plot(timegrid,np.mean(FnonALL[gi,0],axis=0),'-',c='green',lw=2, zorder=0,alpha=0.95)
    plt.plot(timegrid,np.mean(FcontALL[gi,0],axis=0)-np.std(FcontALL[gi,0],axis=0),'.', c='r',lw=4,label=r'$\sigma_{\hat{q}_t^{DPF}}$')
    plt.plot(timegrid,np.mean(FcontALL[gi,0],axis=0)+np.std(FcontALL[gi,0],axis=0),'.', c='r',lw=4)
    plt.plot(timegrid,np.mean(FpiceALL[gi,0],axis=0)-np.std(FpiceALL[gi,0],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0,label=r'$\sigma_{\hat{q}_t^{pice}}$')
    plt.plot(timegrid,np.mean(FpiceALL[gi,0],axis=0)+np.std(FpiceALL[gi,0],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
#mnstdl = np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0)
#mnstdup = np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0)
#plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,linewidth=0, zorder= 0)
  
plt.plot(timegrid[0],y1[1],'go')
plt.plot(timegrid[-1],y2[1],'X', c='silver')
#%%


for ii in range(2):

    for gi in range(4):
        plt.figure(figsize=(6,10)),
        plt.subplot(2,1,1)
        mnstdlc = np.nanmean(FcontALL[gi,ii] ,axis=0)-np.nanstd(FcontALL[gi,ii],axis=0)
        mnstdupc = np.nanmean(FcontALL[gi,ii],axis=0)+np.nanstd(FcontALL[gi,ii],axis=0)
        #plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )
        plt.plot(timegrid,FcontALL[gi,ii,:40].T,alpha=0.5, lw=1, zorder=2)
        plt.plot(timegrid,np.nanmean(FcontALL[gi,ii],axis=0),linestyle=(0, (3, 3)), c='r', lw=4, zorder=2)
        plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='grey',lw=2.,zorder=5)
        #plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
        plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)-np.std(FcontALL[gi,ii],axis=0),'.', c='r',lw=4,label=r'$\sigma_{\hat{q}_t^{DPF}}$')
        plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)+np.std(FcontALL[gi,ii],axis=0),'.', c='r',lw=4)
        
        plt.subplot(2,1,2)
        plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='r',lw=2.,zorder=0)
        plt.plot(timegrid,FpiceALL[gi,ii,:40].T,'-', c='grey',lw=1.,zorder=0,alpha=0.5)
        #plt.plot(timegrid,np.mean(FnonALL[gi,0],axis=0),'-',c='green',lw=2, zorder=0,alpha=0.95)
        plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)-np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0,label=r'$\sigma_{\hat{q}_t^{pice}}$')
        plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)+np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
    # plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
    # plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
    #mnstdl = np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0)
    #mnstdup = np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0)
    #plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,linewidth=0, zorder= 0)
      
    plt.plot(timegrid[0],y1[1],'go')
    plt.plot(timegrid[-1],y2[1],'X', c='silver')


#%% this is the one i used
import matplotlib.gridspec as gridspec    

dt = 0.001    

p1 = 0.5
p2= 0.5
p3 = 0.5
p4 = 0.5

def f(x,t=0):  
  x0 =  ( x[0]**4/ (p1**4 + x[0]**4)) + ( p2**4  / (p2**4 + x[1]**4))- x[0]
  x1 =  ( x[1]**4/ (p3**4 + x[1]**4)) + ( p2**4  / (p4**4 + x[0]**4))- x[1]
  return np.array([x0,x1])

fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

h = 0.001 #sim_prec
t_start = 0.

#x0 = np.array([1.81, -1.41])
x0 = fixed_points[:,0]#np.array([2., 0])
y1 = x0
y2 = fixed_points[:,2]
#import matplotlib.gridspec as gridspec
from matplotlib import cm
fig = plt.figure(figsize=(14, 3))
grs = gridspec.GridSpec(nrows=2, ncols=8, height_ratios=[1,1],wspace=2.2)

grs01 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=grs[0:2, 4:8],
                                         wspace=1.8)


#  Varying density along a streamline

purple_pal = cm.magma_r 
grey_pal = plt.get_cmap('Greys_r')
ds_palette = plt.get_cmap('plasma')
my_mag =  ds_palette(0.33)
my_maga =  ds_palette(0.39)
my_mag2 =  ds_palette(0.45)

gr1 = grey_pal(0.33)
gr2 = grey_pal(0.39)
gr3 = grey_pal(0.45)

my_orange = ds_palette(0.75)
my_orange2 = ds_palette(0.85)

my_palette = sns.color_palette( [my_mag , '#666666' ])
my_mag3 = (my_mag2[0], my_mag2[1],my_mag2[2],0.15)
my_orange3 = (my_orange2[0], my_orange2[1],my_orange2[2],0.15)
my_palette21 = sns.color_palette( [my_mag3 ,'#939393' ])    #'#939393'
my_palette22 = sns.color_palette( [ gr1,gr2,gr3 ])     
import pandas as pd


####### third plot
ax5 = fig.add_subplot(grs01[0:2, 0:2])
for gi in range(1,4):
    controls = np.log(np.array([np.power(np.sum(np.sqrt(np.nansum(np.power(UcontALL[gi,:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt),                             np.power(np.sum(np.sqrt(np.nansum(np.power(UpiceALL[gi,:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt)]) )
    df1 = pd.DataFrame({'u': gs[gi]**2*controls[0, :], 'type': 'DPF', 'g':gs[gi]})
    df2 = pd.DataFrame({'u': gs[gi]**2*controls[1, :], 'type': 'pice', 'g':gs[gi]})
    df = df1.append(df2,ignore_index=True)
    if gi==1:  
        dfall = df
    elif gi>1:
        dfall = dfall.append(df, ignore_index=True)
    

sns.violinplot( data=dfall, palette=my_palette21, alpha=0.65,saturation=0.81,
               x='g',y='u',hue='type')#color="0.8")
sns.stripplot( data=dfall,jitter=0.10,alpha=0.35, palette=my_palette,
              edgecolor='#363636',linewidth=0.25,size=3,y='u',x='g',hue='type',
              dodge=True)


# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.25

for tick, text in zip(ax5.get_xticks(), ax5.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val1 = dfall[  (dfall['type']=='DPF') & (dfall['g']==float(sample_name))  ].mean()['u']
    mean_val2 = dfall[  (dfall['type']=='pice') & (dfall['g']==float(sample_name))  ].mean()['u']
    # plot horizontal lines across the column, centered on the tick
    ax5.plot([tick-mean_width-0.07, tick-0.07], [mean_val1, mean_val1],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
    ax5.plot([tick+0.07, tick+mean_width+0.07], [mean_val2, mean_val2],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=False,left=False,ax=ax5)
#sns.despine(offset=10, trim=True)
plt.ylabel('control\n$\\log \\, \\| u(x,t) \\|_2^2$', multialignment='center',
           labelpad=0)#,fontsize=17)
plt.xlabel(r'noise $\sigma$')
ax5.spines['bottom'].set_color('#363636')
ax5.spines['top'].set_color('#363636')
ax5.xaxis.label.set_color('#363636')
ax5.tick_params(axis='x', colors='#363636')
ax5.yaxis.label.set_color('#363636')
ax5.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major')#, labelsize=16) 
plt.tick_params(axis='x', which='major')#, labelsize=16) 
ax5.xaxis.set_tick_params(width=0)
plt.locator_params(axis='y', nbins=7)

handles, labels = ax5.get_legend_handles_labels()


#handles2[0].set_linewidth(1.5)

leg1 = ax5.legend(handles[2:4], labels[2:4], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
          labelspacing=0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')
for text in leg1.get_texts():
    text.set_color('#4f4949')

################
    
    
ax6 = fig.add_subplot(grs01[0:2, 2:4])

for gi in range(1,4):
    endcost1 =  np.sqrt(np.sum( (FcontALL[gi,:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))    
    endcost2 =  np.sqrt(np.sum( (FpiceALL[gi,:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))              
    df1 = pd.DataFrame({'u': endcost1, 'type': 'DPF', 'g':gs[gi]})
    df2 = pd.DataFrame({'u': endcost2, 'type': 'pice', 'g':gs[gi]})
    df = df1.append(df2,ignore_index=True)
    if gi==1:  
        dfall = df
    elif gi>1:
        dfall = dfall.append(df, ignore_index=True)
    

sns.violinplot( data=dfall, palette=my_palette21, alpha=0.65,saturation=0.81,x='g',y='u',hue='type')#color="0.8")
sns.stripplot( data=dfall,jitter=0.10,alpha=0.35, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3,y='u',x='g',hue='type', dodge=True)


# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.25

for tick, text in zip(ax6.get_xticks(), ax6.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val1 = dfall[  (dfall['type']=='DPF') & (dfall['g']==float(sample_name))  ].mean()['u']
    mean_val2 = dfall[  (dfall['type']=='pice') & (dfall['g']==float(sample_name))  ].mean()['u']
    # plot horizontal lines across the column, centered on the tick
    ax6.plot([tick-mean_width-0.07, tick-0.07], [mean_val1, mean_val1],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
    ax6.plot([tick+0.07, tick+mean_width+0.07], [mean_val2, mean_val2],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=False,left=False,ax=ax6)
#sns.despine(offset=10, trim=True)
ax6.set_ylabel('terminal error\n$(x^*- X_T)^2$', multialignment='center',
               labelpad=-0)
plt.xlabel(r'noise $\sigma$')
ax6.spines['bottom'].set_color('#363636')
ax6.spines['top'].set_color('#363636')
ax6.xaxis.label.set_color('#363636')
ax6.tick_params(axis='x', colors='#363636')
ax6.yaxis.label.set_color('#363636')
ax6.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major')#, labelsize=16) 
plt.tick_params(axis='x', which='major')#, labelsize=16) 
ax6.xaxis.set_tick_params(width=0)
plt.locator_params(axis='y', nbins=7)
plt.ylim(None,0.5)
ax6.get_legend().remove()

################### 1st plot ##########################

w = 2.25
Y, X = np.mgrid[-0.25:w:100j, -0.25:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

#fig = plt.figure(figsize=(8, 4))
#gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])

#  Varying density along a streamline

grs02 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=grs[0:2, 0:4],
                                         wspace=0.6, hspace=0.6)
# Varying color along a streamline
ax1 = fig.add_subplot(grs02[0:2, 0:2])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="magma",levels=200,alpha=0.99)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color='#4f4949', density=[0.45, 0.45])
ax1.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax1.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax1.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax1.plot(Fnon[0],Fnon[1], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], 
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, 
                 label='controlled')
    else:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], 
                 c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)

#fig.colorbar(strm.lines)

ax1.set_aspect(1)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.set_title('Varying Color')
# ax1.set_xticks([-0.5, 0.5, 1.5,2.5])
ax1.set_yticks([0, 1, 2])
ax1.set_xlabel(r'x$_1$')
ax1.set_ylabel(r'x$_2$')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[::-1], labels[::-1], title=None,
          handletextpad=0.5, columnspacing=1.5,handlelength=0.8, 
          bbox_to_anchor=[-0.1, 1.25],
          loc=2, ncol=2, frameon=True,fontsize = 'medium',shadow=None,
          framealpha =0,edgecolor ='#0a0a0a')

##########################

gi=3
for ii in range(2):


    
    ax2 = fig.add_subplot(grs02[ii:ii+1, 2:4])
    mnstdlc = np.nanmean(FcontALL[gi,ii] ,axis=0)-np.nanstd(FcontALL[gi,ii],axis=0)
    mnstdupc = np.nanmean(FcontALL[gi,ii],axis=0)+np.nanstd(FcontALL[gi,ii],axis=0)
    #plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )
    #plt.plot(timegrid,FcontALL[gi,ii,:40].T,alpha=0.5, lw=1, zorder=2)
    plt.plot(timegrid,np.nanmean(FcontALL[gi,ii],axis=0),linestyle=(0, (3, 3.5, 1.5, 3.5)), c=my_mag, lw=4, zorder=2,label=r'$\mu^{\mathrm{DPF}}_t$',dash_capstyle='round')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='grey',lw=4.,zorder=1, label=r'$\mu^{\mathrm{pice}}_t$')
    #plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
    plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)-np.std(FcontALL[gi,ii],axis=0),'--', c=my_mag,lw=4, label=r'$\sigma^{\mathrm{DPF}}_t$')#,label=r'$\sigma_{\hat{q}_t^{DPF}}$')
    plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)+np.std(FcontALL[gi,ii],axis=0),'--', c=my_mag,lw=4)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.tick_params(axis="both", which='major',direction="in", top=True, right=True, bottom=True, left=True,size=3, colors='#4f4949')    
    #plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='r',lw=2.,zorder=0)
    #plt.plot(timegrid,FpiceALL[gi,ii,:40].T,'-', c='grey',lw=1.,zorder=0,alpha=0.5)
    #plt.plot(timegrid,np.mean(FnonALL[gi,ii],axis=0),'-',c='green',lw=2, zorder=0,alpha=0.95,label=r'$\mu^{uncontr}_t$')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)-np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0, label=r'$\sigma^{\mathrm{pice}}_t$')#,label=r'$\sigma_{\hat{q}_t^{pice}}$')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)+np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
    plt.plot(timegrid[0],y1[ii],'go')
    plt.plot(timegrid[-1],y2[ii],'X', c='silver')
    if ii==0:
        plt.ylabel(r'x$_1$')
    else:
        plt.ylabel(r'x$_2$')
    if ii==0:
        
        handles, labels = ax2.get_legend_handles_labels()

        ax2.legend(handles[::1], labels[::1], title=None,
                  handletextpad=0.5, columnspacing=0.85,handlelength=0.8, bbox_to_anchor=[0.13, 1.75],
                  loc=2, ncol=2, frameon=True,fontsize =11,shadow=None,framealpha =0,edgecolor ='#0a0a0a')

        
    plt.xlabel('time')


# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
#mnstdl = np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0)
#mnstdup = np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0)
#plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,linewidth=0, zorder= 0)
  


#plt.subplots_adjust(wspace=1.85,hspace=0.6)



#plt.tight_layout()
plt.show()
plt.savefig('systematic_non_conservative.png', bbox_inches='tight',dpi=300 , pad_inches = 0.,transparent='False',facecolor='white')
plt.savefig('systematic_non_conservative.pdf', bbox_inches='tight',dpi=300,  pad_inches = 0.,transparent='False',facecolor='white')

#plt.close()
#%% Compare control costsv bigger version of figure
dt = 0.001    

p1 = 0.5
p2= 0.5
p3 = 0.5
p4 = 0.5

def f(x,t=0):  
  x0 =  ( x[0]**4/ (p1**4 + x[0]**4)) + ( p2**4  / (p2**4 + x[1]**4))- x[0]
  x1 =  ( x[1]**4/ (p3**4 + x[1]**4)) + ( p2**4  / (p4**4 + x[0]**4))- x[1]
  return np.array([x0,x1])

fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

h = 0.001 #sim_prec
t_start = 0.

#x0 = np.array([1.81, -1.41])
x0 = fixed_points[:,0]#np.array([2., 0])
y1 = x0
y2 = fixed_points[:,2]
import matplotlib.gridspec as gridspec
from matplotlib import cm
fig = plt.figure(figsize=(20, 4))
grs = gridspec.GridSpec(nrows=2, ncols=8, height_ratios=[1,1])

#  Varying density along a streamline

purple_pal = cm.magma_r 
grey_pal = plt.get_cmap('Greys_r')
ds_palette = plt.get_cmap('plasma')
my_mag =  ds_palette(0.33)
my_maga =  ds_palette(0.39)
my_mag2 =  ds_palette(0.45)

gr1 = grey_pal(0.33)
gr2 = grey_pal(0.39)
gr3 = grey_pal(0.45)

my_orange = ds_palette(0.75)
my_orange2 = ds_palette(0.85)

my_palette = sns.color_palette( [my_mag , '#666666' ])
my_mag3 = (my_mag2[0], my_mag2[1],my_mag2[2],0.15)
my_orange3 = (my_orange2[0], my_orange2[1],my_orange2[2],0.15)
my_palette21 = sns.color_palette( [my_mag3 ,'#939393' ])    #'#939393'
my_palette22 = sns.color_palette( [ gr1,gr2,gr3 ])     
import pandas as pd

ax5 = fig.add_subplot(grs[0:2, 4:6])
for gi in range(1,4):
    controls = np.log(np.array([np.power(np.sum(np.sqrt(np.nansum(np.power(UcontALL[gi,:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt),                             np.power(np.sum(np.sqrt(np.nansum(np.power(UpiceALL[gi,:,:,:-1],2), axis=0)),axis=-1),2) /(T/dt)]) )
    df1 = pd.DataFrame({'u': gs[gi]**2*controls[0, :], 'type': 'DPF', 'g':gs[gi]})
    df2 = pd.DataFrame({'u': gs[gi]**2*controls[1, :], 'type': 'pice', 'g':gs[gi]})
    df = df1.append(df2,ignore_index=True)
    if gi==1:  
        dfall = df
    elif gi>1:
        dfall = dfall.append(df, ignore_index=True)
    

sns.violinplot( data=dfall, palette=my_palette21, alpha=0.65,saturation=0.81,x='g',y='u',hue='type')#color="0.8")
sns.stripplot( data=dfall,jitter=0.10,alpha=0.35, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3,y='u',x='g',hue='type', dodge=True)


# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.25

for tick, text in zip(ax5.get_xticks(), ax5.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val1 = dfall[  (dfall['type']=='DPF') & (dfall['g']==float(sample_name))  ].mean()['u']
    mean_val2 = dfall[  (dfall['type']=='pice') & (dfall['g']==float(sample_name))  ].mean()['u']
    # plot horizontal lines across the column, centered on the tick
    ax5.plot([tick-mean_width-0.07, tick-0.07], [mean_val1, mean_val1],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
    ax5.plot([tick+0.07, tick+mean_width+0.07], [mean_val2, mean_val2],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=False,left=False,ax=ax5)
#sns.despine(offset=10, trim=True)
plt.ylabel('control\n$\\log \\, \\| u(x,t) \\|_2^2$', multialignment='center')#,fontsize=17)
plt.xlabel(r'noise $\sigma$')
ax5.spines['bottom'].set_color('#363636')
ax5.spines['top'].set_color('#363636')
ax5.xaxis.label.set_color('#363636')
ax5.tick_params(axis='x', colors='#363636')
ax5.yaxis.label.set_color('#363636')
ax5.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major')#, labelsize=16) 
plt.tick_params(axis='x', which='major')#, labelsize=16) 
ax5.xaxis.set_tick_params(width=0)
plt.locator_params(axis='y', nbins=7)

handles, labels = ax5.get_legend_handles_labels()


#handles2[0].set_linewidth(1.5)

leg1 = ax5.legend(handles[2:4], labels[2:4], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,labelspacing = 0.3,
          loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')
for text in leg1.get_texts():
    text.set_color('#4f4949')

################
    
    
ax6 = fig.add_subplot(grs[0:2, 6:8])

for gi in range(1,4):
    endcost1 =  np.sqrt(np.sum( (FcontALL[gi,:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))    
    endcost2 =  np.sqrt(np.sum( (FpiceALL[gi,:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))              
    df1 = pd.DataFrame({'u': endcost1, 'type': 'DPF', 'g':gs[gi]})
    df2 = pd.DataFrame({'u': endcost2, 'type': 'pice', 'g':gs[gi]})
    df = df1.append(df2,ignore_index=True)
    if gi==1:  
        dfall = df
    elif gi>1:
        dfall = dfall.append(df, ignore_index=True)
    

sns.violinplot( data=dfall, palette=my_palette21, alpha=0.65,saturation=0.81,x='g',y='u',hue='type')#color="0.8")
sns.stripplot( data=dfall,jitter=0.10,alpha=0.35, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3,y='u',x='g',hue='type', dodge=True)


# distance across the "X" or "Y" stipplot column to span, in this case 30%
mean_width = 0.25

for tick, text in zip(ax6.get_xticks(), ax6.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val1 = dfall[  (dfall['type']=='DPF') & (dfall['g']==float(sample_name))  ].mean()['u']
    mean_val2 = dfall[  (dfall['type']=='PICE') & (dfall['g']==float(sample_name))  ].mean()['u']
    # plot horizontal lines across the column, centered on the tick
    ax6.plot([tick-mean_width-0.07, tick-0.07], [mean_val1, mean_val1],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
    ax6.plot([tick+0.07, tick+mean_width+0.07], [mean_val2, mean_val2],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True, bottom=False,left=False,ax=ax6)
#sns.despine(offset=10, trim=True)
ax6.set_ylabel('terminal error\n$(x^*- X_T)^2$', multialignment='center')
plt.xlabel(r'noise $\sigma$')
ax6.spines['bottom'].set_color('#363636')
ax6.spines['top'].set_color('#363636')
ax6.xaxis.label.set_color('#363636')
ax6.tick_params(axis='x', colors='#363636')
ax6.yaxis.label.set_color('#363636')
ax6.tick_params(axis='y', colors='#363636')    
plt.tick_params(axis='y', which='major')#, labelsize=16) 
plt.tick_params(axis='x', which='major')#, labelsize=16) 
ax6.xaxis.set_tick_params(width=0)
plt.locator_params(axis='y', nbins=7)
plt.ylim(None,0.5)
ax6.get_legend().remove()

###################

w = 2.5
Y, X = np.mgrid[-0.5:w:100j, -0.5:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

#fig = plt.figure(figsize=(8, 4))
#gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])

#  Varying density along a streamline


# Varying color along a streamline
ax1 = fig.add_subplot(grs[0:2, 0:2])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color='#4f4949', density=[0.75, 0.75])
ax1.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax1.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax1.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax1.plot(Fnon[0],Fnon[1], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, label='controlled')
    else:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)

#fig.colorbar(strm.lines)

ax1.set_aspect(1)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.set_title('Varying Color')
ax1.set_xticks([-0.5, 0.5, 1.5,2.5])
ax1.set_yticks([-0.5, 0.5, 1.5,2.5])
ax1.set_xlabel(r'x$_1$')
ax1.set_ylabel(r'x$_2$')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[::-1], labels[::-1], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.53, 0.955],
          loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')

##########################

gi=3
for ii in range(2):


    
    ax2 = fig.add_subplot(grs[ii:ii+1, 2:4])
    mnstdlc = np.nanmean(FcontALL[gi,ii] ,axis=0)-np.nanstd(FcontALL[gi,ii],axis=0)
    mnstdupc = np.nanmean(FcontALL[gi,ii],axis=0)+np.nanstd(FcontALL[gi,ii],axis=0)
    #plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )
    #plt.plot(timegrid,FcontALL[gi,ii,:40].T,alpha=0.5, lw=1, zorder=2)
    plt.plot(timegrid,np.nanmean(FcontALL[gi,ii],axis=0),linestyle=(0, (3, 3.5, 1.5, 3.5)), c=my_mag, lw=4, zorder=2,label=r'$\mu^{DPF}_t$',dash_capstyle='round')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='grey',lw=4.,zorder=1, label=r'$\mu^{pice}_t$')
    #plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
    plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)-np.std(FcontALL[gi,ii],axis=0),'--', c=my_mag,lw=4, label=r'$\sigma^{DPF}_t$')#,label=r'$\sigma_{\hat{q}_t^{DPF}}$')
    plt.plot(timegrid,np.mean(FcontALL[gi,ii],axis=0)+np.std(FcontALL[gi,ii],axis=0),'--', c=my_mag,lw=4)
    
    
    #plt.plot(timegrid,np.mean(FpiceALL[gi,ii,:],axis=0),'-', c='r',lw=2.,zorder=0)
    #plt.plot(timegrid,FpiceALL[gi,ii,:40].T,'-', c='grey',lw=1.,zorder=0,alpha=0.5)
    #plt.plot(timegrid,np.mean(FnonALL[gi,ii],axis=0),'-',c='green',lw=2, zorder=0,alpha=0.95,label=r'$\mu^{uncontr}_t$')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)-np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0, label=r'$\sigma^{pice}_t$')#,label=r'$\sigma_{\hat{q}_t^{pice}}$')
    plt.plot(timegrid,np.mean(FpiceALL[gi,ii],axis=0)+np.std(FpiceALL[gi,ii],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
    plt.plot(timegrid[0],y1[ii],'go')
    plt.plot(timegrid[-1],y2[ii],'X', c='silver')
    if ii==0:
        plt.ylabel(r'$x_1$')
    else:
        plt.ylabel(r'$x_2$')
    if ii==0:
        
        handles, labels = ax2.get_legend_handles_labels()

        ax2.legend(handles[::1], labels[::1], title=None,
                  handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.53, 0.955],
                  loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')

        
plt.xlabel('time')


# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
# plt.plot(timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey',zorder=0,alpha=0.95,lw=1)
#mnstdl = np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0)
#mnstdup = np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0)
#plt.fill_between(timegrid,y1=mnstdl,y2 = mnstdup,color=orag, alpha=0.2,linewidth=0, zorder= 0)
  


plt.subplots_adjust(wspace=0.25,hspace=1)



plt.tight_layout()
plt.show()
plt.savefig('systematic_non_conservative.png', bbox_inches='tight',dpi=300 , pad_inches = 0.1,transparent='False',facecolor='white')
plt.savefig('systematic_non_conservative.pdf', bbox_inches='tight',dpi=300,  pad_inches = 0.1,transparent='False',facecolor='white')
#%%



w = 2.5
Y, X = np.mgrid[-0.5:w:100j, -0.5:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

fig = plt.figure(figsize=(13, 4))
grs = gridspec.GridSpec(nrows=1, ncols=3, height_ratios=[1])

#  Varying density along a streamline
for gi in range(1,4):
    ax0 = fig.add_subplot(grs[0, gi-1])
    plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
    #plt.colorbar()
    ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), density=[0.75, 0.75], color='#4f4949')
    
    
    
    #ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1)
    ax0.plot( np.nanmean(FcontALL[gi,0],axis=0), np.mean(FcontALL[gi,1],axis=0),'.',alpha=0.85, c=my_mag,lw=2.5,label=r'$\mu^{\mathrm{DPF}}_t$')
    ax0.plot( np.nanmean(FcontALL[gi,0],axis=0)+ np.nanstd(FcontALL[gi,0],axis=0), np.nanmean(FcontALL[gi,1],axis=0) + np.nanstd(FcontALL[gi,1],axis=0),'--',alpha=0.85,c=my_mag,lw=2.5,label=r'$\sigma^{\mathrm{DPF}}_t$')
    ax0.plot( np.nanmean(FcontALL[gi,0],axis=0)- np.nanstd(FcontALL[gi,0],axis=0), np.nanmean(FcontALL[gi,1],axis=0) - np.nanstd(FcontALL[gi,1],axis=0),'--',alpha=0.85, c=my_mag,lw=2.5)
    
    ax0.plot( np.nanmean(FpiceALL[gi,0],axis=0), np.mean(FpiceALL[gi,1],axis=0),'.',alpha=0.85, c='grey',lw=2.5,label=r'$\mu^{\mathrm{pice}}_t$')
    ax0.plot( np.nanmean(FpiceALL[gi,0],axis=0)+ np.nanstd(FpiceALL[gi,0],axis=0), np.nanmean(FpiceALL[gi,1],axis=0) + np.nanstd(FpiceALL[gi,1],axis=0),'--',alpha=0.85,c='grey',lw=2.5,label=r'$\sigma^{\mathrm{pice}}_t$')
    ax0.plot( np.nanmean(FpiceALL[gi,0],axis=0)- np.nanstd(FpiceALL[gi,0],axis=0), np.nanmean(FpiceALL[gi,1],axis=0) - np.nanstd(FpiceALL[gi,1],axis=0),'--',alpha=0.85, c='grey',lw=2.5)
    
    ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
    ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
    ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
    
    #ax0.set_title('Varying Density')
    ax0.set_aspect(1)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.set_xticks([-0.5, 0.5, 1.5,2.5])
    ax0.set_yticks([-0.5, 0.5, 1.5,2.5])
    ax0.set_xlabel(r'x$_1$')
    ax0.set_ylabel(r'x$_2$')
    plt.title(r'$\sigma = %.1f$'%gs[gi])
    
    if gi==2:
        
        handles, labels = ax0.get_legend_handles_labels()

        ax0.legend(handles[::1], labels[::1], title=None,
                  handletextpad=0.5, columnspacing=3.2,handlelength=0.8,
                  bbox_to_anchor=[-0.53, 0.955],
                  loc=2, ncol=1, frameon=True,fontsize = 'small',
                  shadow=None,framealpha =0,edgecolor ='#0a0a0a')

        
    
plt.savefig('systematic_non_conservative_appendix.png', bbox_inches='tight',
            dpi=300 , pad_inches = 1,transparent='False',facecolor='white')
plt.savefig('systematic_non_conservative_appendix.pdf', bbox_inches='tight',
            dpi=300,  pad_inches = 1,transparent='False',facecolor='white')
#%%
ax6 = fig.add_subplot(gs[1, 2])
end_cost1 = 1*np.sqrt(np.sum( (Fcont[:,:,-1]-np.atleast_2d(y2).T)**2 ,axis=0  ))

end_cost3 = 1*np.sqrt( np.sum((Fcont_pice[ni_pice][:,:,-2]-np.atleast_2d(y2).T)**2   ,axis=0))


df_end = pd.DataFrame({r'$DPF$': end_cost1, r'$pice$': end_cost3})

sns.violinplot(  data=df_end, palette=my_palette2, alpha=0.65,saturation=0.81)#color="0.8")

sns.stripplot( data=df_end,jitter=0.15,alpha=0.5, palette=my_palette,edgecolor='#363636',linewidth=0.25,size=3)
mean_width = 0.5

for tick, text in zip(ax6.get_xticks(), ax6.get_xticklabels()):
    sample_name = text.get_text()  # "X" or "Y"
    # calculate the median value for all replicates of either X or Y
    mean_val = df_end[sample_name].mean()
    # plot horizontal lines across the column, centered on the tick
    ax6.plot([tick-mean_width/2, tick+mean_width/2], [mean_val, mean_val],
            lw=4, color='silver',solid_capstyle='round',zorder=3)
sns.despine(trim=True,  top=True, right=True,bottom=True,ax=ax6)    
plt.ylabel('terminal error\n$(x^*- X_T)^2$', multialignment='center')
ax6.spines['bottom'].set_color('#363636')
ax6.spines['top'].set_color('#363636')
ax6.xaxis.label.set_color('#363636')
ax6.tick_params(axis='x', colors='#363636')

ax6.yaxis.label.set_color('#363636')
ax6.tick_params(axis='y', colors='#363636')       
plt.tick_params(axis='y', which='major',color='#4f4949')#, labelsize=16) 
plt.tick_params(axis='x', which='major',color='#4f4949')#, labelsize=16) 
ax6.xaxis.set_tick_params(width=0)


########################################################
ax1.axis('on')
ax2.axis('on')
plt.tight_layout()
plt.subplots_adjust(wspace=0.5,hspace=0.5)


#################################################
##here I add a nested gridspec to manipulate the horizontal whitespace
##since I dont want it to be equal for every subplot
gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0:2,3], hspace=0.1)
ax10 = fig.add_subplot(gs01[0])#fig.add_subplot(gs[0, 3])

    
sns.pointplot(x="N",  y="control", hue='M',
              data=dpf_df, dodge=.532, join=True, palette=["m", "g", 'grey'],zorder=10,
              markers="o", scale=1, ci=None ,ax=ax10)
#plt.yscale('log')
handles2, labels2 = ax10.get_legend_handles_labels()
for li,lab in enumerate(labels2[:2]):    
    labels2[li] = r'$DPF - M:$' +labels2[li] 
leg52 = ax10.legend(handles2[:3], labels2[:3], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8,# bbox_to_anchor=[-0.5, 0.655],
          loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')

for text in leg52.get_texts():
    text.set_color('#4f4949')
    
ax10.set_ylabel('control\n$ \\| u(x,t) \\|_2^2$', multialignment='center')

ax10.set_xlabel(r'')
ax10.tick_params(axis="both",direction="in", bottom=True, left=True,top=True, right=True,color='#4f4949')

#ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax10.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax10.tick_params(axis="both", which='minor',direction="in", bottom=True, left=True, top=True, right=True,color='#4f4949')
plt.ylim(10**4,3.5*10**4)
ax10.set_yticks([ 2*10**4,3*10**4])
ax10.ticklabel_format(style='sci',axis='y',scilimits=(1,4))
ax10.yaxis.major.formatter._useMathText = True
plt.setp(ax10.get_xticklabels(), visible=False)
ax10.spines['bottom'].set_color('#363636')
ax10.spines['top'].set_color('#363636')
ax10.xaxis.label.set_color('#363636')
ax10.tick_params(axis='x', colors='#4f4949')

ax10.yaxis.label.set_color('#363636')
ax10.tick_params(axis='y', colors='#363636')       
plt.tick_params(axis='y', which='major',color='#363636')#, labelsize=16) 
plt.tick_params(axis='x', which='major',color='#363636')





ax11 = fig.add_subplot(gs01[1], sharex=ax10)#fig.add_subplot(gs[1, 3], sharex=ax10)


g = sns.pointplot(x="N",  y="end_error", hue='M',
              data=dpf_df, dodge=.532, join=True, palette=["m", "g", 'grey'],zorder=10,
              markers="o", scale=1, ci=None ,ax=ax11)

#plt.yscale('log')
# handles3, labels3 = ax11.get_legend_handles_labels()
# leg51 = ax11.legend(handles3[:3], labels3[:3], title=None,
#           handletextpad=0.5, columnspacing=3.2,handlelength=0.8, #bbox_to_anchor=[-0.5, 0.655],
#           loc=1, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')
# for text in leg51.get_texts():
#     text.set_color('#4f4949')
plt.legend([],[], frameon=False)    
ax11.set_ylabel('terminal error\n$(x^*- X_T)^2$', multialignment='center')

ax11.set_xlabel('particles N')
ax11.tick_params(axis="both",direction="in", top=True, left=True,bottom=True, right=True)
#ax10.tick_params(axis="both", which='major',direction="out", top=False, right=False, bottom=True, left=True,size=3, colors='#4f4949')        

#ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax11.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax11.tick_params(axis="both", which='minor',direction="in", bottom=True, left=True,top=True, right=True, color='#4f4949')
plt.ylim(0.02,0.12)
#fig.subplots_adjust(hspace=0.1)
plt.show()
#plt.savefig('Evolutionary_with_path_2d_pheno_final.png', bbox_inches='tight',dpi=300 , pad_inches = 1)
#plt.savefig('Evolutionary_with_path_2d_pheno_final.pdf', bbox_inches='tight',dpi=300,  pad_inches = 1)



#%%
gi=0
plt.figure()

mnstdlc = np.mean(FcontALL[gi,0] ,axis=0)-np.std(FcontALL[gi,0],axis=0)
mnstdupc = np.mean(FcontALL[gi,0],axis=0)+np.std(FcontALL[gi,0],axis=0)
#plt.fill_between(timegrid,y1=mnstdlc,y2 = mnstdupc,color=my_mag, alpha=0.24,linewidth=0.15, zorder= 1 )

plt.plot(timegrid,np.mean(FcontALL[gi,0],axis=0),linestyle=(0, (3, 3)), c='r', lw=4, zorder=2)
plt.plot(timegrid,np.mean(FpiceALL[gi,0],axis=0),'-', c='grey',lw=4.,zorder=0)
#plt.plot(timegrid,np.mean(Fcont_pice[ni_pice][0],axis=0),linestyle=(0, (1, 4)), c='#4f4949',lw=4,dash_capstyle='butt')
plt.plot(timegrid,np.mean(FnonALL[gi,0],axis=0),'-',c='green',lw=2, zorder=0,alpha=0.95)
plt.plot(timegrid,np.mean(FcontALL[gi,0],axis=0)-np.std(FcontALL[gi,0],axis=0),'.', c='r',lw=4,label=r'$\sigma_{\hat{q}_t^{DPF}}$')
plt.plot(timegrid,np.mean(FcontALL[gi,0],axis=0)+np.std(FcontALL[gi,0],axis=0),'.', c='r',lw=4)
plt.plot(timegrid,np.mean(FpiceALL[gi,0],axis=0)-np.std(FpiceALL[gi,0],axis=0),linestyle=(0, (6, 5)),c='grey', lw=4,dash_capstyle='butt',zorder=0,label=r'$\sigma_{\hat{q}_t^{pice}}$')
plt.plot(timegrid,np.mean(FpiceALL[gi,0],axis=0)+np.std(FpiceALL[gi,0],axis=0),linestyle=(0, (6, 5)),c='grey',lw=4,dash_capstyle='butt',zorder=0)
plt.plot(timegrid[0],y1[1],'go')
plt.plot(timegrid[-1],y2[1],'X', c='silver')

#%%


plt.figure(),

plt.plot(FcontALL[gi,0].T)        
#%%
for ti,tt in enumerate(timegrid):
    ### ti is local time, tti is global time - both are time indices
      ## index of timepoint in the initial timegrid- the real time axis
    
    if ti==0:        
        Fnon[:,ti] = y1      
    # elif ti >= bridg2d.timegrid.size-5:
    #     uu = -grad_log_p_Gauss(np.atleast_2d(Fcont[:,ti-1]),bridg2d.timegrid.size-ti)
    else:        
        
        Fnon[:,ti] =  ( Fnon[:,ti-1]+ h* f(Fnon[:,ti-1])+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,)) )
        
        

#%%

import matplotlib.gridspec as gridspec
from matplotlib import cm
w = 2.5
Y, X = np.mgrid[-0.5:w:100j, -0.5:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0, 0])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#plt.colorbar()
ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), density=[0.75, 0.75], color='#4f4949')



#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1)
ax0.plot( np.mean(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0),'.',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
ax0.plot( np.mean(bridg2d.B[0],axis=0)+ np.std(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0) + np.std(bridg2d.B[1],axis=0),'--',alpha=0.85,c=cm.viridis(0.97),lw=2.5)
ax0.plot( np.mean(bridg2d.B[0],axis=0)- np.std(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0) - np.std(bridg2d.B[1],axis=0),'--',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
ax0.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
ax0.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
ax0.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)

ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')

#ax0.set_title('Varying Density')
ax0.set_aspect(1)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([-0.5, 0.5, 1.5,2.5])
ax0.set_yticks([-0.5, 0.5, 1.5,2.5])
ax0.set_xlabel(r'x$_1$')
ax0.set_ylabel(r'x$_2$')


# Varying color along a streamline
ax1 = fig.add_subplot(gs[0, 1])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color='#4f4949', density=[0.75, 0.75])
ax1.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax1.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax1.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax1.plot(Fnon[0],Fnon[1], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, label='controlled')
    else:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)

#fig.colorbar(strm.lines)

ax1.set_aspect(1)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.set_title('Varying Color')
ax1.set_xticks([-0.5, 0.5, 1.5,2.5])
ax1.set_yticks([-0.5, 0.5, 1.5,2.5])
ax1.set_xlabel(r'x$_1$')
ax1.set_ylabel(r'x$_2$')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[::-1], labels[::-1], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.53, 0.955],
          loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')


plt.tight_layout()
plt.show()


#%%


plt.figure(),
# plt.subplot(1,2,1),
plt.plot(bridg2d.timegrid[:],Fcont[0,:],'.')
plt.plot(bridg2d.timegrid[-1],y2[0],'.')
plt.figure(),
# plt.subplot(1,2,2),
plt.plot(bridg2d.timegrid[:],Fcont[1,:],'.')
plt.plot(bridg2d.timegrid[-1],y2[1],'.')
plt.vlines(bridg2d.timegrid[-1],0,4)




#%%
###make animation


import matplotlib.animation as animation 
import matplotlib.gridspec as gridspec
from matplotlib import cm
w = 2.5
Y, X = np.mgrid[-0.5:w:100j, -0.5:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=1, ncols=2, height_ratios=[1])

#  frame 1
ax0 = fig.add_subplot(gs[0, 0])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#plt.colorbar()
ax0.streamplot(X, Y, U.reshape(X.shape), V.reshape(Y.shape), density=[0.75, 0.75], color='#4f4949')
ax0.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax0.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax0.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')

#ax0.set_title('Varying Density')
ax0.set_aspect(1)
ax0.spines['left'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([-0.5, 0.5, 1.5,2.5])
ax0.set_yticks([-0.5, 0.5, 1.5,2.5])
ax0.set_xlabel(r'x$_1$')
ax0.set_ylabel(r'x$_2$')

line1, = ax0.plot( [], [],'.', alpha=0.85, c=cm.viridis(0.97),lw=2.5)
line2, = ax0.plot( [], [],'--', alpha=0.85, c=cm.viridis(0.97),lw=2.5)
line3, = ax0.plot( [], [],'--', alpha=0.85, c=cm.viridis(0.97),lw=2.5)
line4 = ax0.plot( *([[], []]*int(N/2)), alpha=0.51, c='grey',lw=0.5)
line5 = ax0.plot( *([[], []]*int(N/2)), marker='.', alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
line6 = ax0.plot( *([[], []]*int(N/10)), linestyle='-', alpha=0.71, c=cm.viridis(0.685),lw=1)
#lines = plt.plot( *([[], []]*N) )
# initialization function 
def init(): 
    	# creating an empty plot/frame 
    line1.set_data([], []) 
    line2.set_data([], []) 
    line3.set_data([], []) 
    for line in line4:
        line.set_data([], []) 
    for line in line5:
        line.set_data([], [])
    for line in line6:
        line.set_data([], [])
    
    return [line1, line2, line3]+ line4 + line5 + line6 

# lists to store x and y axis points 
xdata1, ydata1 = [], []
xdata2, ydata2 = [], [] 
xdata3, ydata3 = [], []
xdata4, ydata4 = [], [] 
xdata5, ydata5 = [], []
xdata6, ydata6 = [], [] 


# animation function 
def animate(i): 	
	
    # appending new points to x, y axes points list 
    xdata1.append(np.mean(bridg2d.Z[0,:,i],axis=0)) 
    ydata1.append(np.mean(bridg2d.Z[1,:,i],axis=0)) 
    xdata2.append(np.mean(bridg2d.Z[0,:,i],axis=0)+ np.std(bridg2d.Z[0,:,i],axis=0)) 
    ydata2.append(np.mean(bridg2d.Z[1,:,i],axis=0)+ np.std(bridg2d.Z[1,:,i],axis=0))
    xdata3.append(np.mean(bridg2d.Z[0,:,i],axis=0)- np.std(bridg2d.Z[0,:,i],axis=0)) 
    ydata3.append(np.mean(bridg2d.Z[1,:,i],axis=0)- np.std(bridg2d.Z[1,:,i],axis=0))
    xdata4 = (bridg2d.Z[0,::2,:i])  
    ydata4 = (bridg2d.Z[1,::2,:i])  
    xdata5 = bridg2d.Z[0,::2,i] 
    ydata5 = bridg2d.Z[1,::2,i]
    if i<=10:
        xdata6 = bridg2d.Z[0,::10,0:i] 
        ydata6 = bridg2d.Z[1,::10,0:i] 
    else:
        xdata6 = bridg2d.Z[0,::10,i-10:i] 
        ydata6 = bridg2d.Z[1,::10,i-10:i] 
        
    
    line1.set_data(xdata1, ydata1) 
    line2.set_data(xdata2, ydata2) 
    line3.set_data(xdata3, ydata3) 
    for il,line in enumerate(line4):
        line.set_data(xdata4[il], ydata4[il]) 
    for il,line in enumerate(line5):        
        line.set_data(xdata5[il], ydata5[il]) 
    for il,line in enumerate(line6):
        line.set_data(xdata6[il], ydata6[il]) 
    
    return [line1, line2, line3] + line4 + line5 + line6 


#ax0.plot( (bridg2d.Z[0]), (bridg2d.Z[1]),alpha=0.31, c='grey',lw=1)
    
# ax0.plot( np.mean(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0),'.',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.B[0],axis=0)+ np.std(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0) + np.std(bridg2d.B[1],axis=0),'--',alpha=0.85,c=cm.viridis(0.97),lw=2.5)
# ax0.plot( np.mean(bridg2d.B[0],axis=0)- np.std(bridg2d.B[0],axis=0), np.mean(bridg2d.B[1],axis=0) - np.std(bridg2d.B[1],axis=0),'--',alpha=0.85, c=cm.viridis(0.97),lw=2.5)
# ax0.plot( (bridg2d.B[0,::10]).T, (bridg2d.B[1,::10]).T,alpha=0.51, c='grey',lw=0.5)
# ax0.plot( (bridg2d.B[0,::10,200]).T, (bridg2d.B[1,::10,200]).T,'.',alpha=0.91, c=cm.viridis(0.685),lw=0.5,markersize=2)
# ax0.plot( (bridg2d.B[0,::10,200:218]).T, (bridg2d.B[1,::10,200:218]).T,'-',alpha=0.71, c=cm.viridis(0.685),lw=1)



# call the animator	 
anim = animation.FuncAnimation(fig, animate, init_func=init, 
							frames=timegrid.size, interval=20, blit=True,repeat=False) 

#%%
# Varying color along a streamline
ax1 = fig.add_subplot(gs[0, 1])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color='#4f4949', density=[0.75, 0.75])
ax1.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax1.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax1.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax1.plot(Fnon[0],Fnon[1], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, label='controlled')
    else:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)

#fig.colorbar(strm.lines)

ax1.set_aspect(1)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.set_title('Varying Color')
ax1.set_xticks([-0.5, 0.5, 1.5,2.5])
ax1.set_yticks([-0.5, 0.5, 1.5,2.5])
ax1.set_xlabel(r'x$_1$')
ax1.set_ylabel(r'x$_2$')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[::-1], labels[::-1], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.53, 0.955],
          loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')


plt.tight_layout()



# save the animation as mp4 video file 
#anim.save('coil.gif',writer='imagemagick') 
#%%


dim = 2

for rei in range(1):
    print(rei)
    reps = 1000
    Fcontall = np.zeros((2,reps,bridg2d.timegrid.size))
    Fcontpall = np.zeros((2,reps,bridg2d.timegrid.size))
    Fnonall =  np.zeros((2,reps,bridg2d.timegrid.size))
    used_uall =  np.zeros((2,reps,bridg2d.timegrid.size))
    used_upall =  np.zeros((2,reps,bridg2d.timegrid.size))
    new_pasall =  np.zeros((4,reps,bridg2d.timegrid.size))
    for ti,tt in enumerate(bridg2d.timegrid[:]):
        ### ti is local time, tti is global time - both are time indices
          ## index of timepoint in the initial timegrid- the real time axis
        
        if ti==0:
            Fcontall[0,:,ti] = y1[0] 
            Fcontpall[0,:,ti] = y1[0] 
            Fnonall[0,:,ti] = y1[0]  
            Fcontall[1,:,ti] = y1[1] 
            Fcontpall[1,:,ti] = y1[1] 
            Fnonall[1,:,ti] = y1[1]  
        # elif ti >= bridg2d.timegrid.size-5:
        #     uu = -grad_log_p_Gauss(np.atleast_2d(Fcont[:,ti-1]),bridg2d.timegrid.size-ti)
        else:        
            ###use previous grad log for current step
            if False:#ti>= bridg2d.timegrid.size-2:
                uu = bridg2d.calculate_u(np.atleast_2d(Fcontall[:,:,-2]).T,ti-1)
                #uup = bridg2d.calculate_u(np.atleast_2d(Fcontp[:,:,-2]).T,ti-1)
            elif ti>= bridg2d.timegrid.size-4:#-4
                uu = grad_log_p_Gauss( np.atleast_2d(Fcontall[:,:,ti-1]), bridg2d.timegrid.size -ti)
                uu = uu.T
                #uup = grad_log_p_Gauss( np.atleast_2d(Fcontp[:,:,ti-1]).T, bridg2d.timegrid.size -ti)
            else:
                uu = bridg2d.calculate_u(np.atleast_2d(Fcontall[:,:,ti-1]).T,ti)
                #uup = bridg2d.calculate_u(np.atleast_2d(Fcontp[:,:,ti-1]).T,ti)
                
            
            
            used_uall[:,:,ti] = uu.T
            #used_up[:,:,ti] = uup.T
            
            
            
            Fcontall[:,:,ti] =  ( Fcontall[:,:,ti-1]+ h* f(Fcontall[:,:,ti-1])+h*g**2*1 *uu.T+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,reps)) )
            Fnonall[:,:,ti] =  ( Fnonall[:,:,ti-1]+ h* f(Fnonall[:,:,ti-1])+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim,reps)) )

    # FcontALL[:,:,:] = Fcont
    # FcontpALL[:,:,rei] = Fcontp
#%%

plt.figure(figsize=(5,8)),
plt.subplot(2,1,1),
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[0],axis=0),'.')
plt.plot(bridg2d.timegrid[:],np.mean(Fnonall[0],axis=0),'-')
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[0],axis=0)-np.std(Fcontall[0],axis=0),'k--')
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[0],axis=0)+np.std(Fcontall[0],axis=0),'k--')
# plt.plot(bridg2d.timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey')
# plt.plot(bridg2d.timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey')
mnstdl = np.mean(Fnonall[0],axis=0)-np.std(Fnonall[0],axis=0)
mnstdup = np.mean(Fnonall[0],axis=0)+np.std(Fnonall[0],axis=0)
plt.fill_between(bridg2d.timegrid,y1=mnstdl,y2 = mnstdup,color='grey', alpha=0.2,linewidth=0)
plt.plot(bridg2d.timegrid[-1],y2[0],'.')

plt.subplot(2,1,2),
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[1,:],axis=0),'.')
plt.plot(bridg2d.timegrid[:],np.mean(Fnonall[1],axis=0),'-')
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[1],axis=0)-np.std(Fcontall[1],axis=0),'k--')
plt.plot(bridg2d.timegrid[:],np.mean(Fcontall[1],axis=0)+np.std(Fcontall[1],axis=0),'k--')
#plt.plot(bridg2d.timegrid[:],np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0),'--',c='grey')
#plt.plot(bridg2d.timegrid[:],np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0),'--',c='grey')
mnstdl = np.mean(Fnonall[1],axis=0)-np.std(Fnonall[1],axis=0)
mnstdup = np.mean(Fnonall[1],axis=0)+np.std(Fnonall[1],axis=0)
plt.fill_between(bridg2d.timegrid,y1=mnstdl,y2 = mnstdup,color='grey', alpha=0.2,linewidth=0)
plt.plot(bridg2d.timegrid[-1],y2[1],'.')    
    
#%%
indx = 5
plt.figure(),
# plt.subplot(1,2,1),
plt.plot(timegrid,bridg2d.Z[0,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[0,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],Fcont[0,indx],'.')
plt.plot(bridg2d.timegrid[:],Fnon[0,indx],'.')
plt.plot(bridg2d.timegrid[-1],y2[0],'.')
plt.figure(),
# plt.subplot(1,2,2),
plt.plot(timegrid,bridg2d.Z[1,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[1,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],Fcont[1,indx],'.')
plt.plot(bridg2d.timegrid[:],Fnon[1,indx],'.')
plt.plot(bridg2d.timegrid[-1],y2[1],'.')
#plt.vlines(bridg2d.timegrid[-1],0,4)    



#%%

plt.figure(figsize=(5,8)),
plt.subplot(2,1,1),
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[0],axis=0),'.')
plt.plot(bridg2d.timegrid[:],np.mean(FnonALL[0],axis=0),'-')
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[0],axis=0)-np.std(FcontALL[0],axis=0),'k--')
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[0],axis=0)+np.std(FcontALL[0],axis=0),'k--')
# plt.plot(bridg2d.timegrid[:],np.mean(Fnon[0],axis=0)-np.std(Fnon[0],axis=0),'--',c='grey')
# plt.plot(bridg2d.timegrid[:],np.mean(Fnon[0],axis=0)+np.std(Fnon[0],axis=0),'--',c='grey')
mnstdl = np.mean(FnonALL[0],axis=0)-np.std(FnonALL[0],axis=0)
mnstdup = np.mean(FnonALL[0],axis=0)+np.std(FnonALL[0],axis=0)
plt.fill_between(bridg2d.timegrid,y1=mnstdl,y2 = mnstdup,color='grey', alpha=0.2,linewidth=0)
plt.plot(bridg2d.timegrid[-1],y2[0],'.')

plt.subplot(2,1,2),
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[1,:],axis=0),'.')
plt.plot(bridg2d.timegrid[:],np.mean(FnonALL[1],axis=0),'-')
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[1],axis=0)-np.std(FcontALL[1],axis=0),'k--')
plt.plot(bridg2d.timegrid[:],np.mean(FcontALL[1],axis=0)+np.std(FcontALL[1],axis=0),'k--')
#plt.plot(bridg2d.timegrid[:],np.mean(Fnon[1],axis=0)-np.std(Fnon[1],axis=0),'--',c='grey')
#plt.plot(bridg2d.timegrid[:],np.mean(Fnon[1],axis=0)+np.std(Fnon[1],axis=0),'--',c='grey')
mnstdl = np.mean(FnonALL[1],axis=0)-np.std(FnonALL[1],axis=0)
mnstdup = np.mean(FnonALL[1],axis=0)+np.std(FnonALL[1],axis=0)
plt.fill_between(bridg2d.timegrid,y1=mnstdl,y2 = mnstdup,color='grey', alpha=0.2,linewidth=0)
plt.plot(bridg2d.timegrid[-1],y2[1],'.')    
    
#%%
indx = 13
plt.figure(figsize=(4.5,5.)),
plt.subplot(2,1,1),
plt.plot(timegrid,bridg2d.Z[0,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[0,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],FcontALL[0,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[:],FnonALL[0,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[-1],y2[0],'.')
#plt.figure(),
plt.subplot(2,1,2),
plt.plot(timegrid,bridg2d.Z[1,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[1,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],FcontALL[1,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[:],FnonALL[1,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[-1],y2[1],'.')
#plt.vlines(bridg2d.timegrid[-1],0,4)    



#%%


import matplotlib.gridspec as gridspec
from matplotlib import cm
w = 2.5
Y, X = np.mgrid[-0.5:w:100j, -0.5:w:100j]
XY = np.concatenate( (X.reshape(1,-1), Y.reshape(1,-1)), axis=0 )
F_XY = f(XY)
U = F_XY[0]
V = F_XY[1]
speed = np.sqrt(U**2 + V**2)
fixed_points = np.array([[1.996, 0.004, 1], [0.004, 1.996,  1]])

fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(nrows=2, ncols=3, height_ratios=[1,1])




# Varying color along a streamline
ax1 = fig.add_subplot(gs[:2, 0])
plt.contourf(X, Y, np.log10(speed.reshape(X.shape)), cmap="inferno",levels=150)
#strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color=U.reshape(X.shape), linewidth=2, cmap='autumn')
strm = ax1.streamplot(X, Y, U.reshape(X.shape), V.reshape(X.shape), color='#4f4949', density=[0.75, 0.75])
ax1.plot(fixed_points[0,0], fixed_points[1,0], 'go')
ax1.plot(fixed_points[0,2], fixed_points[1,2], 'X', c='silver')
ax1.plot(fixed_points[0,1], fixed_points[1,1], 'o', c='#4f4949')
ax1.plot(Fnon[0],Fnon[1], c='grey', alpha=0.85,lw=1.8, label='uncontrolled')
for ti in range(0,timegrid.size):
    if ti==timegrid.size-200:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8, label='controlled')
    else:
        ax1.plot(Fcont[0,ti:ti+2],Fcont[1,ti:ti+2], c=cm.viridis(ti/(timegrid.size)),zorder=5,lw=1.8)

#fig.colorbar(strm.lines)

ax1.set_aspect(1)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.set_title('Varying Color')
ax1.set_xticks([-0.5, 0.5, 1.5,2.5])
ax1.set_yticks([-0.5, 0.5, 1.5,2.5])
ax1.set_xlabel(r'x$_1$')
ax1.set_ylabel(r'x$_2$')

handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[::-1], labels[::-1], title=None,
          handletextpad=0.5, columnspacing=3.2,handlelength=0.8, bbox_to_anchor=[-0.53, 0.955],
          loc=2, ncol=1, frameon=True,fontsize = 'small',shadow=None,framealpha =0,edgecolor ='#0a0a0a')

ax1 = fig.add_subplot(gs[0, 1])



indx = 13

plt.plot(timegrid,bridg2d.Z[0,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[0,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],FcontALL[0,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[:],FnonALL[0,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[-1],y2[0],'.')



ax1 = fig.add_subplot(gs[1, 1])
plt.plot(timegrid,bridg2d.Z[1,:,:].T,'grey',alpha=0.5)
plt.plot(timegrid,bridg2d.B[1,:,:].T,'maroon',alpha=0.25)
plt.plot(bridg2d.timegrid[:],FcontALL[1,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[:],FnonALL[1,indx],'-',lw=2.5)
plt.plot(bridg2d.timegrid[-1],y2[1],'.')

plt.tight_layout()
plt.show()