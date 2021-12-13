# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 00:02:39 2021

@author: maout
"""



#
import ot
import numba
from . import score_function_estimators
from . import optimal_transport_reweighting



import time
import numpy as np


class DPFC:
    def __init__(self,t1,t2,y1,y2,f,g,N,k,M,reweight=False, U=None,dens_est='nonparametric',reject=True,plotting=True,kern='RBF',f_true=None,brown_bridge=False):
        """
        Deterministic particle flow control - class initialising function
        t1: starting time point
        t2: end time point
        y1: initial position
        y2: terminal position
        f: drift function handle
        g: diffusion coefficient or function handle 
        N: number of particles/trajectories
        k: discretisation steps within bridge 
        M: number of sparse points for grad log density estimation
        reweight: boolean - determines if reweighting will follow
        U: function, reweighting function to be employed during reweighting: dim_y1 \to 1
        dens_est: density estimation function
                  > 'nonparametric' : non parametric density estimation (this was used in the paper)
                  TO BE ADDED:
                  > 'hermit1' : parametic density estimation empoying hermite polynomials (physiscist's)
                  > 'hermit2' : parametic density estimation empoying hermite polynomials (probabilists's)
                  > 'poly' : parametic density estimation empoying simple polynomials
                  > 'RBF' : parametric density estimation employing radial basis functions
        kern: type of kernel: 'RBF' or 'periodic' (only the 'RBF' was used and gives robust results. Do not use 'periodic' yet!)
        reject: boolean parameter indicating whether non valid backward trajectories will be rejected
        plotting: boolean parameter indicating whether bridge statistics will be plotted
        f_true: in case of Brownian bridge reweighting this is the true forward drift for simulating the forward dynaics
        brown_bridge: boolean,determines if the reweighting concearns contstraint or reweighting with respect to brownian bridge
        """
        self.dim = y1.size # dimensionality of the system
        self.t1 = t1
        self.t2 = t2
        self.y1 = y1
        self.y2 = y2

        
        ##density estimation stuff
        self.kern = kern
        if kern=='periodic':
            kern= 'RBF'
            print('Please do not use periodic kernel yet! For all the numerical experiments RBF was used')
            print('We changed your choice to RBF')
        # DRIFT /DIFFUSION
        self.f = f
        self.g = g #scalar or array
        
        ### PARTICLE DISCRETISATION
        self.N = N        
        
        self.N_sparse = M
        
        self.dt = 0.001 #((t2-t1)/k)
        ### reject unreasonable backward trajectories that do not return to initial condition
        self.reject = reject

        
        
        self.timegrid = np.arange(self.t1,self.t2+self.dt/2,self.dt)
        self.k = self.timegrid.size
        ### reweighting
        self.brown_bridge = brown_bridge
        self.reweight = reweight
        if self.reweight:
          self.U = U
          if self.brown_bridge:
              self.Ztr = np.zeros((self.dim,self.N,self.k)) #storage for forward trajectories with true drift
              self.f_true = f_true

        
        
        
        self.Z = np.zeros((self.dim,self.N,self.k)) #storage for forward trajectories
        self.B = np.zeros((self.dim,self.N,self.k)) #storage for backward trajectories
        self.ln_roD = [] ## storing the estimated forward logarithmic gradients
        
        ##this is 
        self.BPWE = np.zeros((self.dim,self.N,self.timegrid.size))
        self.BPWEmean = np.zeros((self.dim,self.k*self.finer))
        self.BPWEstd = np.zeros((self.dim,self.k*self.finer))
        self.BPWEskew = np.zeros((self.dim,self.k*self.finer))
        self.BPWEkurt = np.zeros((self.dim,self.k*self.finer))
        
        
        #self.forward_sampling() ## we do not really use it but this employs stochastic path sampling
        # TO DO: add option to select between stochastic and deterministic path sampling
        
        self.forward_sampling_Otto()
        ### if a Brownian bridge is used for forward sampling
        if self.reweight and self.brown_bridge:
            self.forward_sampling_Otto_true()

           
        self.density_estimation()
        self.backward_simulation()
        self.reject_trajectories() 
        #self.calculate_true_statistics() ##this is only for Ornstein-Uhlenbeck processes
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
                      W[:,0] = np.exp(self.U(self.Z[:,:,ti]))                    
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
    def f_seperate_true(self,x,t):#plain GP prior
        
        dimi, N = x.shape        
        bnds = np.zeros((dimi,2))
        for ii in range(dimi):
            bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
        sum_bnds = np.sum(bnds)        

        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )        
        gpsi = np.zeros((dimi, N))
        lnthsc = 2*np.std(x,axis=1)   
           
        for ii in range(dimi):            
            gpsi[ii,:]= score_function_multid_seperate(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc,which_dim=ii+1, kern=self.kern)     
        
        return (self.f_true(x,t)-0.5* self.g**2* gpsi)
    
    
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
    
     ###same as forward sampling but without reweighting - this is for bridge reweighting
        ### not for constraint reweighting    
    def forward_sampling_Otto_true(self):
        print('Sampling forward with deterministic particles and true drift...')
        #W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):  
            #print(ti)          
            if ti == 0:
                for di in range(self.dim):
                    self.Ztr[di,:,0] = self.y1[di]
                    #self.Z[di,:,-1] = self.y2[di]   
                    #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
            elif ti==1: #propagate one step with stochastic to avoid the delta function
                #for i in range(self.N):                            #substract dt because I want the time at t-1
                self.Ztr[:,:,ti] = (self.Ztr[:,:,ti-1] + self.dt*self.f_true(self.Ztr[:,:,ti-1],tt-self.dt)+\
                                 (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
            else:                
                self.Ztr[:,:,ti] = ( self.Ztr[:,:,ti-1] + self.dt* self.f_seperate_true(self.Ztr[:,:,ti-1],tt-self.dt) )                
                  
        print('Forward sampling with Otto true is ready!')        
        return 0
    
    
    
    def forward_sampling_Otto(self):
        print('Sampling forward with deterministic particles...')
        W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):  
            print(ti)          
            if ti == 0:
                for di in range(self.dim):
                    self.Z[di,:,0] = self.y1[di]
                    if self.brown_bridge:
                        self.Z[di,:,-1] = self.y2[di]   
                    #self.Z[di,:,0] = np.random.normal(self.y1[di], 0.05, self.N)
            elif ti==1: #propagate one step with stochastic to avoid the delta function
                                           #substract dt because I want the time at t-1
                self.Z[:,:,ti] = (self.Z[:,:,ti-1] + self.dt*self.f(self.Z[:,:,ti-1],tt-self.dt)+\
                                 (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )
            else:                
                self.Z[:,:,ti] = ( self.Z[:,:,ti-1] + self.dt* self.f_seperate(self.Z[:,:,ti-1],tt-self.dt) )
                ###WEIGHT
            if self.reweight == True:
              if ti>0:
                  #print(self.U(self.Z[:,:,ti]))
                  W[:,0] = np.exp(self.U(self.Z[:,:,ti]) ) #-1                   
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
        u_t = np.zeros(grid_x.T.shape)
        
        
        lnthsc1 = 2*np.std(self.B[:,:,ti],axis=1)
        lnthsc2 = 2*np.std(self.Z[:,:,ti],axis=1)
        
  
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            if self.reweight==False or self.brown_bridge==False:
                bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
            else:
                bnds[ii] = [max(np.min(self.Ztr[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Ztr[ii,:,ti]),np.max(self.B[ii,:,ti]))]
                
        if ti<=5 or (ti>= self.k-5):
            if self.reweight==False or self.brown_bridge==False:
                ##for the first 5 timesteps, to avoid numerical singularities just assume gaussian densities
                for di in range(self.dim):
                    mutb = np.mean(self.B[di,:,ti])
                    stdtb = np.std(self.B[di,:,ti])
                    mutz = np.mean(self.Z[di,:,ti])
                    stdtz = np.std(self.Z[di,:,ti])                
                    u_t[di] =  -(grid_x[:,di]- mutb)/stdtb**2 - (  -(grid_x[:,di]- mutz)/stdtz**2 )
            elif self.reweight==True and self.brown_bridge==True:
                for di in range(self.dim):
                    mutb = np.mean(self.B[di,:,ti])
                    stdtb = np.std(self.B[di,:,ti])
                    mutz = np.mean(self.Ztr[di,:,ti])
                    stdtz = np.std(self.Ztr[di,:,ti])                
                    u_t[di] =  -(grid_x[:,di]- mutb)/stdtb**2 - (  -(grid_x[:,di]- mutz)/stdtz**2 )
        elif ti>5:
            ###if point for evaluating control falls out of the region where we have points, clip the points to 
            ###fall within the calculated region - we do not change the position of the point, only the control value will be
            ###calculated with clipped positions 
            bndsb = np.zeros((self.dim,2))
            bndsz = np.zeros((self.dim,2))
            for di in range(self.dim):
                bndsb[di] = [np.min(self.B[di,:,ti]), np.max(self.B[di,:,ti])]
                bndsz[di] = [np.min(self.Z[di,:,ti]), np.max(self.Z[di,:,ti])]            
            
            ###cliping the values of points when evaluating the grad log p
            grid_b = grid_x#np.clip(grid_x, bndsb[0], bndsb[1]) 
            grid_z = grid_x#np.clip(grid_x, bndsz[0], bndsz[1])        
  
            Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
            for di in range(self.dim): 
                score_Bw = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc1,which_dim=di+1, kern=self.kern)(grid_b)
                if self.reweight==False or self.brown_bridge==False: 
                    score_Fw = score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc2,which_dim=di+1, kern=self.kern)(grid_z)
                else:
                    bndsztr = np.zeros((self.dim,2))
                    for ii in range(self.dim):                        
                        bndsztr[di] = [np.min(self.Ztr[di,:,ti]), np.max(self.Ztr[di,:,ti])]  
                    grid_ztr = np.clip(grid_x, bndsztr[0], bndsztr[1])
                    lnthsc3 = 2*np.std(self.Ztr[:,:,ti],axis=1)
                    score_Fw = score_function_multid_seperate(self.Ztr[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc3,which_dim=di+1, kern=self.kern)(grid_ztr)
                
                u_t[di] = score_Bw - score_Fw
            # for di in range(self.dim):  
            #     u_t[di] = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T) \
            #              - score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T)
                     
                    
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