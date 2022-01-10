# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 12:18:45 2022

@author: maout
"""


import time
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Union
# GPU + autodiff library
from torch.autograd import grad
import numba
import math
from functools import reduce

#%%

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device

#%%
  


class RBF:
    def __init__(self, length_scale: Union[float, torch.tensor, np.ndarray]=1.0, signal_variance: float=1.0, device: Union[bool,str]=None, multil: Union[bool, None]=False) -> None:

        # initialize parameters
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.length_scale = torch.tensor(length_scale, dtype=torch.float32, device=self.device,
            requires_grad=True)        
        self.signal_variance = torch.tensor(signal_variance, dtype=torch.float32, device=self.device,
            requires_grad=True)
        self.multil = torch.tensor(multil, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.multil:
            ##expand dimensions of lengtscale vector to enable broadcasting
            self.length_scale = self.length_scale[None, None,  :]
        self.K_data = torch.tensor(0, dtype=torch.float32, device=self.device, requires_grad=False)

        

    def Kernel(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:

        if not torch.is_tensor(X):
          # convert inputs to pytorch tensors if not already pytorched
          X = torch.tensor(X, dtype=torch.float32, device=self.device) 
          N, D = X.shape       
        if Y is None:
            Y = X
        elif not torch.is_tensor(Y):
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        M, _ = Y.shape
        # Re-indexing
        X_i = X[:, None, :] # shape (N, D) -> (N, 1, D)
        Y_j = Y[None, :, :] # shape (M, D) -> (1, M, D)
        
        if not self.multil: ##if a single lengthscale is provided
            
            sqd     = torch.sum( (X_i - Y_j)**2, 2)         # |X_i - Y_j|^2 # (N, M, D)
            # Divide by length scale            
            sqd  = torch.div(sqd, self.length_scale**2)
            K    = torch.exp( -0.5* sqd )               
        else:          
          sqd1     = torch.div( (X_i - Y_j)**2, self.length_scale**2) 
          sqd     = torch.sum( sqd1, 2)  
          K    = torch.exp( -0.5* sqd )

        K   = torch.mul(self.signal_variance, K) # Signal Variance
        self.K_data = K
        return K#.detach().to_numpy()

    def gradient_X(self, X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:
        N, D = X.shape    
        M,_ = Y.shape
        diffs = X[:,None]-Y            
        #if self.multil:            
        redifs = torch.div(diffs, self.length_scale**2)
        redifs = torch.einsum( 'ijk,ij->ijk', redifs, self.K_data)
        #redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
        #else:
            #redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)
              
        return redifs

        

    def gradient_X2(self, X):

        return None

    def gradient_XX(self,X: np.ndarray, Y: Union[bool, np.ndarray]=None) -> torch.tensor:

        # Convert to tensor that requires Grad
        X = torch.tensor(self.length_scale, dtype=torch.float32, device=self.device,requires_grad=True)

        if Y is None:
            Y = X
        else:
            Y = torch.tensor(Y, dtype=torch.float32, device=self.device, requires_grad=True)
        # compute the gradient kernel w.r.t. to the two inputs
        J = grad(self.__call__(X, Y))

        return J

    def gradient_XX2(self, X, Y=None):

        return None
    
    
    
#%%
        
def Knp(x,y,l,multil=False):
    if multil:   
        res = np.ones((x.shape[0],y.shape[0]))                
        for ii in range(len(l)):             
            tempi = np.zeros((x[:,ii].size, y[:,ii].size ))            
            tempi = cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),metric='sqeuclidean')
            res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
            
        return res
    else:
        tempi = np.zeros((x.shape[0], y.shape[0] ))
        
        tempi = cdist(x, y,'sqeuclidean') #this sets into the array tempi the cdist result
        return np.exp(-0.5*tempi/(l*l))
    
    
    
#%%
        
def grdx_K_all(x,y,l,multil=False): #gradient with respect to the 1st argument - only which_dim
      N,dim = x.shape    
      M,_ = y.shape
      diffs = x[:,None]-y                         
      redifs = np.zeros((1*N,M,dim))
      for ii in range(dim):          
      
          if multil:
              redifs[:,:,ii] = np.multiply(diffs[:,:,ii],Knp(x,y,l,True))/(l[ii]*l[ii])   
          else:
              redifs[:,:,ii] = np.multiply(diffs[:,:,ii],Knp(x,y,l))/(l*l)            
      return redifs




#%% torched score function multidimensional
      
  
def torched_score_function_multid_seperate_all_dims(X: np.ndarray, Z: np.ndarray, 
                                                    l: Union[float, torch.tensor, np.ndarray]=1.0,
                                                    func_out: Union[bool, None]=False, 
                                                    C: Union[float, None]=0.001,
                                                    kern: Union[None, str]  ='RBF') -> torch.tensor:
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X for all dimensions simultaneously. 
    Implemented with pytorch.
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
    
    Parameters
    -----------
            X: N x dim array,
               N samples from the density (N x dim), where dim>=2 the 
               dimensionality of the system.
            Z: M x dim array,
              inducing points points (M x dim).
            func_out : Boolean, 
                      True returns function, 
                      if False returns grad-log-p evaluated on samples X.                    
            l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
            C: float,
              weighting constant 
              (leave it at default value to avoid unreasonable contraction 
              of deterministic trajectories).
            kern: string,
                options:
                    - 'RBF': radial basis function/Gaussian kernel  
                    - 'periodic': periodic, not functional yet.           
           
    Returns
    -------
        res1: array with logarithmic gradient of the density  N_s x dim or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    """   
    
    

    M, dim = Z.shape
    if not torch.is_tensor(X):
          # convert inputs to pytorch tensors if not already pytorched
          X = torch.tensor(X, dtype=torch.float32, device=DEVICE) 
          N, D = X.shape       
          if Z is None:
              Z = X
          elif not torch.is_tensor(Z):
              Z = torch.tensor(Z, dtype=torch.float32, device=self.device)

    if isinstance(l, (list, tuple, np.ndarray)):
          multil = True
          ### for different lengthscales for each dimension         
          
          # pytorched
          K_instance = RBF(length_scale=l, multil=True, device=DEVICE) ##instance of kernel object - non-evaluated
          K_xz = K_instance.Kernel(X, Z)#.detach().numpy()
          
          K_instancez = RBF(length_scale=l, multil=True, device=DEVICE) ##instance of kernel object - non-evaluated
          K_s = K_instancez.Kernel(Z, Z)#.detach().numpy()  
       
    else:
          multil = False
          K_instance = RBF(length_scale=l, multil=False, device=DEVICE) ##instance of kernel object - non-evaluated
          K_xz = K_instance.Kernel(X, Z)#.detach().numpy()
        
          K_instancez = RBF(length_scale=l, multil=False, device=DEVICE) ##instance of kernel object - non-evaluated
          K_s = K_instancez.Kernel(Z, Z)#.detach().numpy()
           
        
          
    Ksinv = torch.linalg.inv(K_s+ 1e-3 * torch.eye(M))
    A = torch.t(K_xz) @ K_xz ##matrix multilication  
    #compute the gradient of the X x Z kernel       
    gradx_K = -K_instance.gradient_X(X, Z) #shape: (N,M,dim)
    ##last axis will have the gradient for each dimension ### shape (M, dim)
    sumgradx_K = torch.sum(gradx_K ,axis=0) 
    
    if func_out==False: #if output wanted is evaluation at data points
        
        # res1 = np.zeros((N, dim))    
        # ### evaluatiion at data points
        # for di in range(dim):
        #     res1[:,di] = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K[:,di]
        
        
        res1 = -K_xz @ torch.linalg.inv( C*torch.eye(M, M) + Ksinv @ A + 1e-3 * torch.eye(M))@ Ksinv@sumgradx_K        
        
        #res1 = np.einsum('ik,kj->ij', -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv, sumgradx_K)
        
        
    else:           
        #### for function output 
        
        if multil:      
            if kern=='RBF':  
                def K_sz(x):
                    if not torch.is_tensor(x):
                          # convert inputs to pytorch tensors if not already pytorched
                          x = torch.tensor(x, dtype=torch.float32, device=DEVICE) 
                    X_i = x[:, None, :] # shape (n, D) -> (n, 1, D)
                    Z_j = Z[None, :, :] # shape (M, D) -> (1, M, D)          
                    sqd1     = torch.div( (X_i - Z_j)**2, l**2) 
                    sqd     = torch.sum( sqd1, 2)  
                    K_sz    = torch.exp( -0.5* sqd )

                    #K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])        
                    return K_sz
            #elif kern=='periodic':
                  #K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
        else:
            if kern=='RBF':
                  def K_sz(x):
                      if not torch.is_tensor(x):
                          # convert inputs to pytorch tensors if not already pytorched
                          x = torch.tensor(x, dtype=torch.float32, device=DEVICE) 
                      X_i = x[:, None, :] # shape (n, D) -> (n, 1, D)
                      Z_j = Z[None, :, :] # shape (M, D) -> (1, M, D)
                      sqd     = torch.sum( (X_i - Z_j)**2, 2)         # |X_i - Y_j|^2 # (N, M, D)
                      # Divide by length scale            
                      sqd  = torch.div(sqd, l**2)
                      K_sz    = torch.exp( -0.5* sqd )
                      #K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
                      return K_sz
            #elif kern=='periodic':
                #K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            
        
        res1 = lambda x: K_sz(x) @ (-torch.linalg.inv(C*torch.eye(M, M) + Ksinv @ A + 1e-3 * torch.eye(M)) ) @ Ksinv@sumgradx_K
        
    
    return res1   ### shape out N x dim

#%% numpyed score function multidimensional
    

def my_cdist(r,y, output,dist='euclidean'):
    """   
    Fast computation of pairwise distances between data points in r and y matrices.
    Stores the distances in the output array.
    Available distances: 'euclidean' and 'seucledian'
    
    Parameters
    ----------
    r : NxM array
        First set of N points of dimension M.
    y : N2xM array
        Second set of N2 points of dimension M.
    output : NxN2 array
        Placeholder for storing the output of the computed distances.
    dist : type of distance, optional
        Select 'euclidian' or 'sqeuclidian' for Euclidian or squared Euclidian
        distances. The default is 'euclidean'.

    Returns
    -------
    None. (The result is stored in place in the provided array "output").

    """
    N, M = r.shape
    N2, M2 = y.shape
    #assert( M == M2, 'The two inpus have different second dimention! Input should be N1xM and N2xM')
    if dist == 'euclidean':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2            
                output[i,j] = math.sqrt(tmp)
    elif dist == 'sqeuclidean':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2            
                output[i,j] = tmp   
    elif dist == 'l1':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2          
                output[i,j] = math.sqrt(tmp)   
    return 0
#%%
    

def score_function_multid_seperate_all_dims(X,Z,func_out=False, C=0.001,kern ='RBF',l=1):
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X for all dimensions simultaneously. 
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
    
    Parameters
    -----------
            X: N x dim array,
               N samples from the density (N x dim), where dim>=2 the 
               dimensionality of the system.
            Z: M x dim array,
              inducing points points (M x dim).
            func_out : Boolean, 
                      True returns function, 
                      if False returns grad-log-p evaluated on samples X.                    
            l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
            C: float,
              weighting constant 
              (leave it at default value to avoid unreasonable contraction 
              of deterministic trajectories).
            kern: string,
                options:
                    - 'RBF': radial basis function/Gaussian kernel  
                    - 'periodic': periodic, not functional yet.           
           
    Returns
    -------
        res1: array with logarithmic gradient of the density  N_s x dim or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    """
    
    if kern=='RBF':
        """
        #@numba.njit(parallel=True,fastmath=True)
        def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
            if multil:                                        
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                    ##puts into tempi the cdist result                    
                    my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                res = np.exp(-tempi/(2*l*l))
            return 0
        """
        
        def K(x,y,l,multil=False):
            if multil:   
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                    ##puts into tempi the cdist result
                    my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi,'sqeuclidean')
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    
                return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ))
                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                return np.exp(-tempi/(2*l*l))
            
        #njit
        def grdx_K_all(x,y,l,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape    
            M,_ = y.shape
            diffs = x[:,None]-y                         
            redifs = np.zeros((1*N,M,dim))
            for ii in range(dim):          
            
                if multil:
                    redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
                else:
                    redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
            
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            #_,dim = x.shape 
            #M,_ = y.shape
            diffs = x[:,None]-y                         
            #redifs = np.zeros((1*N,M))
            ii = which_dim -1            
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])  
                
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
     
        
            #############################################################################
    elif kern=='periodic': ###############################################################################################
        ### DO NOT USE "periodic" yet!!!!!!!
      ###periodic kernel
        ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
        
        ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
        ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
      #per = 2*np.pi ##period of the kernel
      
      def K(x,y,l,multil=False):
        
        if multil:       
          res = np.ones((x.shape[0],y.shape[0]))                
          for ii in range(len(l)): 
              #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
              ##puts into tempi the cdist result
              #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
              #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
              res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
          return -res
        else:
            #tempi = np.zeros((x.shape[0], y.shape[0] ))
            ##puts into tempi the cdist result
            #my_cdist(x, y, tempi,'l1')
            #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
            res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            return res
        
      def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
          #N,dim = x.shape            
          diffs = x[:,None]-y             
          #redifs = np.zeros((1*N,N))
          ii = which_dim -1          
          if multil:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*K(x,y,l,True),diffs[:,:,ii] ),np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,np.cos( np.abs(diffs[:,:,ii])  / 2) ) , (l[ii]*l[ii]* np.abs(diffs[:,:,ii]))  ) 
          else:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*diffs[:,:,ii],np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,K(x,y,l) ),np.cos( np.abs(diffs[:,:,ii]) / 2) ) ,(l*l* np.abs(diffs[:,:,ii])) )           
          return -redifs

    dim = X.shape[1]

    if isinstance(l, (list, tuple, np.ndarray)):
       multil = True
       ### for different lengthscales for each dimension 
       #K_xz =  np.ones((X.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(X,Z,l,K_xz,multil=True) 
       
       #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(Z,Z,l,Ks,multil=True) 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz    
              
       gradx_K = -grdx_K_all(X,Z,l,multil=True) #-
       gradxK = np.zeros((X.shape[0],Z.shape[0],dim))
       for ii in range(dim):
           gradxK[:,:,ii] = -grdx_K(X,Z,l,multil=True,which_dim=ii+1)
       
       np.testing.assert_allclose(gradxK, gradx_K) 
    else:
        multil = False
        
        K_xz = K(X,Z,l,multil=False) 
        
        Ks = K(Z,Z,l,multil=False)    
        
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        
        gradx_K = -grdx_K_all(X,Z,l,multil=False)   #shape: (N,M,dim)
    sumgradx_K = np.sum(gradx_K ,axis=0) ##last axis will have the gradient for each dimension ### shape (M, dim)
    
    if func_out==False: #if output wanted is evaluation at data points
        
        # res1 = np.zeros((N, dim))    
        # ### evaluatiion at data points
        # for di in range(dim):
        #     res1[:,di] = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K[:,di]
        
        
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K        
        
        #res1 = np.einsum('ik,kj->ij', -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv, sumgradx_K)
        
        
    else:           
        #### for function output 
        if multil:      
            if kern=='RBF':
                 
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])        
                
            elif kern=='periodic':
                K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
        else:
            if kern=='RBF':
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            elif kern=='periodic':
                K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
        
            
        
    
    return res1   ### shape out N x dim


#%% test the two versions of the score functions
DEVICE = set_device()
dtype = torch.float  
sigma = 0.5
mean_1 = 1
sigma_1 = sigma
mean_2 = 2
sigma_2 = 1*sigma
N = 500
X2 = np.array([[np.random.normal(loc=mean_1,scale=sigma_1,size=N),np.random.normal(loc=mean_2,scale=sigma_2,size=N)]]).reshape(N,2) #np.hstack((X,Y))

M = 20 ##inducing point number
dimi = 2
bnds = np.zeros((dimi, 2))
for ii in range(dimi):
    bnds[ii] = [np.min(X2[ii, :]), np.max(X2[ii, :])]

Z = np.array([np.random.uniform(low=bnd[0], high=bnd[1], size=M) for bnd in bnds])
## kernel lengthscale
lengthsc = 2* np.std(X2)


##fig_1 = plt.figure()
#fig_2 = plt.figure()



ln1 = score_function_multid_seperate_all_dims(X2, Z.T, func_out=False, C=0.001, kern='RBF', l=lengthsc)
ln_torch = torched_score_function_multid_seperate_all_dims(torch.tensor(X2), 
                                                           torch.tensor(Z.T),
                                                           func_out=False, 
                                                           C=0.001, kern='RBF', l=lengthsc)

#ax_1_1 = fig_1.add_subplot(1,2,1) 
#ax_1_1.plot(X2[:,0],ln1[:,0],'r.',label='efficient')
#ax_1_1.plot(X2[:,0], -(X2[:,0]-mean_1)/sigma_1**2,'k.')

#ax_1_2 = fig_1.add_subplot(1,2,2) 
#ax_1_2.plot(X2[:,0],ln_torch[:,0].detach().numpy(),'r.',label='efficient')
#ax_1_2.plot(X2[:,0], -(X2[:,0]-mean_1)/sigma_1**2,'k.')



#ax_2_1 = fig_2.add_subplot(1,2,1) 
#ax_2_1.plot(X2[:,1],ln1[:,1],'r.',label='efficient')
#ax_2_1.plot(X2[:,1], -(X2[:,1]-mean_2)/sigma_2**2,'k.')

#ax_2_2 = fig_2.add_subplot(1,2,2) 
#ax_2_2.plot(X2[:,1],ln_torch[:,1].detach().numpy(),'r.',label='efficient')
#ax_2_2.plot(X2[:,1], -(X2[:,1]-mean_2)/sigma_2**2,'k.')


#plt.figure()
#plt.imshow(ln1-ln_torch.detach().numpy(),aspect= 0.0020)
#plt.colorbar()
np.testing.assert_allclose(ln1, ln_torch.detach().numpy(), rtol=1e-06)


#%% test functional output version



ln2 = score_function_multid_seperate_all_dims(X2, Z.T, func_out=True, C=0.001, kern='RBF', l=lengthsc)
ln_torch2 = torched_score_function_multid_seperate_all_dims(torch.tensor(X2), 
                                                           torch.tensor(Z.T),
                                                           func_out=True, 
                                                           C=0.001, kern='RBF', l=lengthsc)


np.testing.assert_allclose(ln2(Z.T), ln_torch2(Z.T).detach().numpy(), rtol=1e-06)
# plt.figure()
# plt.imshow(ln2(Z.T)-ln_torch2(Z.T).detach().numpy(),aspect= 0.0020)
# plt.colorbar()