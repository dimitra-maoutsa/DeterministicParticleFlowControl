# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 00:02:39 2021

@author: maout
"""


import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from scipy.spatial.distance import cdist
#
import ot
import numba
### calculate score function from empirical distribution
### uses RBF kernel




def score_function_multid_seperate(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
    
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X across dimension "which_dim" only. 
    When "funct_out == False": computes grad-log at the sample points.
    When "funct_out == True" : return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
    (For estimation across all dimensions simultaneously see score_function_multid_seperate_all_dims )
    Input: X: N samples from the density (N x dim), where dim>=2 the dimensionality of the system
           Z: inducing points points (M x dim)
           func_out : Boolean, True returns function, if False return grad-log-p on data points                    
           l: lengthscale of rbf kernel (scalar or vector of size dim)
           C: weighting constant (leave it at default value to avoid unreasonable contraction of deterministic trajectories)          
           which: return 1: grad log p(x) 
           which_dim: which gradient of log density we want to compute (starts from 1 for the 0-th dimension)
    Output: res1: array with density along the given dimension  N_s x 1 or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1
    
    """
    if kern=='RBF':       
        
        def K(x,y,l,multil=False):
            if multil:                
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))            
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape            
            diffs = x[:,None]-y               
            redifs = np.zeros((1*N,N))
            ii = which_dim -1            
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs            
     
        def grdy_K(x,y): # gradient with respect to the second argument
            N,dim = x.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            ii = which_dim -1              
            redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
            return -redifs            
                
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs            
     
    if isinstance(l, (list, tuple, np.ndarray)):
       ### for different lengthscales for each dimension 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       multil = True ##just a boolean to keep track if l is scalar or vector
       
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True)        
        
    else:
        multil = False
        K_xz = K(X,Z,l,multil=False) 
        Ks = K(Z,Z,l,multil=False)            
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=False)
    sumgradx_K = np.sum(gradx_K ,axis=0)
    if func_out==False: #For evaluation at data points!!!
        ### evaluatiion at data points
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:           
        #### For functional output!!!! 
        if multil:                            
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
        
        else:
            K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))           

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K

    
    return res1
