# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 04:31:59 2021

@author: maout
"""


import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import pinv
from functools import reduce
from scipy.stats import gamma,norm,dweibull,tukeylambda,skewnorm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist
import time
from scipy.spatial.distance import cdist
def score_function_multid_seperate2(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
    
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
