# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:15:09 2021

@author: maout
"""



###Systematic multi-N inhomogeneous Kuramoto

import numpy as np


#from matplotlib import pyplot as plt

from scipy.spatial.distance import cdist
import time
import ot
import numba
import math
import random
import sys
import pickle
from functools import reduce

@numba.njit(parallel=True,fastmath=True)
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
    None. (The result is stored in place in the input array output).

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
        #@numba.njit(parallel=True,fastmath=True)
        def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
            if multil:         
                #print('here')
                #res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                    ##puts into tempi the cdist result
                    #print(x[:,ii:ii+1].shape)
                    my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                #return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)
                #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                res = np.exp(-tempi/(2*l*l))
            return 0
        
        def K(x,y,l,multil=False):
            if multil:         
                #print('here')
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                    ##puts into tempi the cdist result
                    my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi,'sqeuclidean')
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ))
                #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                return np.exp(-tempi/(2*l*l))
            #return np.exp(-(x-y.T)**2/(2*l*l))
            #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
        #@njit
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape            
            diffs = x[:,None]-y                         
            redifs = np.zeros((1*N,N))
            ii = which_dim -1
            #print('diffs:',diffs)
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
        #@njit        
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y          
            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs
            #return np.multiply((K(x,y)),(np.power(x[:,None]-y,2)-l**2))/l**4
            #############################################################################
    elif kern=='periodic': ###############################################################################################
      ###periodic kernel
        ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
        
        ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
        ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
      per = 2*np.pi ##period of the kernel
      #l = 0.5
      def K(x,y,l,multil=False):
        
        if multil:          
          #print('here')
          res = np.ones((x.shape[0],y.shape[0]))                
          for ii in range(len(l)): 
              tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
              ##puts into tempi the cdist result
              #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
              #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
              res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
          return -res
        else:
            tempi = np.zeros((x.shape[0], y.shape[0] ))
            ##puts into tempi the cdist result
            #my_cdist(x, y, tempi,'l1')
            #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
            res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            return res
        
      def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
          N,dim = x.shape            
          diffs = x[:,None]-y   
          #print('diffs:',diffs)
          redifs = np.zeros((1*N,N))
          ii = which_dim -1
          #print(ii)
          if multil:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*K(x,y,l,True),diffs[:,:,ii] ),np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,np.cos( np.abs(diffs[:,:,ii])  / 2) ) , (l[ii]*l[ii]* np.abs(diffs[:,:,ii]))  ) 
          else:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*diffs[:,:,ii],np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,K(x,y,l) ),np.cos( np.abs(diffs[:,:,ii]) / 2) ) ,(l*l* np.abs(diffs[:,:,ii])) )           
          return -redifs



    if isinstance(l, (list, tuple, np.ndarray)):
       ### for different lengthscales for each dimension 
       #K_xz =  np.ones((X.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(X,Z,l,K_xz,multil=True) 
       K_xz = K(X,Z,l,multil=True) 
       #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(Z,Z,l,Ks,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       multil = True
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True) #-
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
    #print( sumgradx_K.shape )
    if func_out==False: #if output wanted is evaluation at data points
        ### evaluatiion at data points
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:           
        #### for function output 
        if multil:                
            #res = np.ones((x.shape[0],y.shape[0]))                
            #for ii in range(len(l)): 
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
        
                
            elif kern=='periodic':
                K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
            
            #return K_sz
        else:
            if kern=='RBF':
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            elif kern=='periodic':
                K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            #return K_sz

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K


    
    return res1





def score_function_multid_seperate_all_dims(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1):
    """
    returns function psi(z)
    Input: X: N observations
           Z: sparse points
           func_out : Boolean, True returns function if False return grad-log-p on data points                    
           l: lengthscale of rbf kernel
           C: weighting constant           
           which: return 1: grad log p(x) 
           
    Output: psi: array with density along the given dimension N or N_s x 1
    
    """
    
    if kern=='RBF':
        #l = 1 # lengthscale of RBF kernel
        #@numba.njit(parallel=True,fastmath=True)
        def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
            if multil:         
                #print('here')
                #res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                    ##puts into tempi the cdist result
                    #print(x[:,ii:ii+1].shape)
                    my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                #return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)
                #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                res = np.exp(-tempi/(2*l*l))
            return 0
        
        def K(x,y,l,multil=False):
            if multil:         
                #print('here')
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                    ##puts into tempi the cdist result
                    my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi,'sqeuclidean')
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    ##res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ))
                #return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                return np.exp(-tempi/(2*l*l))
            #return np.exp(-(x-y.T)**2/(2*l*l))
            #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
        #@njit
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
            #return -(1./(l*l))*(x-y.T)*K(x,y)
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape 
            M,_ = y.shape
            diffs = x[:,None]-y                         
            redifs = np.zeros((1*N,M))
            ii = which_dim -1
            #print('diffs:',diffs)
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])  
                #print(redifs.shape)
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
     
        
            #############################################################################
    elif kern=='periodic': ###############################################################################################
      ###periodic kernel
        ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
        
        ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
        ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
      per = 2*np.pi ##period of the kernel
      #l = 0.5
      def K(x,y,l,multil=False):
        
        if multil:          
          #print('here')
          res = np.ones((x.shape[0],y.shape[0]))                
          for ii in range(len(l)): 
              tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
              ##puts into tempi the cdist result
              #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
              #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
              res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
          return -res
        else:
            tempi = np.zeros((x.shape[0], y.shape[0] ))
            ##puts into tempi the cdist result
            #my_cdist(x, y, tempi,'l1')
            #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
            res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            return res
        
      def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
          N,dim = x.shape            
          diffs = x[:,None]-y   
          #print('diffs:',diffs)
          redifs = np.zeros((1*N,N))
          ii = which_dim -1
          #print(ii)
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
       K_xz = K(X,Z,l,multil=True) 
       #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(Z,Z,l,Ks,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz    
              
       gradx_K = -grdx_K_all(X,Z,l,multil=True) #-
       gradxK = np.zeros((X.shape[0],Z.shape[0],dim))
       for ii in range(dim):
           gradxK[:,:,ii] = -grdx_K(X,Z,l,multil=True,which_dim=ii+1)
       # if not(Test_p == 'None'):
       #     K_sz = K(Test_p,Z,l,multil=True)
       np.testing.assert_allclose(gradxK, gradx_K) 
    else:
        multil = False
        
        K_xz = K(X,Z,l,multil=False) 
        
        Ks = K(Z,Z,l,multil=False)    
        
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        
        gradx_K = -grdx_K_all(X,Z,l,multil=False)   #shape: (N,M,dim)
    sumgradx_K = np.sum(gradx_K ,axis=0) ##last axis will have the gradient for each dimension ### shape (M, dim)
    #print( sumgradx_K.shape )
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
            #res = np.ones((x.shape[0],y.shape[0]))                
            #for ii in range(len(l)): 
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
        
                
            elif kern=='periodic':
                K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
            
            #return K_sz
        else:
            if kern=='RBF':
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            elif kern=='periodic':
                K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            #return K_sz

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
        # res1 = np.zeros((N, dim))
        # for di in range(dim):
        #     res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K[:,di]
        
            
        #np.testing.assert_allclose(res2, res1)
    
    return res1   ### shape out N x dim

#%%

class BRIDGE_ND_reweight:
    def __init__(self,t1,t2,y1,y2,f,g,N,k,M,reweight=False, U=1,dens_est='nonparametric',reject=True,plotting=True,kern='RBF',dt=0.001):
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
        dt: integration time step [default: 0.001]
        """
        self.dim = y1.shape[0] # dimensionality of the problem
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
        self.k = k
        self.N_sparse = M
        
        self.dt = dt#0.001 #((t2-t1)/k)
        
        ### reweighting
        self.reweight = reweight
        if self.reweight:
          self.U = U

        ### reject
        self.reject = reject

        
        self.finer = 1#200 #discetasation ratio between numerical BW solution and particle bridge solution
        self.timegrid = np.arange(self.t1,self.t2+self.dt/2,self.dt)
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
        
        print('start forward')
        #self.forward_sampling()
        self.forward_sampling_Otto()
        # plt.figure(figsize=(6,4)),plt.plot(self.Z[0].T,self.Z[1].T,alpha=0.3);
        # plt.plot(self.y1[0],self.y1[1],'go')
        # plt.plot(self.y2[0],self.y2[1],'ro')
        # plt.show()

           
        #self.density_estimation()
        self.backward_simulation()
        #self.reject_trajectories() 
        #self.calculate_true_statistics()
        #if plotting:
        #    self.plot_statistics()
        
    def forward_sampling(self): 
        print('Sampling forward...')
        W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):

            if ti == 0:
                self.Z[:,:,0] = self.y1
                #self.Z[1,:,0] = self.y1[1]
            else:
                for i in range(self.N):
                    #self.Z[:,i,:] = sdeint.itoint(self.f, self.g, self.Z[i,0], self.timegrid)[:,0] 
                    self.Z[:,i,ti] = ( self.Z[:,i,ti-1] + self.dt* self.f(self.Z[:,i,ti-1]) + \
                                      (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,)) )% (2*np.pi)
        
                ###WEIGHT
                if self.reweight == True:
                  if ti>0:
                      W[:,0] = np.exp(1*self.dt*self.U(self.Z[:,:,ti]))                    
                      W = W/np.sum(W)
                      
                      ###REWEIGHT                    
                      #Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                      #P = Tstar *N
                      # print(Tstar.shape)
                      # print(X.shape)
                      #self.Z[:,:,ti] = (  (self.Z[:,:,ti])@Tstar  )% (2*np.pi)
                      M = ot.dist(self.Z[:,:,ti].T, self.Z[:,:,ti].T)
                      M /= M.max()
                      a = W[:,0]
                      b =  np.ones_like(W[:,0])/self.N
                      T2 = ot.emd(a, b, M)
                      #T2 = ot.sinkhorn(a, b, M, 0.1)
                      #T2 = ot.bregman.sinkhorn_epsilon_scaling(a, b, M, 0.01)
                      #T2 = ot.bregman.sinkhorn_stabilized(a, b, M, 0.001)
                      #T2 = ot.optim.cg(a, b, M, reg, fi, dfi, verbose=False)
                      self.Z[:,:,ti] = (self.N*self.Z[:,:,ti]@T2) % (2*np.pi)
                
        #for di in range(self.dim):
        #  self.Z[di,:,-1] = self.y2[di]
        print('Forward sampling done!')
        return 0
    
    ### effective forward drift - estimated seperatelly for each dimension
    def f_seperate(self,x,t):#plain GP prior
        
        dimi, N = x.shape        
        bnds = np.zeros((dimi,2))
        for ii in range(dimi):
            bnds[ii] = [np.min(x[ii,:]),np.max(x[ii,:])]
        sum_bnds = np.sum(bnds)
        

        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )        
        
        lnthsc = 2*np.std(x,axis=1)    
        # gpsi2 = np.zeros((dimi, N))   
        # for ii in range(dimi):            
        #     gpsi2[ii,:]= score_function_multid_seperate(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc,which_dim=ii+1, kern=self.kern)     
        gpsi = score_function_multid_seperate_all_dims(x.T,Sxx.T,False,C=0.001,which=1,l=lnthsc, kern=self.kern).T     
        #
        #np.testing.assert_allclose(gpsi, gpsi2)
        return (self.f(x,t)-0.5* self.g**2* gpsi)
    
    
    def forward_sampling_Otto(self):
        print('Sampling forward with deterministic particles...')
        W = np.ones((self.N,1))/self.N
        for ti,tt in enumerate(self.timegrid):  
            print(ti)          
            if ti == 0:
                # for di in range(self.dim):
                #     self.Z[di,:,0] = self.y1[di]   
                self.Z[:,:,0] = np.random.multivariate_normal(self.y1, np.eye(self.dim)*self.g/2, self.N).T
            elif ti==1: #propagate one step with stochastic to avoid the delta function
                #for i in range(self.N):                            #substract dt because I want the time at t-1
                self.Z[:,:,ti] = (self.Z[:,:,ti-1] + self.dt*self.f(self.Z[:,:,ti-1],tt-self.dt)+\
                                 (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )% (2*np.pi)
            else:
                
                self.Z[:,:,ti] = ( self.Z[:,:,ti-1] + self.dt* self.f_seperate(self.Z[:,:,ti-1],tt-self.dt) )% (2*np.pi)
                ###WEIGHT
            if self.reweight == True:
              if ti>0:
                  W[:,0] = np.exp(self.U(self.Z[:,:,ti])) #-1 
                  #Neff = 1/np.sum(W[:,0]**2)
                  #print('Neff:', Neff)
                  #print(W[:,0])
                  W = W/np.sum(W)       
                  #Neff = 1/np.sum(W[:,0]**2)
                  #print('Neff:', Neff)
                  #print(W[:,0])
                  #print('-----')
                  ###REWEIGHT    
                  #start = time.time()
                  #Tstar = reweight_optimal_transport_multidim(self.Z[:,:,ti].T,W)
                  #print(Tstar)
                  # if ti ==3:
                  #     stop = time.time()
                  #     print('Timepoint: %d needed '%ti, stop-start)
                  #P = Tstar *N
                  # print(Tstar.shape)
                  # print(X.shape)
                  #self.Z[:,:,ti] = ((self.Z[:,:,ti])@Tstar )% (2*np.pi) ##### 
                  M = ot.dist(self.Z[:,:,ti].T, self.Z[:,:,ti].T)
                  M /= M.max()
                  a = W[:,0]
                  b =  np.ones_like(W[:,0])/self.N
                  T2 = ot.emd(a, b, M,numItermax=1000000)
                  #T2 = ot.sinkhorn(a, b, M, 0.1)
                  #T2 = ot.bregman.sinkhorn_epsilon_scaling(a, b, M, 0.01)
                  #T2 = ot.bregman.sinkhorn_stabilized(a, b, M, 0.001)
                  #T2 = ot.optim.cg(a, b, M, reg, fi, dfi, verbose=False)
                  self.Z[:,:,ti] = (self.N*self.Z[:,:,ti]@T2) % (2*np.pi)
        
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
        #print(bnds)
        #print(np.min(self.B[:,:,rev_ti]))
        #print(np.max(self.B[:,:,rev_ti]))
        #print('-------')
        
        #sparse points
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
        
        # for di in range(self.dim):     
        #     #estimate density from forward (Z) and evaluate at current postitions of backward particles (B)       
        #     grad_ln_ro[di,:] = score_function_multid_seperate(self.Z[:,:,rev_t].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(self.B[:,:,rev_ti].T)
            #print(grad_ln_ro[:,:3])           
        grad_ln_ro = (score_function_multid_seperate_all_dims(self.Z[:,:,rev_t].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc, kern=self.kern)(self.B[:,:,rev_ti].T) ).T    
        #np.testing.assert_allclose(grad_ln_ro2.T, grad_ln_ro)
        return grad_ln_ro 


    def bw_density_estimation(self, ti, rev_ti):
        #grad_ln_b = np.zeros((self.dim,self.N))
        lnthsc = 2*np.std(self.B[:,:,rev_ti],axis=1)
        #print(ti, rev_ti, rev_ti-1)
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,rev_ti-1]),np.min(self.B[ii,:,rev_ti])),min(np.max(self.Z[ii,:,rev_ti-1]),np.max(self.B[ii,:,rev_ti]))]
        #sparse points
        #print(bnds)
        sum_bnds = np.sum(bnds)
        #if np.isnan(sum_bnds) or np.isinf(sum_bnds):
        #  plt.figure(figsize=(6,4)),plt.plot(self.B[0].T,self.B[1].T,alpha=0.3);
        #  plt.plot(self.y1[0],self.y1[1],'go')
        #  plt.plot(self.y2[0],self.y2[1],'ro')
        #  plt.show()
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
        
        # for di in range(self.dim):            
        #     grad_ln_b[di,:] = score_function_multid_seperate(self.B[:,:,rev_ti].T,Sxx.T,func_out= False,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)#(self.B[:,:,-ti].T)
        grad_ln_b = score_function_multid_seperate_all_dims(self.B[:,:,rev_ti].T,Sxx.T,func_out= False,C=0.001,which=1,l=lnthsc, kern=self.kern).T#(self.B[:,:,-ti].T)
        
        return grad_ln_b # this should be function
    
    
    def backward_simulation(self):   
        
        for ti,tt in enumerate(self.timegrid[:-1]): 
            W = np.ones((N,1))/N           
            if ti==0:
                for di in range(self.dim):
                    self.B[di,:,-1] = self.Z[di,:,-1]#self.y2[di]                
            else:
                #tti = np.where(self.timegrid==tt)[0][0] 
                Ti = self.timegrid.size
                rev_ti = Ti- ti     
                #print(rev_ti) 
                grad_ln_ro = self.density_estimation(ti,rev_ti) #density estimation of forward particles  
                
                if ti==1: 
                  #print(rev_ti,rev_ti-1)
                  self.B[:,:,rev_ti-1] = (self.B[:,:,rev_ti] - self.f(self.B[:,:,rev_ti], self.timegrid[rev_ti])*self.dt + self.dt*self.g**2*grad_ln_ro \
                                         + (self.g)*np.random.normal(loc = 0.0, scale = np.sqrt(self.dt),size=(self.dim,self.N)) )% (2*np.pi)
                else:
                  grad_ln_b = self.bw_density_estimation(ti,rev_ti)
                  self.B[:,:,rev_ti-1] = (self.B[:,:,rev_ti] -\
                                        ( self.f(self.B[:,:,rev_ti], self.timegrid[rev_ti])- self.g**2*grad_ln_ro +0.5*self.g**2 * grad_ln_b )*self.dt)% (2*np.pi)
                
        
            
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
        
        lnthsc = 2*np.std(self.B[:,:,ti],axis=1)
  
        bnds = np.zeros((self.dim,2))
        for ii in range(self.dim):
            bnds[ii] = [max(np.min(self.Z[ii,:,ti]),np.min(self.B[ii,:,ti])),min(np.max(self.Z[ii,:,ti]),np.max(self.B[ii,:,ti]))]
     
        #sum_bnds = np.sum(bnds)
      
        Sxx = np.array([ np.random.uniform(low=bnd[0],high=bnd[1],size=(self.N_sparse)) for bnd in bnds ] )
        #u_t = np.zeros(grid_x.shape)
        # for di in range(self.dim):  
        #     u_t[di] = score_function_multid_seperate(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T) \
        #              - score_function_multid_seperate(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc,which_dim=di+1, kern=self.kern)(grid_x.T)
        
        u_t = (score_function_multid_seperate_all_dims(self.B[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc, kern=self.kern)(grid_x.T) \
                     - score_function_multid_seperate_all_dims(self.Z[:,:,ti].T,Sxx.T,func_out= True,C=0.001,which=1,l=lnthsc, kern=self.kern)(grid_x.T) ).T
        #np.testing.assert_allclose(u_t2, u_t)
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




script_name = sys.argv[0] 
dim = int(sys.argv[1])  #number of oscillators
input_num = int(sys.argv[2]) ##task id
noise = int(sys.argv[3])
repetition = int(sys.argv[4]) ##here I repeat from outside different instances with same parameters
fre = int(sys.argv[5])

freqs = [0.25, 0.5, 1]
freq = freqs[fre-1]

Ks   = np.linspace(0,10,11) 

K = Ks[input_num-1]








def fsquare(x): ##apply dot product for every 2dim vector in the array f^2
  x = np.atleast_2d( x )
  #return np.apply_along_axis(lambda xi: np.dot(f(xi),f(xi.T) ) , 1, x)
  return np.linalg.norm(f(x.T),axis=0)** 2



###order parameter
def R_t(x):
  return (1/dim)* np.abs( np.sum(np.exp( 1j* x)) )

@numba.njit(parallel=False,fastmath=True)
def U(x): #constraint
  ### 1- order parameter
  #difs = (x-x[:,None] ) %(2*np.pi)
  #print(difs.shape)
  #return -(1/dim)* np.array([np.sum(np.triu(difs[:,:,ii])) for ii in range(difs.shape[-1]) ])
  return -1*(1-(1/dim)* np.abs( np.sum(np.exp( 1j* x),axis=0) ) )





h = 0.001 #sim_prec
t_start = 0.

 
t1 = 0
t2 = 0.5
T = t2-t1
timegrid = np.arange(0,T+h/2,h)
if dim==6:
    N = 3000#0#0#0#0#2000
elif dim==10:
    N=4000
if noise==0:
    g = 0.5
elif noise==1:
    g = 1
k = timegrid.size
M = 300#0


y2 = np.ones(dim)  

sigmas = np.array([0.1,0.5,1])
rep_bridge = 1   #10 different bridge instances for every setting
reps = 20 ##instanses for stochastic path evaluation of each bridge
   
### first run in kura had scale 0.5
### in kura_wide0 I have scale 1
random.seed(input_num*100+repetition)
y0 = np.random.normal( loc=3, scale=1,size=dim )  
ws = np.ones(dim)
ws[:round(dim/2)] = freq
ws[round(dim/2):] = -freq



def f(x,t=0):
    #print(x.shape)
    difs = -np.sin(x-x[:,None] )
    #print(difs)
    #print( (np.ones((1,dim)) @ difs).shape )
    dn = x.shape ## shape of input state
    if len(dn) ==1: ##if function is called for sinle point, set ni to 1
        
        xout = ws + (K/dim)* np.sum( difs, axis=0)
    else:
        ni = dn[1] ## else if functionn is called for an ensemble of points, set ni to that number   
    
        xout = np.tile(ws,(ni,1)).T + (K/dim)* np.sum( difs, axis=0)
    
    return xout

save_dir = '/work/maoutsa/kura_wide0/'
naming = 'Delta%d_N_syst_Kuramoto_inh_k_%d_gi_%d_N_%d_M_%d_repetition_%d_fr_%.3f'%(dim, K,noise,N, M,repetition, freq)
bridg2d = BRIDGE_ND_reweight(t1,t2,y0,y2,f,g,N,k,M,reweight=True, U=U,dens_est='nonparametric',reject=False,plotting=True,kern='RBF',dt=h)

Fnon = np.zeros((dim,timegrid.size,reps))  * np.nan
Fcont = np.zeros((dim,timegrid.size,reps))  * np.nan
used_us = np.zeros((dim,timegrid.size,reps))
Rttcont = np.zeros((timegrid.size,reps))
Rttnon = np.zeros((timegrid.size,reps))
for repi in range(reps):
    with open(save_dir+naming+"output.txt", "a") as fi:
        print('>>>>>>>>>>>>>>>>>>>',repi, file=fi)
    for ti,tt in enumerate(timegrid[:-1]):
        ### ti is local time, tti is global time - both are time indices
        tti =  ti  ## index of timepoint in the initial timegrid- the real time axis
        #print(tti)
        if ti==0:
            Fcont[:,tti,repi] = y0 
            Fnon[:,tti,repi] = y0 
        else:        
            ###use previous grad log for current step
            if ti== bridg2d.timegrid.size-1: ##this is the timegrid_sub size
                uu = bridg2d.calculate_u(np.atleast_2d(Fcont[:,tti-1,repi]).T,ti-1).reshape(-1)
            else:
                uu = bridg2d.calculate_u(np.atleast_2d(Fcont[:,tti-1,repi]).T,ti).reshape(-1)
            
            used_us[:,tti,repi] = uu#.T
            Fcont[:,tti,repi] =  ( Fcont[:,tti-1,repi]+ h* f(Fcont[:,tti-1,repi])+h*g**2 *uu+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim)) )% (2*np.pi)
            Fnon[:,tti,repi] =  ( Fnon[:,tti-1,repi]+ h* f(Fnon[:,tti-1,repi])+(g)*np.random.normal(loc = 0.0, scale = np.sqrt(h),size=(dim)) )% (2*np.pi)
            Rttcont[ti,repi] = R_t(Fcont[:,ti,repi]) 
            Rttnon[ti,repi] = R_t(Fnon[:,ti,repi])
            
    with open(save_dir+naming+"output.txt", "a") as fi:
        print(Rttcont[:,repi], file=fi)
            
            
            
to_save = dict()
to_save['Fcont'] = Fcont
to_save['Rttcont'] = Rttcont
to_save['Fnon'] = Fnon
to_save['Rttnon'] = Rttnon
to_save['timegrid'] = timegrid
to_save['B'] = bridg2d.B
to_save['Z'] = bridg2d.Z
to_save['K'] = K
to_save['ws'] = ws
to_save['g'] = g
to_save['N'] = N
to_save['M'] = M
to_save['used_us'] = used_us
to_save['y0'] = y0
to_save['repetition'] = repetition
to_save['freq'] = freq
pickle.dump(to_save, open(save_dir+naming+'.dat', "wb"))
